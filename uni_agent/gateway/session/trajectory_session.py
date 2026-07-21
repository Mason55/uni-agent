"""Transport-independent trajectory session state and lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from uni_agent.gateway.session.types import (
    CapturedGeneration,
    CaptureReceipt,
    CaptureTransaction,
    CaptureUsage,
    InternalGenerationRequest,
    SessionHandle,
    Trajectory,
    mutable_capture_mapping,
    mutable_capture_value,
    normalize_finish_reason,
)

if TYPE_CHECKING:
    from uni_agent.gateway.session.codec import MessageCodec

_EMPTY_PREFIX_HASH = hashlib.sha256(b"uni-agent-prefix-v1\0empty").hexdigest()


class SessionPhase(str, Enum):
    """Lifecycle state for a trajectory session."""

    ACTIVE = "ACTIVE"
    FINALIZED = "FINALIZED"
    ABORTED = "ABORTED"


class SessionLifecycleError(RuntimeError):
    """Raised when an operation is invalid for the current session phase."""


@dataclass
class TrajectoryBuffer:
    """Mutable token buffer for one active trajectory chain.

    ``routed_experts`` represents the backend's latest full-context routing
    value. Continuations replace it instead of accumulating it token-by-token.
    """

    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    routed_experts: Any | None = None


@dataclass
class ChainState:
    """One active linear trajectory chain in a session."""

    chain_id: int
    message_history: list[dict[str, Any]]
    message_tip_hash: str
    active_tool_schemas: list[dict[str, Any]] | None
    buffer: TrajectoryBuffer
    image_data: list[Any] | None
    video_data: list[Any] | None
    updated_seq: int


@dataclass
class MaterializedChain:
    """A closed chain plus the ordering metadata needed at finalize."""

    trajectory: Trajectory
    order_seq: int


@dataclass
class _PreparedSingleTurnCapture:
    """Session-private state retained between capture preparation and commit."""

    context_ids: tuple[int, ...]
    sampling_params: dict[str, Any]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    image_data: list[Any] | None
    video_data: list[Any] | None


class _CaptureTransaction:
    """Opaque asynchronous transaction for one externally-owned model call."""

    def __init__(self, session: TrajectorySession, prepared: _PreparedSingleTurnCapture):
        self._session = session
        self._prepared = prepared
        self._commit_lock = asyncio.Lock()
        self._receipt: CaptureReceipt | None = None

    @property
    def context_ids(self) -> tuple[int, ...]:
        """Return the token IDs prepared for upstream inference."""
        return self._prepared.context_ids

    @property
    def sampling_params(self) -> dict[str, Any]:
        """Return a copy of the merged sampling parameters."""
        return deepcopy(self._prepared.sampling_params)

    @property
    def image_data(self) -> tuple[Any, ...] | None:
        """Return optional image inputs prepared for upstream inference."""
        if self._prepared.image_data is None:
            return None
        return tuple(self._prepared.image_data)

    @property
    def video_data(self) -> tuple[Any, ...] | None:
        """Return optional video inputs prepared for upstream inference."""
        if self._prepared.video_data is None:
            return None
        return tuple(self._prepared.video_data)

    async def commit(self, generation: CapturedGeneration) -> CaptureReceipt:
        """Commit externally-produced rollout truth and return its receipt."""
        async with self._commit_lock:
            if self._receipt is not None:
                raise RuntimeError("capture transaction is already committed")
            self._receipt = await self._session._commit_single_turn_capture(self._prepared, generation)
            return self._receipt


class TrajectorySession:
    """Own trajectory state and lifecycle without inference or transport.

    The session can be constructed independently from a model codec or backend.
    Rollout adapters compose it with request preparation and generation while
    reusing this single state owner.
    """

    def __init__(
        self,
        handle: SessionHandle,
        codec: MessageCodec | None = None,
        *,
        response_length: int | None = None,
        sampling_params: dict[str, Any] | None = None,
    ):
        """Create an empty active trajectory session."""
        if response_length is not None and response_length <= 0:
            raise ValueError(f"response_length must be positive when set, got {response_length}")
        self.handle = handle
        self._codec = codec
        self._response_length = response_length
        self._sampling_params = deepcopy(sampling_params or {})
        self.active_chains: list[ChainState] = []
        self.materialized_chains: list[MaterializedChain] = []
        self.reserved_chain_ids: set[int] = set()
        self._next_chain_id = 1
        self._order_seq = 0
        self.reward_info: dict[str, Any] = {}
        self.phase = SessionPhase.ACTIVE
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.request_lock = asyncio.Lock()

    async def capture(self, request: InternalGenerationRequest) -> CaptureTransaction:
        """Prepare an initial passive-capture transaction for external rollout."""
        async with self.request_lock:
            if self.phase != SessionPhase.ACTIVE:
                raise SessionLifecycleError(f"Session {self.handle.session_id} is {self.phase.value.lower()}")
            if self._codec is None:
                raise RuntimeError("TrajectorySession.capture requires a message codec")
            if self.active_chains:
                raise RuntimeError("single-turn passive capture does not support continuation")

            messages = deepcopy(request["messages"])
            tools = deepcopy(request["tools"])
            image_data, video_data = await self._codec.extract_multi_modal_data(messages)
            context_ids = tuple(
                self._codec.encode_full(
                    messages,
                    tools=tools,
                    image_data=image_data,
                    video_data=video_data,
                )
            )
            merged_sampling_params = deepcopy(self._sampling_params)
            merged_sampling_params.update(deepcopy(request["sampling_params"]))
            if self._response_length is not None:
                merged_sampling_params["max_tokens"] = min(
                    merged_sampling_params.get("max_tokens", self._response_length),
                    self._response_length,
                )
            prepared = _PreparedSingleTurnCapture(
                context_ids=context_ids,
                sampling_params=merged_sampling_params,
                messages=messages,
                tools=tools,
                image_data=self._copy_media_list(image_data),
                video_data=self._copy_media_list(video_data),
            )
            return _CaptureTransaction(self, prepared)

    async def _commit_single_turn_capture(
        self,
        prepared: _PreparedSingleTurnCapture,
        generation: CapturedGeneration,
    ) -> CaptureReceipt:
        async with self.request_lock:
            if self.phase != SessionPhase.ACTIVE:
                raise SessionLifecycleError(f"Session {self.handle.session_id} is {self.phase.value.lower()}")
            if self.active_chains:
                raise RuntimeError("single-turn passive capture does not support continuation")

            assistant_message = mutable_capture_mapping(generation.assistant_message)
            message_history = [*prepared.messages, assistant_message]
            message_prefix_hashes = self._extend_message_prefix_hashes([], message_history)
            chain_id = self._allocate_chain_id()
            order_seq = self._next_order_seq()
            response_ids = tuple(generation.completion_ids)
            response_logprobs = tuple(generation.completion_logprobs)
            self.active_chains.append(
                ChainState(
                    chain_id=chain_id,
                    message_history=message_history,
                    message_tip_hash=message_prefix_hashes[-1],
                    active_tool_schemas=prepared.tools,
                    buffer=TrajectoryBuffer(
                        prompt_ids=list(generation.prompt_ids),
                        response_ids=list(response_ids),
                        response_mask=[1] * len(response_ids),
                        response_logprobs=list(response_logprobs),
                        routed_experts=mutable_capture_value(generation.routed_experts),
                    ),
                    image_data=self._copy_media_list(prepared.image_data),
                    video_data=self._copy_media_list(prepared.video_data),
                    updated_seq=order_seq,
                )
            )
            self._touch()
            return CaptureReceipt(
                prompt_ids=tuple(generation.prompt_ids),
                response_ids=response_ids,
                response_mask=(1,) * len(response_ids),
                response_logprobs=response_logprobs,
                chain_id=chain_id,
                turn_id=1,
                usage=CaptureUsage(
                    prompt_tokens=len(generation.prompt_ids),
                    completion_tokens=len(response_ids),
                ),
                assistant_message=assistant_message,
                finish_reason=normalize_finish_reason(generation.stop_reason),
                routed_experts=generation.routed_experts,
                routing_metadata=generation.routing_metadata,
            )

    async def set_reward_info(self, reward_info: dict[str, Any] | None = None) -> None:
        """Store session-level reward metadata without closing the session."""
        async with self.request_lock:
            if self.phase != SessionPhase.ACTIVE:
                raise SessionLifecycleError(f"Session {self.handle.session_id} is {self.phase.value.lower()}")
            if reward_info is not None:
                self.reward_info = dict(reward_info)
            self._touch()

    async def finalize(self) -> list[Trajectory]:
        """Close the session and return ordered trajectories with rewards."""
        async with self.request_lock:
            if self.phase == SessionPhase.ABORTED:
                raise SessionLifecycleError(f"Session {self.handle.session_id} is aborted")
            if self.phase == SessionPhase.FINALIZED:
                raise SessionLifecycleError(f"Session {self.handle.session_id} is finalized")
            self._touch()
            self._materialize_active_chains()
            self.reserved_chain_ids.clear()
            self.phase = SessionPhase.FINALIZED
            self._touch()
            ordered_trajectories = [
                materialized.trajectory
                for materialized in sorted(self.materialized_chains, key=lambda chain: chain.order_seq)
            ]
            return [replace(trajectory, reward_info=dict(self.reward_info)) for trajectory in ordered_trajectories]

    async def abort(self) -> None:
        """Abort the session, discard trajectory state, and reject future work."""
        async with self.request_lock:
            if self.phase == SessionPhase.ABORTED:
                return
            if self.phase == SessionPhase.FINALIZED:
                raise SessionLifecycleError(f"Session {self.handle.session_id} is finalized")
            self.phase = SessionPhase.ABORTED
            self.active_chains = []
            self.materialized_chains = []
            self.reserved_chain_ids.clear()
            self._touch()

    def snapshot_state(self) -> dict[str, Any]:
        """Return a JSON-serializable lifecycle and trajectory-state snapshot."""
        return {
            "session_id": self.handle.session_id,
            "phase": self.phase.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "num_trajectories": len(self.materialized_chains),
            "has_active_trajectory": bool(self.active_chains),
            "num_active_chains": len(self.active_chains),
            "active_chain_ids": [chain.chain_id for chain in self.active_chains],
            "active_chain_tip_hashes": {chain.chain_id: chain.message_tip_hash for chain in self.active_chains},
        }

    def _copy_trajectory_buffer(self, buffer: TrajectoryBuffer) -> TrajectoryBuffer:
        return TrajectoryBuffer(
            prompt_ids=list(buffer.prompt_ids),
            response_ids=list(buffer.response_ids),
            response_mask=list(buffer.response_mask),
            response_logprobs=list(buffer.response_logprobs),
            routed_experts=buffer.routed_experts,
        )

    def _copy_chain_media(self, chain: ChainState) -> tuple[list[Any] | None, list[Any] | None]:
        return self._copy_media_list(chain.image_data), self._copy_media_list(chain.video_data)

    def _copy_media_list(self, media: list[Any] | None) -> list[Any] | None:
        # Copy only the container; media payloads may not be deepcopyable.
        return list(media) if media is not None else None

    def _extend_message_prefix_hashes(
        self,
        existing_prefix_hashes: list[str],
        new_messages: list[dict[str, Any]],
    ) -> list[str]:
        prefix_hashes = list(existing_prefix_hashes)
        previous_prefix_hash = prefix_hashes[-1] if prefix_hashes else _EMPTY_PREFIX_HASH
        for message in new_messages:
            message_hash = self._compute_message_hash(message)
            prefix_hash = hashlib.sha256(
                b"uni-agent-prefix-v1\0" + previous_prefix_hash.encode("ascii") + b"\0" + message_hash.encode("ascii")
            ).hexdigest()
            prefix_hashes.append(prefix_hash)
            previous_prefix_hash = prefix_hash
        return prefix_hashes

    def _compute_message_hash(self, message: dict[str, Any]) -> str:
        if self._codec is None:
            raise RuntimeError("message hashing requires a message codec")
        canonical = self._codec.canonicalize_message_for_prefix_comparison(message)
        canonical_json = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(b"uni-agent-message-v1\0" + canonical_json).hexdigest()

    def _find_active_chain(self, chain_id: int) -> tuple[int, ChainState]:
        for index, chain in enumerate(self.active_chains):
            if chain.chain_id == chain_id:
                return index, chain
        raise RuntimeError(f"active chain {chain_id} not found")

    def _allocate_chain_id(self) -> int:
        chain_id = self._next_chain_id
        self._next_chain_id += 1
        return chain_id

    async def _release_chain_reservation(self, chain_id: int) -> None:
        async with self.request_lock:
            self.reserved_chain_ids.discard(chain_id)

    def _next_order_seq(self) -> int:
        self._order_seq += 1
        return self._order_seq

    def _materialize_active_chains(self) -> None:
        for chain in self.active_chains:
            self.materialized_chains.append(
                MaterializedChain(
                    trajectory=self._build_materialized_trajectory(chain=chain),
                    order_seq=chain.updated_seq,
                )
            )
        self.active_chains = []

    def _build_materialized_trajectory(
        self,
        *,
        chain: ChainState,
        extra_fields: dict[str, Any] | None = None,
    ) -> Trajectory:
        response_logprobs = None
        if chain.buffer.response_logprobs and len(chain.buffer.response_logprobs) == len(chain.buffer.response_ids):
            response_logprobs = list(chain.buffer.response_logprobs)
        return Trajectory(
            prompt_ids=list(chain.buffer.prompt_ids),
            response_ids=list(chain.buffer.response_ids),
            response_mask=list(chain.buffer.response_mask),
            response_logprobs=response_logprobs,
            reward_info={},
            num_turns=self._count_chat_turns(chain.message_history),
            routed_experts=chain.buffer.routed_experts,
            multi_modal_data=self._build_multi_modal_trajectory_data(chain.image_data, chain.video_data),
            extra_fields=dict(extra_fields) if extra_fields else {},
        )

    def _count_chat_turns(self, message_history: list[dict[str, Any]]) -> int:
        return sum(1 for message in message_history if message.get("role") in ("user", "assistant")) + 1

    def _build_multi_modal_trajectory_data(
        self,
        image_data: list[Any] | None,
        video_data: list[Any] | None,
    ) -> dict[str, Any] | None:
        multi_modal_data: dict[str, Any] = {}
        if image_data:
            multi_modal_data["images"] = list(image_data)
        if video_data:
            multi_modal_data["videos"] = list(video_data)
        return multi_modal_data or None

    def _touch(self) -> None:
        self.updated_at = time.time()
