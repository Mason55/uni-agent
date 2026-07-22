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
class _PreparedCapture:
    """Session-private state retained between capture preparation and commit."""

    buffer: TrajectoryBuffer
    context_ids: tuple[int, ...]
    sampling_params: dict[str, Any]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    image_data: list[Any] | None
    video_data: list[Any] | None
    chain_id: int | None
    turn_id: int
    incoming_message_prefix_hashes: list[str]
    length_exhausted_trajectory: Trajectory | None = None


class _CaptureTransaction:
    """Opaque asynchronous transaction for one externally-owned model call."""

    def __init__(
        self,
        session: TrajectorySession,
        prepared: _PreparedCapture,
        reserved_chain_id: int | None,
    ):
        self._session = session
        self._prepared = prepared
        self._reserved_chain_id = reserved_chain_id
        self._commit_lock = asyncio.Lock()
        self._receipt: CaptureReceipt | None = None
        self._closed = False

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

    @property
    def length_exhausted(self) -> bool:
        """Return whether the session budget forbids another model token."""
        return self._prepared.length_exhausted_trajectory is not None

    async def __aenter__(self) -> _CaptureTransaction:
        """Enter the transaction scope used around externally-owned rollout."""
        if self._closed:
            raise RuntimeError("capture transaction is closed")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        """Release any uncommitted reservation on scope exit."""
        await self.rollback()
        return False

    async def commit(self, generation: CapturedGeneration) -> CaptureReceipt:
        """Commit externally-produced rollout truth and return its receipt."""
        async with self._commit_lock:
            if self._receipt is not None:
                raise RuntimeError("capture transaction is already committed")
            if self._closed:
                raise RuntimeError("capture transaction is closed")
            try:
                self._receipt = await self._session._commit_capture(self._prepared, generation)
                return self._receipt
            finally:
                await self._close()

    async def rollback(self) -> None:
        """Discard this capture and release its chain reservation."""
        async with self._commit_lock:
            if self._closed:
                return
            await self._close()

    async def _close(self) -> None:
        reserved_chain_id = self._reserved_chain_id
        self._reserved_chain_id = None
        self._closed = True
        if reserved_chain_id is None:
            return
        release_task = asyncio.create_task(self._session._release_chain_reservation(reserved_chain_id))
        cancelled = False
        while not release_task.done():
            try:
                await asyncio.shield(release_task)
            except asyncio.CancelledError:
                cancelled = True
        release_task.result()
        if cancelled:
            raise asyncio.CancelledError


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
        self._reserved_chain_ids: set[int] = set()
        self._next_chain_id = 1
        self._order_seq = 0
        self.reward_info: dict[str, Any] = {}
        self.phase = SessionPhase.ACTIVE
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.request_lock = asyncio.Lock()

    async def capture(self, request: InternalGenerationRequest) -> CaptureTransaction:
        """Prepare a passive-capture transaction for external rollout."""
        async with self.request_lock:
            if self.phase != SessionPhase.ACTIVE:
                raise SessionLifecycleError(f"Session {self.handle.session_id} is {self.phase.value.lower()}")
            if self._codec is None:
                raise RuntimeError("TrajectorySession.capture requires a message codec")
            messages = deepcopy(request["messages"])
            tools = deepcopy(request["tools"])
            incoming_message_prefix_hashes = self._extend_message_prefix_hashes([], messages)
            selected_chain = self._select_chain(
                tools=tools,
                incoming_message_prefix_hashes=incoming_message_prefix_hashes,
            )
            sampling_params = deepcopy(self._sampling_params)
            sampling_params.update(deepcopy(request["sampling_params"]))

            if selected_chain is None:
                image_data, video_data = await self._codec.extract_multi_modal_data(messages)
                prompt_ids = self._codec.encode_full(
                    messages,
                    tools=tools,
                    image_data=image_data,
                    video_data=video_data,
                )
                buffer = TrajectoryBuffer(prompt_ids=prompt_ids)
                chain_id = None
                turn_id = 1
            else:
                buffer = self._copy_trajectory_buffer(selected_chain.buffer)
                image_data, video_data = self._copy_chain_media(selected_chain)
                chain_id = selected_chain.chain_id
                turn_id = sum(
                    1 for message in selected_chain.message_history if message.get("role") == "assistant"
                ) + 1
                incremental_messages = messages[len(selected_chain.message_history) :]
                incremental_ids: list[int] = []
                new_image_data = None
                new_video_data = None
                already_exhausted = (
                    self._response_length is not None and len(buffer.response_mask) >= self._response_length
                )
                if incremental_messages and not already_exhausted:
                    new_image_data, new_video_data = await self._codec.extract_multi_modal_data(incremental_messages)
                    incremental_ids = self._codec.encode_incremental(
                        incremental_messages,
                        image_data=new_image_data,
                        video_data=new_video_data,
                    )
                if already_exhausted or (
                    self._response_length is not None
                    and len(buffer.response_mask) + len(incremental_ids) >= self._response_length
                ):
                    prepared = _PreparedCapture(
                        buffer=buffer,
                        context_ids=tuple(buffer.prompt_ids + buffer.response_ids),
                        sampling_params={},
                        messages=messages,
                        tools=tools,
                        image_data=image_data,
                        video_data=video_data,
                        chain_id=chain_id,
                        turn_id=turn_id,
                        incoming_message_prefix_hashes=incoming_message_prefix_hashes,
                        length_exhausted_trajectory=self._build_materialized_trajectory(
                            chain=selected_chain,
                            extra_fields={"materialization_reason": "max_response_length"},
                        ),
                    )
                    return self._open_capture_transaction(prepared)
                buffer.response_ids.extend(incremental_ids)
                buffer.response_mask.extend([0] * len(incremental_ids))
                if sampling_params.get("logprobs", False):
                    buffer.response_logprobs.extend([0.0] * len(incremental_ids))
                if new_image_data:
                    if image_data is None:
                        image_data = []
                    image_data.extend(new_image_data)
                if new_video_data:
                    if video_data is None:
                        video_data = []
                    video_data.extend(new_video_data)

            context_ids = tuple(buffer.prompt_ids + buffer.response_ids)
            remaining_response_budget = (
                self._response_length - len(buffer.response_mask) if self._response_length is not None else None
            )
            if remaining_response_budget is not None:
                sampling_params["max_tokens"] = min(
                    sampling_params.get("max_tokens", remaining_response_budget),
                    remaining_response_budget,
                )
            prepared = _PreparedCapture(
                buffer=buffer,
                context_ids=context_ids,
                sampling_params=sampling_params,
                messages=messages,
                tools=tools,
                image_data=self._copy_media_list(image_data),
                video_data=self._copy_media_list(video_data),
                chain_id=chain_id,
                turn_id=turn_id,
                incoming_message_prefix_hashes=list(incoming_message_prefix_hashes),
            )
            return self._open_capture_transaction(prepared)

    def _open_capture_transaction(self, prepared: _PreparedCapture) -> _CaptureTransaction:
        reserved_chain_id = prepared.chain_id
        if reserved_chain_id is not None:
            self._reserve_chain(reserved_chain_id)
        return _CaptureTransaction(self, prepared, reserved_chain_id)

    async def _commit_capture(
        self,
        prepared: _PreparedCapture,
        generation: CapturedGeneration,
    ) -> CaptureReceipt:
        async with self.request_lock:
            if self.phase != SessionPhase.ACTIVE:
                raise SessionLifecycleError(f"Session {self.handle.session_id} is {self.phase.value.lower()}")

            if prepared.length_exhausted_trajectory is not None:
                if prepared.chain_id is None:
                    raise RuntimeError("length-exhausted capture is missing its chain")
                chain_index, _ = self._find_active_chain(prepared.chain_id)
                self.materialized_chains.append(
                    MaterializedChain(
                        trajectory=prepared.length_exhausted_trajectory,
                        order_seq=self._next_order_seq(),
                    )
                )
                del self.active_chains[chain_index]
                self._touch()
                return CaptureReceipt(
                    prompt_ids=tuple(generation.prompt_ids),
                    response_ids=(),
                    response_mask=(),
                    response_logprobs=(),
                    chain_id=prepared.chain_id,
                    turn_id=prepared.turn_id,
                    usage=CaptureUsage(prompt_tokens=len(generation.prompt_ids), completion_tokens=0),
                    assistant_message={"role": "assistant", "content": ""},
                    finish_reason="length",
                    routed_experts=generation.routed_experts,
                    routing_metadata=generation.routing_metadata,
                )

            assistant_message = mutable_capture_mapping(generation.assistant_message)
            message_history = [*prepared.messages, assistant_message]
            message_prefix_hashes = self._extend_message_prefix_hashes(
                prepared.incoming_message_prefix_hashes,
                [assistant_message],
            )
            order_seq = self._next_order_seq()
            response_ids = tuple(generation.completion_ids)
            response_logprobs = tuple(generation.completion_logprobs)
            if prepared.chain_id is None:
                prepared.buffer.prompt_ids = list(generation.prompt_ids)
            prepared.buffer.response_ids.extend(response_ids)
            prepared.buffer.response_mask.extend([1] * len(response_ids))
            prepared.buffer.response_logprobs.extend(response_logprobs)
            if generation.routed_experts is not None:
                prepared.buffer.routed_experts = mutable_capture_value(generation.routed_experts)
            if prepared.chain_id is None:
                chain_id = self._allocate_chain_id()
                chain_index = None
            else:
                chain_id = prepared.chain_id
                chain_index, _ = self._find_active_chain(chain_id)
            next_chain = ChainState(
                chain_id=chain_id,
                message_history=message_history,
                message_tip_hash=message_prefix_hashes[-1],
                active_tool_schemas=prepared.tools,
                buffer=prepared.buffer,
                image_data=self._copy_media_list(prepared.image_data),
                video_data=self._copy_media_list(prepared.video_data),
                updated_seq=order_seq,
            )
            if chain_index is None:
                self.active_chains.append(next_chain)
            else:
                self.active_chains[chain_index] = next_chain
            self._touch()
            return CaptureReceipt(
                prompt_ids=tuple(generation.prompt_ids),
                response_ids=response_ids,
                response_mask=(1,) * len(response_ids),
                response_logprobs=response_logprobs,
                chain_id=chain_id,
                turn_id=prepared.turn_id,
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
            self._reserved_chain_ids.clear()
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
            self._reserved_chain_ids.clear()
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

    def _select_chain(
        self,
        *,
        tools: list[dict[str, Any]] | None,
        incoming_message_prefix_hashes: list[str],
    ) -> ChainState | None:
        candidates = [
            chain
            for chain in self.active_chains
            if not self._is_chain_reserved(chain.chain_id)
            and chain.active_tool_schemas == tools
            and self._is_chain_prefix_hash_match(
                chain=chain,
                incoming_message_prefix_hashes=incoming_message_prefix_hashes,
            )
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda chain: (len(chain.message_history), chain.updated_seq, chain.chain_id))

    def _is_chain_prefix_hash_match(
        self,
        *,
        chain: ChainState,
        incoming_message_prefix_hashes: list[str],
    ) -> bool:
        history_len = len(chain.message_history)
        if history_len > len(incoming_message_prefix_hashes):
            return False
        if history_len == 0:
            return True
        return chain.message_tip_hash == incoming_message_prefix_hashes[history_len - 1]

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
            self._discard_chain_reservation(chain_id)

    def _reserve_chain(self, chain_id: int) -> None:
        self._reserved_chain_ids.add(chain_id)

    def _discard_chain_reservation(self, chain_id: int) -> None:
        self._reserved_chain_ids.discard(chain_id)

    def _is_chain_reserved(self, chain_id: int) -> bool:
        return chain_id in self._reserved_chain_ids

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
