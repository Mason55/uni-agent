"""Transport-independent trajectory session state and lifecycle."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from uni_agent.gateway.session.types import SessionHandle, Trajectory


class SessionPhase(str, Enum):
    """Lifecycle state for a trajectory session."""

    ACTIVE = "ACTIVE"
    FINALIZED = "FINALIZED"
    ABORTED = "ABORTED"


class SessionLifecycleError(RuntimeError):
    """Raised when an operation is invalid for the current session phase."""


@dataclass
class TrajectoryBuffer:
    """Mutable token buffer for one active trajectory chain."""

    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)


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


class TrajectorySession:
    """Own trajectory state and lifecycle without owning inference or transport.

    The session can be constructed independently from a model codec or backend.
    Rollout adapters may extend it with request preparation and generation while
    reusing this single state owner.
    """

    def __init__(self, handle: SessionHandle):
        """Create an empty active trajectory session."""
        self.handle = handle
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

    async def set_reward_info(self, reward_info: dict[str, Any] | None = None) -> None:
        """Store session-level reward metadata without closing the session."""
        async with self.request_lock:
            if self.phase != SessionPhase.ACTIVE:
                raise SessionLifecycleError(
                    f"Session {self.handle.session_id} is {self.phase.value.lower()}"
                )
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
        )

    def _copy_chain_media(self, chain: ChainState) -> tuple[list[Any] | None, list[Any] | None]:
        return self._copy_media_list(chain.image_data), self._copy_media_list(chain.video_data)

    def _copy_media_list(self, media: list[Any] | None) -> list[Any] | None:
        # Copy only the container; media payloads may not be deepcopyable.
        return list(media) if media is not None else None

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
