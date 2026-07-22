"""Per-session gateway state, generation envelope, and lifecycle handling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException

from uni_agent.gateway.session.codec import MessageCodec
from uni_agent.gateway.session.trajectory_session import (
    ChainState,
    MaterializedChain,
    SessionPhase,
    TrajectoryBuffer,
    TrajectorySession,
)
from uni_agent.gateway.session.types import InternalGenerationRequest, SessionHandle, Trajectory


@dataclass
class EncodedData:
    """Session-private data prepared before backend generation.

    The session uses this as the handoff between input preparation, backend
    generation, and the commit step. It is not an actor/runtime API.

    Attributes:
        buffer: Working trajectory buffer that becomes active only after commit.
        context_ids: Token IDs sent to the inference backend.
        sampling_params: Sampling params after request merge and budget clamp.
        messages: Normalized request messages snapshotted for commit.
        tools: Tool schemas used for both encoding and response decoding.
        image_data: Image inputs carried into backend generation and trajectory
            materialization.
        video_data: Video inputs carried into backend generation and trajectory
            materialization.
        length_exhausted_trajectory: Materialized trajectory for a length-budget
            early return, or ``None`` on the normal path.
        chain_id: Selected active chain id, or ``None`` when commit should append
            a new chain.
        incoming_message_prefix_hashes: Stable prefix hashes for the normalized
            request history.
    """

    buffer: TrajectoryBuffer
    context_ids: list[int]
    sampling_params: dict[str, Any]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    image_data: list[Any] | None
    video_data: list[Any] | None
    length_exhausted_trajectory: Trajectory | None
    chain_id: int | None
    incoming_message_prefix_hashes: list[str] = field(default_factory=list)


@dataclass
class GenerationOutcome:
    """Business result returned by ``GatewaySession.run_generation``.

    The session emits this instead of an HTTP response dict. ``_GatewayActor``
    passes it to the provider adapter for wire response serialization.

    Attributes:
        assistant_msg: Decoded assistant message, or an empty assistant message
            for length-exhausted early returns.
        finish_reason: Finish reason returned to the actor for serialization.
        prompt_tokens: Number of context tokens sent to the backend.
        completion_tokens: Number of generated response tokens.
    """

    assistant_msg: dict[str, Any]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class GatewaySession:
    """Behavior-bearing state container for one gateway session.

    ``_GatewayActor`` owns instances of this class, calls ``run_generation`` for
    chat requests, and delegates lifecycle operations here. The session owns the
    conversation state and trajectory materialization, while the actor owns
    HTTP routing and provider response serialization.
    """

    def __init__(
        self,
        handle: SessionHandle,
        codec: MessageCodec,
        *,
        prompt_length: int | None = None,
        response_length: int | None = None,
        sampling_params: dict[str, Any] | None = None,
    ):
        """Create an active session bound to a handle and model codec."""
        self._trajectory_session = TrajectorySession(
            handle,
            codec,
            response_length=response_length,
            sampling_params=sampling_params,
        )
        self._codec = codec
        # Provider adapters merge these trusted defaults before calling the
        # session; the response budget is enforced here during preparation.
        self._prompt_length = prompt_length
        self._response_length = response_length
        self._sampling_params = dict(sampling_params or {})

    @property
    def sampling_params(self) -> dict[str, Any]:
        """Return a copy of the trusted per-session sampling defaults."""
        return dict(self._sampling_params)

    @property
    def handle(self) -> SessionHandle:
        """Return the handle owned by the trajectory session."""
        return self._trajectory_session.handle

    @property
    def active_chains(self) -> list[ChainState]:
        """Expose active chains for backwards-compatible diagnostics."""
        return self._trajectory_session.active_chains

    @property
    def materialized_chains(self) -> list[MaterializedChain]:
        """Expose materialized chains for backwards-compatible diagnostics."""
        return self._trajectory_session.materialized_chains

    @property
    def reward_info(self) -> dict[str, Any]:
        """Expose reward metadata for backwards compatibility."""
        return self._trajectory_session.reward_info

    @property
    def phase(self) -> SessionPhase:
        """Return the trajectory-session lifecycle phase."""
        return self._trajectory_session.phase

    @property
    def created_at(self) -> float:
        """Return the trajectory-session creation time."""
        return self._trajectory_session.created_at

    @property
    def updated_at(self) -> float:
        """Return the trajectory-session last-update time."""
        return self._trajectory_session.updated_at

    @property
    def request_lock(self) -> asyncio.Lock:
        """Return the lock shared by rollout and trajectory lifecycle work."""
        return self._trajectory_session.request_lock

    async def run_generation(self, request: InternalGenerationRequest, backend) -> GenerationOutcome:
        """Run one provider-normalized generation request and return its business outcome.

        The backend is passed in for this call only; the session does not own the
        backend lifecycle. The actor/provider adapter has already lowered the
        wire payload to the internal canonical request; session never sees raw
        wire payloads. Protocol capability checks happen in the actor before
        this method, while backend errors are converted into HTTP exceptions
        here.
        """
        # Same-session requests overlap backend generation and commit in backend
        # completion order. The framework currently scores session_trajectories[-1]
        # and broadcasts that reward, so concurrent siblings share one reward target.
        reserved_chain_id: int | None = None
        try:
            async with self.request_lock:
                if self.phase != SessionPhase.ACTIVE:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Session {self.handle.session_id} is {self.phase.value.lower()}",
                    )
                # Prepare can touch codec and multimodal extractor state, so only
                # backend generation runs outside the session lock.
                encoded = await self._prepare_generation_inputs(request)
                if encoded.length_exhausted_trajectory is not None:
                    empty_msg = {"role": "assistant", "content": ""}
                    self._close_length_exhausted_chain(encoded)
                    self._touch()
                    return GenerationOutcome(
                        assistant_msg=empty_msg,
                        finish_reason="length",
                        prompt_tokens=len(encoded.context_ids),
                        completion_tokens=0,
                    )
                if encoded.chain_id is not None:
                    self._trajectory_session._reserve_chain(encoded.chain_id)
                    reserved_chain_id = encoded.chain_id

            try:
                output = await backend.generate(
                    request_id=self.handle.session_id,
                    prompt_ids=encoded.context_ids,
                    sampling_params=encoded.sampling_params,
                    image_data=encoded.image_data,
                    video_data=encoded.video_data,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"{e.__class__.__name__}: {e}") from e

            response_ids = list(output.token_ids)
            encoded.buffer.response_ids.extend(response_ids)
            encoded.buffer.response_mask.extend([1] * len(response_ids))
            if encoded.sampling_params.get("logprobs", False):
                if output.log_probs is None:
                    raise RuntimeError("backend omitted logprobs when requested")
                log_probs = list(output.log_probs)
                if len(log_probs) != len(response_ids):
                    raise RuntimeError(
                        "backend logprobs must align with token_ids: "
                        f"got {len(log_probs)} logprobs for {len(response_ids)} tokens"
                    )
                encoded.buffer.response_logprobs.extend(log_probs)

            # R3 router replay: the backend returns routing for the full context
            # it just prefilled (prompt + response so far + new tokens), so keep
            # the latest value; it supersedes prior turns. The framework aligns it
            # to input_ids when writing to TransferQueue.
            routed_experts = getattr(output, "routed_experts", None)
            if routed_experts is not None:
                encoded.buffer.routed_experts = routed_experts

            async with self.request_lock:
                if self.phase != SessionPhase.ACTIVE:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Session {self.handle.session_id} is {self.phase.value.lower()}",
                    )
                # Decode runs under request_lock so this session's prepare/commit and
                # decode stay serialized. It does not serialize decode across sessions,
                # which share the actor codec.
                assistant_msg, finish_reason = await self._codec.decode_response(
                    response_ids,
                    tools=encoded.tools,
                    stop_reason=output.stop_reason,
                )
                self._commit_generation_to_chain(encoded, assistant_msg)
                if reserved_chain_id is not None:
                    self._trajectory_session._discard_chain_reservation(reserved_chain_id)
                    reserved_chain_id = None
                self._touch()
                return GenerationOutcome(
                    assistant_msg=assistant_msg,
                    finish_reason=finish_reason,
                    prompt_tokens=len(encoded.context_ids),
                    completion_tokens=len(response_ids),
                )
        finally:
            if reserved_chain_id is not None:
                await asyncio.shield(self._release_chain_reservation(reserved_chain_id))

    async def set_reward_info(self, reward_info: dict[str, Any] | None = None) -> None:
        """Delegate reward metadata to the trajectory state owner."""
        await self._trajectory_session.set_reward_info(reward_info)

    async def finalize(self) -> list[Trajectory]:
        """Delegate finalization to the trajectory state owner."""
        return await self._trajectory_session.finalize()

    async def abort(self) -> None:
        """Delegate abort to the trajectory state owner."""
        await self._trajectory_session.abort()

    def snapshot_state(self) -> dict[str, Any]:
        """Delegate state inspection to the trajectory state owner."""
        return self._trajectory_session.snapshot_state()

    async def _prepare_generation_inputs(
        self,
        request: InternalGenerationRequest,
    ) -> EncodedData:
        messages = request["messages"]
        tools = request["tools"]
        sampling_params = dict(request["sampling_params"])
        incoming_message_prefix_hashes = self._extend_message_prefix_hashes([], messages)
        selected_chain = self._select_chain(
            tools=tools,
            incoming_message_prefix_hashes=incoming_message_prefix_hashes,
        )

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
        else:
            buffer = self._copy_trajectory_buffer(selected_chain.buffer)
            image_data, video_data = self._copy_chain_media(selected_chain)
            chain_id = selected_chain.chain_id
            incremental_messages = messages[len(selected_chain.message_history) :]
            new_image_data = None
            new_video_data = None
            incremental_ids = []
            already_exhausted = self._response_length is not None and len(buffer.response_mask) >= self._response_length
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
                context_ids = buffer.prompt_ids + buffer.response_ids
                return EncodedData(
                    buffer=buffer,
                    context_ids=context_ids,
                    sampling_params={},
                    messages=list(messages),
                    tools=tools,
                    image_data=image_data,
                    video_data=video_data,
                    length_exhausted_trajectory=self._build_materialized_trajectory(
                        chain=selected_chain,
                        extra_fields={"materialization_reason": "max_response_length"},
                    ),
                    chain_id=selected_chain.chain_id,
                    incoming_message_prefix_hashes=list(incoming_message_prefix_hashes),
                )

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

        context_ids = buffer.prompt_ids + buffer.response_ids
        remaining_response_budget = (
            self._response_length - len(buffer.response_mask) if self._response_length is not None else None
        )
        if remaining_response_budget is not None:
            sampling_params["max_tokens"] = min(
                sampling_params.get("max_tokens", remaining_response_budget),
                remaining_response_budget,
            )
        return EncodedData(
            buffer=buffer,
            context_ids=context_ids,
            sampling_params=sampling_params,
            messages=list(messages),
            tools=tools,
            image_data=image_data,
            video_data=video_data,
            length_exhausted_trajectory=None,
            chain_id=chain_id,
            incoming_message_prefix_hashes=list(incoming_message_prefix_hashes),
        )

    def _select_chain(
        self,
        *,
        tools: list[dict[str, Any]] | None,
        incoming_message_prefix_hashes: list[str],
    ) -> ChainState | None:
        candidates = [
            chain
            for chain in self.active_chains
            if not self._trajectory_session._is_chain_reserved(chain.chain_id)
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

    def _extend_message_prefix_hashes(
        self,
        existing_prefix_hashes: list[str],
        new_messages: list[dict[str, Any]],
    ) -> list[str]:
        return self._trajectory_session._extend_message_prefix_hashes(existing_prefix_hashes, new_messages)

    def _copy_trajectory_buffer(self, buffer: TrajectoryBuffer) -> TrajectoryBuffer:
        return self._trajectory_session._copy_trajectory_buffer(buffer)

    def _copy_chain_media(self, chain: ChainState) -> tuple[list[Any] | None, list[Any] | None]:
        return self._trajectory_session._copy_chain_media(chain)

    def _copy_media_list(self, media: list[Any] | None) -> list[Any] | None:
        return self._trajectory_session._copy_media_list(media)

    def _commit_generation_to_chain(self, encoded: EncodedData, assistant_msg: dict[str, Any]) -> None:
        message_history = list(encoded.messages) + [assistant_msg]
        message_prefix_hashes = self._extend_message_prefix_hashes(
            encoded.incoming_message_prefix_hashes,
            [assistant_msg],
        )
        assert len(message_prefix_hashes) == len(message_history)
        if encoded.chain_id is None:
            order_seq = self._next_order_seq()
            chain_id = self._allocate_chain_id()
            self.active_chains.append(
                ChainState(
                    chain_id=chain_id,
                    message_history=message_history,
                    message_tip_hash=message_prefix_hashes[-1],
                    active_tool_schemas=encoded.tools,
                    buffer=encoded.buffer,
                    image_data=self._copy_media_list(encoded.image_data),
                    video_data=self._copy_media_list(encoded.video_data),
                    updated_seq=order_seq,
                )
            )
            return

        chain_index, previous_chain = self._find_active_chain(encoded.chain_id)
        order_seq = self._next_order_seq()
        self.active_chains[chain_index] = ChainState(
            chain_id=previous_chain.chain_id,
            message_history=message_history,
            message_tip_hash=message_prefix_hashes[-1],
            active_tool_schemas=encoded.tools,
            buffer=encoded.buffer,
            image_data=self._copy_media_list(encoded.image_data),
            video_data=self._copy_media_list(encoded.video_data),
            updated_seq=order_seq,
        )

    def _close_length_exhausted_chain(self, encoded: EncodedData) -> None:
        if encoded.chain_id is None or encoded.length_exhausted_trajectory is None:
            raise RuntimeError("length-exhausted chain metadata is missing")
        chain_index, chain = self._find_active_chain(encoded.chain_id)
        order_seq = self._next_order_seq()
        self.materialized_chains.append(
            MaterializedChain(
                trajectory=encoded.length_exhausted_trajectory,
                order_seq=order_seq,
            )
        )
        del self.active_chains[chain_index]

    def _find_active_chain(self, chain_id: int) -> tuple[int, ChainState]:
        return self._trajectory_session._find_active_chain(chain_id)

    def _allocate_chain_id(self) -> int:
        return self._trajectory_session._allocate_chain_id()

    async def _release_chain_reservation(self, chain_id: int) -> None:
        await self._trajectory_session._release_chain_reservation(chain_id)

    def _next_order_seq(self) -> int:
        return self._trajectory_session._next_order_seq()

    def _build_materialized_trajectory(
        self,
        *,
        chain: ChainState,
        extra_fields: dict[str, Any] | None = None,
    ) -> Trajectory:
        return self._trajectory_session._build_materialized_trajectory(
            chain=chain,
            extra_fields=extra_fields,
        )

    def _touch(self) -> None:
        self._trajectory_session._touch()
