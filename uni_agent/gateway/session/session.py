"""Per-session gateway state, generation envelope, and lifecycle handling."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from uni_agent.gateway.session.codec import MessageCodec
from uni_agent.gateway.session.trajectory_session import (
    CaptureDomainError,
    ChainState,
    MaterializedChain,
    SessionLifecycleError,
    SessionPhase,
    TrajectorySession,
)
from uni_agent.gateway.session.types import (
    CapturedGeneration,
    CaptureReceipt,
    InternalGenerationRequest,
    SessionHandle,
    Trajectory,
)


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
        capture_receipt: Per-call receipt committed through ``TrajectorySession``.
    """

    assistant_msg: dict[str, Any]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    capture_receipt: CaptureReceipt | None = None


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
        # wrapper; TrajectorySession.capture enforces the response budget.
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
        try:
            capture_request = deepcopy(request)
            capture_request["sampling_params"]["logprobs"] = True
            transaction = await self._trajectory_session.capture(capture_request)
            tools = capture_request["tools"]
            async with transaction:
                if transaction.length_exhausted:
                    assistant_msg = {"role": "assistant", "content": ""}
                    receipt = await transaction.commit(
                        CapturedGeneration(
                            assistant_message=assistant_msg,
                            prompt_ids=transaction.context_ids,
                            completion_ids=(),
                            completion_logprobs=(),
                            stop_reason="length",
                        )
                    )
                else:
                    try:
                        output = await backend.generate(
                            request_id=self.handle.session_id,
                            prompt_ids=list(transaction.context_ids),
                            sampling_params=transaction.sampling_params,
                            image_data=(list(transaction.image_data) if transaction.image_data is not None else None),
                            video_data=(list(transaction.video_data) if transaction.video_data is not None else None),
                        )
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=str(exc)) from exc
                    except Exception as exc:
                        raise HTTPException(
                            status_code=500,
                            detail=f"{exc.__class__.__name__}: {exc}",
                        ) from exc

                    response_ids = tuple(output.token_ids)
                    assistant_msg, finish_reason = await self._codec.decode_response(
                        list(response_ids),
                        tools=tools,
                        stop_reason=output.stop_reason,
                    )
                    receipt = await transaction.commit(
                        CapturedGeneration(
                            assistant_message=assistant_msg,
                            prompt_ids=transaction.context_ids,
                            completion_ids=response_ids,
                            completion_logprobs=output.log_probs,
                            stop_reason=finish_reason,
                            routed_experts=getattr(output, "routed_experts", None),
                        )
                    )
        except HTTPException:
            raise
        except SessionLifecycleError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except CaptureDomainError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"{exc.__class__.__name__}: {exc}",
            ) from exc

        return GenerationOutcome(
            assistant_msg=assistant_msg,
            finish_reason=receipt.finish_reason,
            prompt_tokens=receipt.usage.prompt_tokens,
            completion_tokens=receipt.usage.completion_tokens,
            capture_receipt=receipt,
        )

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

    def _extend_message_prefix_hashes(
        self,
        existing_prefix_hashes: list[str],
        new_messages: list[dict[str, Any]],
    ) -> list[str]:
        """Retain the legacy diagnostic helper while delegating its algorithm."""
        return self._trajectory_session._extend_message_prefix_hashes(existing_prefix_hashes, new_messages)
