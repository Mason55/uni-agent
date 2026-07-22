"""Shared gateway-owned types passed across session boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

if TYPE_CHECKING:
    import numpy as np
    import torch

_FINISH_REASON_MAP = {
    "completed": "stop",
    "stop": "stop",
    "matched_stop": "stop",
    "eos": "stop",
    "length": "length",
    "max_tokens": "length",
    "aborted": "stop",
    "abort": "stop",
}


def normalize_finish_reason(stop_reason: str | None) -> str:
    """Map backend stop reasons into the gateway finish-reason vocabulary."""
    if stop_reason is None:
        return "stop"
    return _FINISH_REASON_MAP.get(stop_reason, stop_reason)


def _freeze_capture_value(value: Any) -> Any:
    """Recursively detach and freeze JSON-like capture metadata."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_capture_value(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_capture_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_capture_value(item) for item in value)
    return deepcopy(value)


def mutable_capture_value(value: Any) -> Any:
    """Return detached mutable containers for session-owned capture state."""
    if isinstance(value, Mapping):
        return {key: mutable_capture_value(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [mutable_capture_value(nested) for nested in value]
    if isinstance(value, frozenset):
        return {mutable_capture_value(nested) for nested in value}
    return deepcopy(value)


def mutable_capture_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached mutable mapping for session-owned message state."""
    return mutable_capture_value(value)


class InternalGenerationRequest(TypedDict):
    """Lowered request consumed by GatewaySession.run_generation.

    Provider adapters lower OpenAI / Anthropic wire requests into this
    template-facing canonical before the session sees them. It is not a
    provider-neutral block model.
    """

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    sampling_params: dict[str, Any]


@dataclass(frozen=True)
class CapturedGeneration:
    """Immutable rollout truth supplied by an external generation owner."""

    assistant_message: Mapping[str, Any]
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    completion_logprobs: tuple[float, ...]
    stop_reason: str
    routed_experts: Any | None = None
    routing_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "assistant_message", _freeze_capture_value(self.assistant_message))
        object.__setattr__(self, "prompt_ids", tuple(self.prompt_ids))
        object.__setattr__(self, "completion_ids", tuple(self.completion_ids))
        object.__setattr__(self, "completion_logprobs", tuple(self.completion_logprobs))
        if self.routed_experts is not None:
            object.__setattr__(self, "routed_experts", _freeze_capture_value(self.routed_experts))
        if self.routing_metadata is not None:
            object.__setattr__(self, "routing_metadata", _freeze_capture_value(self.routing_metadata))


@dataclass(frozen=True)
class CaptureUsage:
    """Token usage for one committed model call."""

    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class CaptureReceipt:
    """Immutable public result of one successfully committed capture."""

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    chain_id: int
    turn_id: int
    usage: CaptureUsage
    assistant_message: Mapping[str, Any]
    finish_reason: str
    routed_experts: Any | None = None
    routing_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_ids", tuple(self.prompt_ids))
        object.__setattr__(self, "response_ids", tuple(self.response_ids))
        object.__setattr__(self, "response_mask", tuple(self.response_mask))
        object.__setattr__(self, "response_logprobs", tuple(self.response_logprobs))
        object.__setattr__(self, "assistant_message", _freeze_capture_value(self.assistant_message))
        if self.routed_experts is not None:
            object.__setattr__(self, "routed_experts", _freeze_capture_value(self.routed_experts))
        if self.routing_metadata is not None:
            object.__setattr__(self, "routing_metadata", _freeze_capture_value(self.routing_metadata))


class CaptureTransaction(Protocol):
    """Public opaque interface for one asynchronous passive capture."""

    @property
    def context_ids(self) -> tuple[int, ...]: ...

    @property
    def sampling_params(self) -> dict[str, Any]: ...

    @property
    def image_data(self) -> tuple[Any, ...] | None: ...

    @property
    def video_data(self) -> tuple[Any, ...] | None: ...

    @property
    def length_exhausted(self) -> bool: ...

    async def __aenter__(self) -> CaptureTransaction: ...

    async def __aexit__(self, exc_type, exc, traceback) -> bool: ...

    async def commit(self, generation: CapturedGeneration) -> CaptureReceipt: ...

    async def rollback(self) -> None: ...


@dataclass
class SessionHandle:
    """Address returned to agent runners for a newly created gateway session.

    Attributes:
        session_id: Stable session identifier assigned by the caller.
        base_url: Per-session provider-compatible ``/v1`` API root, or ``None``
            when the handle only needs to identify the session.
        reward_info_url: Per-session endpoint used by runners to attach reward
            metadata; this is a sibling of the provider ``/v1`` root rather
            than part of that API.
    """

    session_id: str
    base_url: str | None = None
    reward_info_url: str | None = None


@dataclass
class Trajectory:
    """Token-level training trajectory produced when a gateway session finalizes.

    Attributes:
        prompt_ids: Prompt token IDs used to seed the trajectory.
        response_ids: Tokens generated by the model plus interstitial prompt
            tokens added during continuation turns.
        response_mask: Per-token labels for ``response_ids``; ``1`` marks model
            output and ``0`` marks interstitial context tokens.
        response_logprobs: Optional log probabilities aligned with
            ``response_ids``; continuation context tokens use ``0.0``.
        reward_info: Reward metadata attached through the session reward-info
            endpoint before finalization.
        reward_score: Optional scalar reward assigned by downstream training.
        num_turns: Chat-turn count materialized with the trajectory.
        routed_experts: Optional expert-routing data captured by the backend.
        multi_modal_data: Optional image/video data associated with the prompt.
        extra_fields: Gateway-owned extension fields, such as trajectory
            materialization metadata consumed by training adapters.
    """

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None = None
    reward_info: dict[str, Any] = field(default_factory=dict)
    reward_score: float | None = None
    num_turns: int = 0
    routed_experts: torch.Tensor | np.ndarray | None = None
    multi_modal_data: dict[str, Any] | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)
