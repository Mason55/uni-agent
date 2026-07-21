"""Session domain for the gateway: per-session state and model codec.

The gateway is a thin HTTP layer; this package holds the session-side logic it
serves: trajectory buffering and message encoding/decoding.
``SessionHandle`` / ``Trajectory`` are consumed by framework runners, while
``InternalGenerationRequest`` is the adapter-to-session request boundary.
"""

from typing import Any

__all__ = [
    "InternalGenerationRequest",
    "GatewaySession",
    "MessageCodec",
    "SessionHandle",
    "SessionLifecycleError",
    "SessionPhase",
    "Trajectory",
    "TrajectoryBuffer",
    "TrajectorySession",
]


def __getattr__(name: str) -> Any:
    """Load rollout and collection interfaces without eager runtime imports."""
    if name == "MessageCodec":
        from .codec import MessageCodec

        return MessageCodec
    if name == "GatewaySession":
        from .session import GatewaySession

        return GatewaySession
    if name in {"SessionLifecycleError", "SessionPhase", "TrajectoryBuffer", "TrajectorySession"}:
        from .trajectory_session import SessionLifecycleError, SessionPhase, TrajectoryBuffer, TrajectorySession

        return {
            "SessionLifecycleError": SessionLifecycleError,
            "SessionPhase": SessionPhase,
            "TrajectoryBuffer": TrajectoryBuffer,
            "TrajectorySession": TrajectorySession,
        }[name]
    if name in {"InternalGenerationRequest", "SessionHandle", "Trajectory"}:
        from .types import InternalGenerationRequest, SessionHandle, Trajectory

        return {
            "InternalGenerationRequest": InternalGenerationRequest,
            "SessionHandle": SessionHandle,
            "Trajectory": Trajectory,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
