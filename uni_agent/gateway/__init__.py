from typing import Any

__all__ = [
    "GatewayActor",
    "GatewayManager",
]


def __getattr__(name: str) -> Any:
    """Preserve public exports without importing Ray-backed runtime eagerly."""
    if name == "GatewayActor":
        from .gateway import GatewayActor

        return GatewayActor
    if name == "GatewayManager":
        from .manager import GatewayManager

        return GatewayManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
