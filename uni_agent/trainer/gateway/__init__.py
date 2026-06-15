__all__ = [
    "GatewayActor",
    "GatewayManager",
    "GatewayServingRuntime",
]


def __getattr__(name):
    if name == "GatewayActor":
        from .gateway import GatewayActor

        return GatewayActor
    if name == "GatewayManager":
        from .manager import GatewayManager

        return GatewayManager
    if name == "GatewayServingRuntime":
        from .runtime import GatewayServingRuntime

        return GatewayServingRuntime
    raise AttributeError(name)
