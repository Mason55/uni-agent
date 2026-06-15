__all__ = [
    "AgentFramework",
    "OpenAICompatibleAgentFramework",
    "SessionHandle",
    "Trajectory",
]


def __getattr__(name):
    if name in {"AgentFramework", "OpenAICompatibleAgentFramework"}:
        from .framework import AgentFramework, OpenAICompatibleAgentFramework

        return {"AgentFramework": AgentFramework, "OpenAICompatibleAgentFramework": OpenAICompatibleAgentFramework}[name]
    if name in {"SessionHandle", "Trajectory"}:
        from .types import SessionHandle, Trajectory

        return {"SessionHandle": SessionHandle, "Trajectory": Trajectory}[name]
    raise AttributeError(name)
