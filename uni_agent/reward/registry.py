"""Reward spec registry: register by name and load by config (mirrors tools/registry)."""

from typing import Any

from uni_agent.reward.base import AbstractRewardSpec

REWARD_SPEC_REGISTRY: dict[str, type[AbstractRewardSpec]] = {}


def register_reward_spec(name: str) -> type[AbstractRewardSpec]:
    """Decorator to register a reward spec class with a given name."""

    def decorator(cls: type[AbstractRewardSpec]) -> type[AbstractRewardSpec]:
        if name in REWARD_SPEC_REGISTRY and REWARD_SPEC_REGISTRY[name] != cls:
            raise ValueError(f"Reward spec {name} has already been registered: {REWARD_SPEC_REGISTRY[name]} vs {cls}")
        REWARD_SPEC_REGISTRY[name] = cls
        return cls

    return decorator


def load_reward_spec(config: dict[str, Any]) -> AbstractRewardSpec:
    """
    Load a reward spec instance by config.

    Config must contain "name" (registered name). Other keys are passed as kwargs
    to the reward spec class constructor.

    Example:
        config = {"name": "swe_bench", "metadata": {...}}
        spec = load_reward_spec(config)
    """
    if not config or "name" not in config:
        raise ValueError("Reward config must contain 'name'")
    name = config["name"]
    if name not in REWARD_SPEC_REGISTRY:
        raise ValueError(f"Unknown reward spec: {name}. Registered: {list(REWARD_SPEC_REGISTRY.keys())}")
    kwargs = {k: v for k, v in config.items() if k != "name"}
    return REWARD_SPEC_REGISTRY[name](**kwargs)
