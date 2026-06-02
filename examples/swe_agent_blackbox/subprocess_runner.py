"""Ray-based subprocess runner for agent_runner execution.

Launches agent_runner in a separate Ray worker process to prevent blocking
operations (sleep, sync I/O, etc.) from stalling the framework's event loop.

The agent_runner receives stub SessionHandle/SessionRuntime objects that are
compatible with its existing interface — no changes to any runner needed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import ray


# ──────────── Stub objects ────────────


@dataclass
class _StubSessionHandle:
    """Transparent SessionHandle replacement for agent_runner.

    Provides the real base_url so the runner can make LLM calls to the Gateway.
    """

    session_id: str
    base_url: str | None = None


class _StubSessionRuntime:
    """Intercepts complete_session to capture reward_info.

    All other SessionRuntime methods are no-ops because agent_runner only
    calls complete_session (the parent process handles the full lifecycle).
    """

    def __init__(self):
        self.reward_info: dict[str, Any] | None = None

    async def create_session(self, session_id: str, **kwargs):
        return _StubSessionHandle(session_id=session_id, base_url=None)

    async def complete_session(self, session_id: str, reward_info: dict[str, Any] | None = None):
        self.reward_info = reward_info

    async def finalize_session(self, session_id: str) -> list:
        return []

    async def abort_session(self, session_id: str) -> None:
        pass

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        pass


# ──────────── Remote function ────────────


@ray.remote(num_cpus=0)
def remote_agent_run(
    agent_runner_fqn: str,
    raw_prompt,
    session_id: str,
    base_url: str,
    sample_index: int,
    runner_kwargs: dict,
) -> dict[str, Any] | None:
    """Run agent_runner in a dedicated Ray worker process.

    Args:
        agent_runner_fqn: Fully qualified name, e.g.
            "examples.swe_agent_blackbox.agent_runner.swe_agent_runner".
        raw_prompt: Prompt text or chat messages for the agent.
        session_id: Gateway session identifier.
        base_url: Gateway HTTP base URL for LLM calls.
        sample_index: Sample index in the training batch.
        runner_kwargs: Additional keyword arguments — tools_kwargs,
            agent_config_path, tool_config, etc.

    Returns:
        reward_info captured from the agent_runner's complete_session call.
    """
    from verl.utils.import_utils import load_class_from_fqn

    agent_runner = load_class_from_fqn(agent_runner_fqn)

    stub_runtime = _StubSessionRuntime()
    stub_handle = _StubSessionHandle(session_id=session_id, base_url=base_url)

    async def _run():
        await agent_runner(
            raw_prompt=raw_prompt,
            session=stub_handle,
            sample_index=sample_index,
            session_runtime=stub_runtime,
            **runner_kwargs,
        )
        return stub_runtime.reward_info

    return asyncio.run(_run())
