"""OpenYuanRong (AKernel) remote sandbox command execution.

Uses ``akernel_sdk.Sandbox`` with sidecar ``Mount`` to inject the
mini-swe-agent tool image.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from examples.swe_agent_blackbox.sandbox.docker_sandbox import CommandResult

logger = logging.getLogger(__name__)


def _configure_akernel_env() -> None:
    """Map OPENYUANRONG_* env vars to AKERNEL_* before importing akernel_sdk."""
    server = os.getenv("OPENYUANRONG_SERVER_ADDRESS")
    token = os.getenv("OPENYUANRONG_TOKEN")
    if not server or not token:
        raise ValueError(
            "OPENYUANRONG_SERVER_ADDRESS and OPENYUANRONG_TOKEN "
            "environment variables must be set for YR sandbox"
        )
    os.environ["AKERNEL_SERVER_ADDRESS"] = server
    os.environ["AKERNEL_TOKEN"] = token


class YRSandboxCommands:
    """Command execution via OpenYuanRong (AKernel) remote sandbox."""

    def __init__(self, sandbox: Any) -> None:
        self._sandbox = sandbox

    # -- Factory --------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        *,
        image: str,
        sidecar_image: str,
        env: dict[str, str] | None = None,
        cpu: int = 4000,
        memory: int = 8192,
        idle_timeout: int = 600,
        **sandbox_kwargs: Any,
    ) -> "YRSandboxCommands":
        """Create an OpenYuanRong sandbox with sidecar tool mounted.

        The sidecar image is mounted at ``/opt/mini-swe-agent`` inside the
        sandbox via ``akernel_sdk.Mount``.
        """
        _configure_akernel_env()
        from akernel_sdk import Mount, Sandbox

        sb_kwargs: dict[str, Any] = {
            "image": image,
            "cpu": cpu,
            "memory": memory,
            "idle_timeout": idle_timeout,
            "mounts": [
                Mount(target="/opt/mini-swe-agent", image_url=sidecar_image),
            ],
        }
        if env:
            sb_kwargs["env"] = env
        sb_kwargs.update(sandbox_kwargs)

        logger.info(
            "Creating YR sandbox (image=%s, cpu=%d, memory=%d, sidecar=%s)",
            image, cpu, memory, sidecar_image,
        )
        sandbox = await asyncio.to_thread(lambda: Sandbox(**sb_kwargs))
        logger.info("YR sandbox created: %s", getattr(sandbox, "sandbox_id", "?"))
        return cls(sandbox=sandbox)

    # -- SandboxCommands implementation ----------------------------------------

    async def run(self, cmd: str, *, timeout: int = 60) -> CommandResult:
        """Execute *cmd* inside the YR sandbox via ``sandbox.commands.run``."""
        try:
            result = await asyncio.to_thread(
                self._sandbox.commands.run, cmd, timeout=timeout,
            )
            return CommandResult(
                stdout=getattr(result, "stdout", ""),
                stderr=getattr(result, "stderr", ""),
                exit_code=getattr(result, "exit_code", -1),
            )
        except Exception as e:
            return CommandResult(stdout="", stderr=str(e), exit_code=-1)

    async def cleanup(self) -> None:
        """Kill the YR sandbox."""
        if self._sandbox is not None:
            sandbox_id = getattr(self._sandbox, "sandbox_id", "?")
            try:
                await asyncio.to_thread(self._sandbox.kill)
                logger.info("YR sandbox %s killed", sandbox_id)
            except Exception as e:
                logger.warning("Failed to kill YR sandbox %s: %s", sandbox_id, e)
            self._sandbox = None
