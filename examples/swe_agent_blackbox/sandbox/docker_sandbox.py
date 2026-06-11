"""Local Docker sandbox command execution."""

from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class CommandResult:
    """Result of a command executed inside a sandbox."""

    stdout: str
    stderr: str
    exit_code: int


logger = logging.getLogger(__name__)


class DockerSandboxCommands:
    """Command execution via local Docker (``docker exec``).

    The sidecar tool is loaded from the tool image via ``docker cp``,
    so both local and remote sandboxes use the same image artifact.
    """

    def __init__(self, container_id: str) -> None:
        self._container_id = container_id

    @property
    def sandbox_id(self) -> str:
        return self._container_id[:12]

    @classmethod
    async def create(
        cls,
        *,
        image: str,
        sidecar_image: str = "",
        env: dict[str, str] | None = None,
        container_timeout: str = "2h",
        docker_executable: str = "docker",
        extra_run_args: list[str] | None = None,
    ) -> "DockerSandboxCommands":
        """Create and start a Docker sandbox with the sidecar tool copied in.

        The sidecar venv is copied from the tool image into the sandbox
        container via ``docker create`` + ``docker cp``, so both local
        and remote sandboxes use the same ``mini-swe-agent-tool`` image.
        """
        container_name = f"mwea-sandbox-{uuid.uuid4().hex[:8]}"

        cmd = [
            docker_executable, "run", "-d",
            "--name", container_name,
        ]

        for k, v in (env or {}).items():
            cmd.extend(["-e", f"{k}={v}"])

        cmd.extend(extra_run_args or [])
        cmd.extend([image, "sleep", container_timeout])

        logger.debug("Starting sandbox with: %s", shlex.join(cmd))
        r = await asyncio.to_thread(
            lambda: subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=120, check=True,
            ),
        )
        container_id = r.stdout.strip()
        logger.info("Started sandbox container %s (id=%s)", container_name, container_id[:12])

        # Copy sidecar tool from tool image into the sandbox container
        if sidecar_image:
            await _copy_sidecar(sidecar_image, container_id, docker_executable)

        return cls(container_id=container_id)

    async def run(self, cmd: str, *, timeout: int = 60) -> CommandResult:
        """Execute *cmd* inside the sandbox via ``docker exec``."""
        exec_cmd = [
            "docker", "exec",
            self._container_id, "bash", "-c", cmd,
        ]
        try:
            r = await asyncio.to_thread(
                lambda: subprocess.run(
                    exec_cmd,
                    capture_output=True, text=True,
                    timeout=timeout,
                    encoding="utf-8", errors="replace",
                ),
            )
            return CommandResult(stdout=r.stdout, stderr=r.stderr, exit_code=r.returncode)
        except subprocess.TimeoutExpired as e:
            output = e.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            return CommandResult(
                stdout=output,
                stderr=f"Command timed out after {timeout}s",
                exit_code=-1,
            )
        except Exception as e:
            return CommandResult(stdout="", stderr=str(e), exit_code=-1)

    async def cleanup(self) -> None:
        """Stop and remove the Docker container (fire-and-forget)."""
        if self._container_id:
            cmd = (
                f"(timeout 60 docker stop {self._container_id} "
                f"|| docker rm -f {self._container_id}) >/dev/null 2>&1 &"
            )
            subprocess.Popen(cmd, shell=True)
            logger.info("Cleaning up container %s", self._container_id[:12])
            self._container_id = ""


async def _copy_sidecar(
    sidecar_image: str,
    container_id: str,
    docker_executable: str = "docker",
) -> None:
    """Copy /opt/mini-swe-agent from the tool image into a running container.

    Uses a two-step process: cp from tool container to host tmpdir,
    then cp from host into the target sandbox container.
    """
    import tempfile

    def _do() -> None:
        # Create a temporary container from the tool image
        r = subprocess.run(
            [docker_executable, "create", sidecar_image],
            capture_output=True, text=True, timeout=60, check=True,
        )
        tmp_id = r.stdout.strip()
        try:
            # Step 1: cp from tool container to host tmpdir
            with tempfile.TemporaryDirectory() as tmpdir:
                subprocess.run(
                    [docker_executable, "cp", f"{tmp_id}:/opt/mini-swe-agent", tmpdir],
                    capture_output=True, text=True, timeout=120, check=True,
                )
                # Step 2: cp from host into the running sandbox container
                subprocess.run(
                    [docker_executable, "cp", f"{tmpdir}/mini-swe-agent", f"{container_id}:/opt/"],
                    capture_output=True, text=True, timeout=120, check=True,
                )
            logger.info("Copied sidecar from %s into sandbox %s", sidecar_image, container_id[:12])
        finally:
            subprocess.run(
                [docker_executable, "rm", tmp_id],
                capture_output=True, text=True, timeout=30,
            )

    await asyncio.to_thread(_do)
