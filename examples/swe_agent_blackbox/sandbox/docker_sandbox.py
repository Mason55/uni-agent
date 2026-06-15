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

    def __init__(self, container_id: str, *, login_shell: bool = False) -> None:
        self._container_id = container_id
        self._login_shell = login_shell

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
        container_name_prefix: str = "mwea-sandbox",
        sidecar_source_path: str = "/opt/mini-swe-agent",
        sidecar_target_path: str = "/opt",
        sidecar_copy_contents: bool = False,
        login_shell: bool = False,
    ) -> "DockerSandboxCommands":
        """Create and start a Docker sandbox with the sidecar tool copied in.

        By default this preserves the mini-swe-agent layout:
        ``tool:/opt/mini-swe-agent`` -> ``sandbox:/opt/mini-swe-agent``.
        Claude Code can reuse the same path with
        ``sidecar_copy_contents=True`` and target ``/opt/claude-code``.
        """
        container_name = f"{container_name_prefix}-{uuid.uuid4().hex[:8]}"

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
            await _copy_sidecar(
                sidecar_image=sidecar_image,
                container_id=container_id,
                docker_executable=docker_executable,
                source_path=sidecar_source_path,
                target_path=sidecar_target_path,
                copy_contents=sidecar_copy_contents,
            )

        return cls(container_id=container_id, login_shell=login_shell)

    async def run(self, cmd: str, *, timeout: int = 60) -> CommandResult:
        """Execute *cmd* inside the sandbox via ``docker exec``."""
        shell_arg = "-lc" if self._login_shell else "-c"
        exec_cmd = ["docker", "exec", self._container_id, "bash", shell_arg, cmd]
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
    *,
    sidecar_image: str,
    container_id: str,
    docker_executable: str = "docker",
    source_path: str = "/opt/mini-swe-agent",
    target_path: str = "/opt",
    copy_contents: bool = False,
) -> None:
    """Copy a sidecar path from the tool image into a running container.

    ``copy_contents=False`` copies the source directory itself under
    ``target_path``. ``copy_contents=True`` copies source contents into
    ``target_path``; this supports scratch images mounted at a custom path.
    """
    import tempfile

    def _do() -> None:
        r = subprocess.run(
            [docker_executable, "create", sidecar_image],
            capture_output=True, text=True, timeout=60, check=True,
        )
        tmp_id = r.stdout.strip()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                if copy_contents:
                    host_sidecar = f"{tmpdir}/sidecar"
                    host_target = host_sidecar
                else:
                    source_basename = source_path.rstrip("/").rsplit("/", 1)[-1] or "sidecar"
                    host_sidecar = f"{tmpdir}/{source_basename}"
                    host_target = tmpdir
                subprocess.run(
                    [docker_executable, "cp", f"{tmp_id}:{source_path}", host_target],
                    capture_output=True, text=True, timeout=120, check=True,
                )
                if copy_contents:
                    subprocess.run(
                        [docker_executable, "exec", container_id, "bash", "-lc", f"mkdir -p {shlex.quote(target_path)}"],
                        capture_output=True, text=True, timeout=120, check=True,
                    )
                    copy_source = f"{host_sidecar}/."
                else:
                    copy_source = host_sidecar
                subprocess.run(
                    [docker_executable, "cp", copy_source, f"{container_id}:{target_path}/"],
                    capture_output=True, text=True, timeout=120, check=True,
                )
            logger.info(
                "Copied sidecar %s:%s -> %s:%s",
                sidecar_image,
                source_path,
                container_id[:12],
                target_path,
            )
        finally:
            subprocess.run(
                [docker_executable, "rm", tmp_id],
                capture_output=True, text=True, timeout=30,
            )

    await asyncio.to_thread(_do)
