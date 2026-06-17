"""Sandbox helpers for Claude Code sidecar execution."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_PROXY_PORT = 38197
CLAUDE_CODE_TARGET = "/opt/claude-code"


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int


def extract_upstream(gateway_url: str) -> str:
    parsed = urlparse(gateway_url)
    return f"{parsed.hostname}:{parsed.port}"


def rewrite_gateway_url(gateway_url: str, proxy_port: int = DEFAULT_PROXY_PORT) -> str:
    parsed = urlparse(gateway_url)
    return f"http://127.0.0.1:{proxy_port}{parsed.path.removesuffix('/v1')}"


def _configure_akernel_env() -> None:
    server = os.getenv("OPENYUANRONG_SERVER_ADDRESS")
    token = os.getenv("OPENYUANRONG_TOKEN")
    os.environ.setdefault("TUNNEL_SSL_VERIFY", "0")
    if not server or not token:
        raise ValueError(
            "OPENYUANRONG_SERVER_ADDRESS and OPENYUANRONG_TOKEN must be set for OpenYuanRong sandbox"
        )
    os.environ["AKERNEL_SERVER_ADDRESS"] = server
    os.environ["AKERNEL_TOKEN"] = token


class ClaudeDockerSandbox:
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
        sidecar_image: str,
        env: dict[str, str] | None = None,
        container_timeout: str = "2h",
        docker_executable: str = "docker",
    ) -> "ClaudeDockerSandbox":
        container_name = f"claude-swe-sandbox-{uuid.uuid4().hex[:8]}"
        cmd = [docker_executable, "run", "-d", "--name", container_name, "--entrypoint", "sleep"]
        for key, value in (env or {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([image, container_timeout])

        r = await asyncio.to_thread(
            lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
        )
        container_id = r.stdout.strip()
        await _copy_sidecar_root(
            sidecar_image=sidecar_image,
            container_id=container_id,
            target=CLAUDE_CODE_TARGET,
            docker_executable=docker_executable,
        )
        return cls(container_id)

    async def run(self, cmd: str, *, timeout: int = 60) -> CommandResult:
        exec_cmd = ["docker", "exec", self._container_id, "bash", "-lc", cmd]
        try:
            r = await asyncio.to_thread(
                lambda: subprocess.run(
                    exec_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding="utf-8",
                    errors="replace",
                )
            )
            return CommandResult(stdout=r.stdout, stderr=r.stderr, exit_code=r.returncode)
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            return CommandResult(stdout=output, stderr=f"Command timed out after {timeout}s", exit_code=-1)
        except Exception as exc:
            return CommandResult(stdout="", stderr=str(exc), exit_code=-1)

    async def cleanup(self) -> None:
        if self._container_id:
            cmd = (
                f"(timeout 60 docker stop {self._container_id} "
                f"|| docker rm -f {self._container_id}) >/dev/null 2>&1 &"
            )
            subprocess.Popen(cmd, shell=True)
            self._container_id = ""


async def _copy_sidecar_root(
    *,
    sidecar_image: str,
    container_id: str,
    target: str,
    docker_executable: str = "docker",
) -> None:
    def _do() -> None:
        r = subprocess.run(
            [docker_executable, "create", sidecar_image, "/bin/true"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        tmp_id = r.stdout.strip()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                sidecar_dir = f"{tmpdir}/sidecar"
                subprocess.run(
                    [docker_executable, "cp", f"{tmp_id}:/.", sidecar_dir],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True,
                )
                subprocess.run(
                    [
                        docker_executable,
                        "exec",
                        container_id,
                        "bash",
                        "-lc",
                        f"mkdir -p {shlex.quote(target)}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True,
                )
                subprocess.run(
                    [docker_executable, "cp", f"{sidecar_dir}/.", f"{container_id}:{target}/"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True,
                )
        finally:
            subprocess.run([docker_executable, "rm", tmp_id], capture_output=True, text=True, timeout=30)

    await asyncio.to_thread(_do)


class ClaudeYRSandbox:
    def __init__(self, sandbox: Any) -> None:
        self._sandbox = sandbox

    @property
    def sandbox_id(self) -> str:
        return getattr(self._sandbox, "sandbox_id", "unknown")

    @classmethod
    async def create(
        cls,
        *,
        image: str,
        sidecar_image: str,
        upstream: str = "",
        proxy_port: int = DEFAULT_PROXY_PORT,
        env: dict[str, str] | None = None,
        cpu: int = 1000,
        memory: int = 2048,
        cpu_limit: int = 4000,
        mem_limit: int = 8192,
        idle_timeout: int = 600,
    ) -> "ClaudeYRSandbox":
        _configure_akernel_env()
        from akernel_sdk import Mount, Sandbox

        kwargs: dict[str, Any] = {
            "image": image,
            "cpu": cpu,
            "memory": memory,
            "cpu_limit": cpu_limit,
            "mem_limit": mem_limit,
            "idle_timeout": idle_timeout,
            "mounts": [Mount(target=CLAUDE_CODE_TARGET, image_url=sidecar_image)],
        }
        if upstream:
            kwargs["upstream"] = upstream
            kwargs["proxy_port"] = proxy_port
        if env:
            kwargs["env"] = env

        sandbox = await asyncio.to_thread(lambda: Sandbox(**kwargs))
        return cls(sandbox)

    async def run(self, cmd: str, *, timeout: int = 60) -> CommandResult:
        try:
            result = await asyncio.to_thread(self._sandbox.commands.run, cmd, timeout=timeout)
            return CommandResult(
                stdout=getattr(result, "stdout", ""),
                stderr=getattr(result, "stderr", ""),
                exit_code=getattr(result, "exit_code", -1),
            )
        except Exception as exc:
            return CommandResult(stdout="", stderr=str(exc), exit_code=-1)

    async def cleanup(self) -> None:
        if self._sandbox is not None:
            try:
                await asyncio.to_thread(self._sandbox.kill)
            finally:
                self._sandbox = None


async def create_claude_sandbox(
    *,
    image: str,
    sidecar_image: str,
    sandbox_type: str,
    gateway_url: str,
) -> ClaudeDockerSandbox | ClaudeYRSandbox:
    if sandbox_type == "local":
        return await ClaudeDockerSandbox.create(image=image, sidecar_image=sidecar_image)
    if sandbox_type == "openyuanrong":
        upstream = extract_upstream(gateway_url) if gateway_url else ""
        return await ClaudeYRSandbox.create(image=image, sidecar_image=sidecar_image, upstream=upstream)
    raise ValueError(f"Unknown sandbox_type={sandbox_type!r}; expected 'local' or 'openyuanrong'")
