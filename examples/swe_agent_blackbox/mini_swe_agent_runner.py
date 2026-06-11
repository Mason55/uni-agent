"""Mini-swe-agent runner for the blackbox SWE-agent recipe.

In-sandbox execution model: mini-swe-agent is mounted as a sidecar tool image
into the sandbox and runs *inside* the sandbox using LocalEnvironment (local
bash). The external runner creates the sandbox, triggers the agent, and
evaluates reward — all through the unified ``SandboxCommands`` interface.

Task config is piped to run_agent.py via stdin (base64 encoded to avoid shell
escaping issues). Result JSON is returned via stdout (logging goes to stderr).
This eliminates extra file write/read round trips.

Supported sandbox types:
  - ``"local"``: local Docker with sidecar bind-mount
  - ``"openyuanrong"``: remote YR sandbox with akernel_sdk.Mount
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path

from uni_agent.trainer.framework.types import SessionHandle, SessionRuntime

from examples.swe_agent_blackbox.dataset import extract_image
from examples.swe_agent_blackbox.reward import build_reward_context, evaluate_in_env
from examples.swe_agent_blackbox.sandbox import CommandResult

logger = logging.getLogger(__name__)
if os.environ.get("DEBUG_MODE"):
    logger.setLevel(logging.DEBUG)


# Default sidecar image — user can change this or set MINI_SWE_AGENT_IMAGE env var.
MINI_SWE_AGENT_IMAGE = os.environ.get("MINI_SWE_AGENT_IMAGE", "swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest")


async def _create_sandbox(
    *,
    image: str,
    sandbox_type: str,
    sidecar_image: str = MINI_SWE_AGENT_IMAGE,
    gateway_url: str = "",
):
    """Create a sandbox with the mini-swe-agent sidecar tool mounted."""
    if sandbox_type == "local":
        from examples.swe_agent_blackbox.sandbox.docker_sandbox import DockerSandboxCommands

        return await DockerSandboxCommands.create(image=image, sidecar_image=sidecar_image)
    elif sandbox_type == "openyuanrong":
        from examples.swe_agent_blackbox.sandbox.yr_sandbox import YRSandboxCommands, extract_upstream

        upstream = extract_upstream(gateway_url) if gateway_url else ""
        return await YRSandboxCommands.create(
            image=image, sidecar_image=sidecar_image, upstream=upstream,
        )
    else:
        raise ValueError(f"Unknown sandbox_type={sandbox_type!r}. Supported: 'local', 'openyuanrong'")

# =====================================================================
# SandboxEnvForReward — adapts SandboxCommands to reward spec interface
# =====================================================================


class SandboxEnvForReward:
    """Adapts :class:`SandboxCommands` to the async env interface used by
    reward specs (``communicate``, ``write_file``, ``read_file``).

    Drop-in replacement for the old ``DockerEnvForReward``.
    """

    def __init__(self, sandbox):
        self._sandbox = sandbox

    async def communicate(self, input: str, timeout=60, check="ignore", error_msg="Command failed") -> str:
        result = await self._sandbox.run(input, timeout=int(timeout))
        if check == "raise" and result.exit_code != 0:
            raise RuntimeError(f"{error_msg}: {result.stdout[:200]}")
        return result.stdout

    async def write_file(self, path: str | Path, content: str) -> None:
        encoded = base64.b64encode(content.encode()).decode()
        await self.communicate(f"echo {encoded} | base64 -d > {path}", check="raise", error_msg=f"write {path}")

    async def read_file(self, path: str | Path, **_) -> str:
        return await self.communicate(f"cat {path}")


# =====================================================================
# Task config builder
# =====================================================================


def _extract_task(raw_prompt) -> str:
    """Extract task text from raw_prompt (str or message list)."""
    if isinstance(raw_prompt, str):
        return raw_prompt
    return next(
        (m["content"] for m in raw_prompt if isinstance(m, dict) and m.get("role") == "user"),
        str(raw_prompt),
    )


def _build_task_config(
    *,
    task: str,
    gateway_url: str,
    agent_step_limit: int = 250,
) -> dict:
    """Build the task config passed to run_agent.py via stdin."""
    return {
        "task": task,
        "gateway_url": gateway_url,
        "agent": {
            "step_limit": agent_step_limit,
            "cost_limit": 0,
        },
    }


# =====================================================================
# Agent runner
# =====================================================================


async def mini_swe_agent_runner(
    *,
    raw_prompt,
    session: SessionHandle,
    sample_index: int,
    session_runtime: SessionRuntime,
    tools_kwargs: dict | None = None,
    **kwargs,
) -> None:
    """Run mini-swe-agent inside a sandbox with sidecar tool mount.

    Flow:
        1. Create sandbox (Docker or YR) with mini-swe-agent sidecar
        2. Pipe task config to run_agent.py via stdin
        3. Parse agent result from stdout
        4. Evaluate reward in the same sandbox
        5. Complete session with reward_info
    """
    tools_kwargs = tools_kwargs or {}
    logger.info("mini_swe_agent_runner called, sample_index=%d", sample_index)

    # 1. Extract task text
    task = _extract_task(raw_prompt)
    logger.info("task extracted, %d chars", len(task))

    # 2. Dataset config (image from parquet)
    env_config = tools_kwargs.get("env", {})
    image = extract_image(env_config)
    if not image:
        raise ValueError(f"No Docker image found in tools_kwargs.env for sample {sample_index}")

    # 3. Sandbox deployment config (hardcoded + global env var)
    sandbox_type = os.environ.get("SWE_AGENT_SANDBOX_TYPE", "openyuanrong")

    # 4. Gateway URL — get early to pass upstream for YR tunnel
    gateway_url = session.base_url
    if not gateway_url:
        raise ValueError(f"gateway_url is empty for sample {sample_index}")
    with open(f"/tmp/gateway_url_{sample_index}.txt", "w") as f:
        f.write(gateway_url)

    sandbox = await _create_sandbox(
        image=image,
        sandbox_type=sandbox_type,
        sidecar_image=MINI_SWE_AGENT_IMAGE,
        gateway_url=gateway_url,
    )
    sandbox_id = sandbox.sandbox_id
    logger.info("Sandbox created (type=%s, image=%s, sandbox_id=%s)", sandbox_type, image, sandbox_id)

    # 5. Build task config — for YR, rewrite gateway URL to use tunnel
    step_limit = int(os.environ.get("SWE_AGENT_MAX_TURNS", "250"))
    if sandbox_type == "openyuanrong":
        from examples.swe_agent_blackbox.sandbox.yr_sandbox import rewrite_gateway_url
        agent_gateway_url = rewrite_gateway_url(gateway_url)
        print(f"Tunnel gateway URL: {agent_gateway_url}", flush=True)
    else:
        agent_gateway_url = gateway_url
    task_config = _build_task_config(
        task=task,
        gateway_url=agent_gateway_url,
        agent_step_limit=step_limit,
    )

    try:
        # 5. Run post_setup_cmd if provided
        post_setup_cmd = env_config.get("post_setup_cmd", "")
        if post_setup_cmd:
            logger.info("Running post_setup_cmd (%d chars)...", len(post_setup_cmd))
            r = await sandbox.run(post_setup_cmd, timeout=120)
            if r.exit_code != 0:
                logger.warning("post_setup_cmd failed (rc=%d): %s", r.exit_code, r.stdout[:200])
            else:
                logger.info("post_setup_cmd done")

        # 6. Run agent inside sandbox — pipe config via stdin, read result from stdout
        #    base64 encode the config JSON to avoid shell escaping issues
        config_b64 = base64.b64encode(json.dumps(task_config).encode()).decode()
        agent_cmd = (
            "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy; "
            f"echo {config_b64} | base64 -d | "
            "/opt/mini-swe-agent/bin/python /opt/mini-swe-agent/bin/run_agent.py"
        )
        logger.debug("[sample %d] starting agent inside sandbox", sample_index)
        t0 = time.perf_counter()
        agent_result = await sandbox.run(agent_cmd, timeout=7200)
        elapsed = time.perf_counter() - t0
        logger.debug(
            "[sample %d] agent process finished: rc=%d (%.1fs)",
            sample_index, agent_result.exit_code, elapsed,
        )
        with open(f"/tmp/agent_stdout_{sample_index}.txt", "w") as f:
            f.write(agent_result.stdout)
        with open(f"/tmp/agent_stderr_{sample_index}.txt", "w") as f:
            f.write(agent_result.stderr)

        # 7. Parse result from stdout (last non-empty line is the result JSON)
        agent_info = _parse_agent_result(agent_result.stdout, sample_index)
        logger.info(
            "[sample %d] agent: exit_status=%s, submission=%d chars",
            sample_index, agent_info.get("exit_status"),
            len(agent_info.get("submission", "")),
        )

        # 8. Evaluate reward (external, in the same sandbox)
        metadata, eval_timeout = build_reward_context(tools_kwargs)
        t0 = time.perf_counter()
        reward_env = SandboxEnvForReward(sandbox)
        score, eval_result = await evaluate_in_env(reward_env, metadata, eval_timeout)
        logger.debug(
            "[sample %d] reward done: score=%s, resolved=%s (%.1fs)",
            sample_index, score, eval_result.get("resolved"), time.perf_counter() - t0,
        )

        # 9. Signal completion with reward_info
        reward_info = {"reward_score": score, **eval_result}
        await session_runtime.complete_session(session.session_id, reward_info=reward_info)

    except Exception as e:
        logger.warning("Mini-swe-agent runner failed for sample %d (sandbox_id=%s): %s", sample_index, sandbox_id, e)
        raise
    finally:
        try:
            await sandbox.cleanup()
        except Exception:
            pass


def _parse_agent_result(stdout: str, sample_index: int) -> dict:
    """Parse agent result JSON from run_agent.py stdout.

    litellm may print error messages to stdout, polluting the output.
    The last line starting with '{' is the result JSON.
    """
    stdout = stdout.strip()
    if not stdout:
        return {"exit_status": "error", "submission": ""}
    # Try the last line that looks like JSON first
    lines = [l.strip() for l in stdout.split("\n") if l.strip()]
    for line in reversed(lines):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    # Fallback: try entire stdout
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        logger.warning("[sample %d] Failed to parse agent result (full stdout): %s", sample_index, stdout[:1000])
        return {"exit_status": "error", "submission": ""}
