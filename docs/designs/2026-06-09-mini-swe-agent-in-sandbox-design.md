# Feature Design: Mini-SWE-Agent In-Sandbox Execution

> Date: 2026-06-09
> Status: Draft
> Author: erpim

## 1. Background & Motivation

### 1.1 Current Architecture

mini-swe-agent runs **outside** the sandbox (on the rollouter host). It creates a `DockerEnvironment` which wraps `subprocess.run(["docker", "exec", ...])` to execute commands in a separate Docker container.

```
[Rollouter Host: mini_swe_agent_runner]
  ├── _FixedCmdDockerEnvironment → docker run + docker exec (subprocess)
  ├── LitellmModel → session.base_url (gateway)
  ├── DefaultAgent(docker_env, model)
  │     └── agent loop: LLM call (HTTP) + docker exec (subprocess)
  ├── DockerEnvForReward → docker exec (subprocess)
  └── session.complete(reward_info)
```

**问题：**

1. **Docker 耦合**: `DockerEnvironment` 直接调用本地 `docker` CLI，无法用于远程沙箱（OpenYuanRong、Modal 等）
2. **不一致**: uniagent runner 通过 swerex HTTP 与沙箱交互，mini-swe-agent 通过 docker exec，两套路径并存
3. **扩展受限**: 新增沙箱类型需要改动 runner 代码

### 1.2 Target Architecture

mini-swe-agent 打包为 **sidecar 工具镜像**，在创建沙箱时挂载到沙箱内部。agent 在沙箱内以"本地"模式运行 — 命令通过本地 bash 执行，LLM 调用走注入的 gateway URL。

```
[Rollouter Host: mini_swe_agent_runner]
  ├── Creates SandboxHandle (unified interface)
  │     ├── sidecar mount: /opt/mini-swe-agent ← tool image
  │     └── sandbox image: dataset image (e.g. swebench image)
  ├── sandbox.write_file("/tmp/task.json", {task, gateway_url, ...})
  ├── sandbox.run("/opt/mini-swe-agent/bin/run_agent.py ...")
  │     └── [Inside Sandbox]
  │           ├── Read task.json → gateway_url
  │           ├── LocalEnvironment (bash -c, no docker)
  │           ├── LitellmModel(gateway_url)
  │           ├── DefaultAgent(local_env, model)
  │           │     └── agent loop: LLM call (HTTP) + local bash (subprocess)
  │           └── Writes result → /tmp/agent_result.json
  ├── sandbox.read_file("/tmp/agent_result.json")
  ├── SandboxEnvForReward(sandbox).evaluate()
  └── session.complete(reward_info)
```

### 1.3 Goals

1. **统一**: 同一份 runner 代码适用于 local Docker、OpenYuanRong、Modal 及任何未来沙箱类型
2. **可移植**: Agent 视角下无需依赖 Docker API，所有命令本地执行
3. **解耦**: Agent 执行自包含在沙箱内部，外部只需触发 + 收集结果
4. **可维护**: 沙箱管理与 agent 逻辑清晰分离

## 2. Components

### 2.1 Tool Image: mini-swe-agent sidecar

参考 ossutil sidecar 模式，构建一个仅包含 mini-swe-agent 运行时的镜像，在创建沙箱时挂载。

由于 mini-swe-agent 是 Python 包，无法像 ossutil 一样做成 `FROM scratch` 的静态二进制。采用 **Python venv** 方案：

```dockerfile
# Dockerfile.mini-swe-agent-tool
FROM python:3.12-slim AS builder

RUN python -m venv /opt/mini-swe-agent
RUN /opt/mini-swe-agent/bin/pip install --no-cache-dir \
    minisweagent>=2.2.0 \
    litellm

COPY run_agent.py /opt/mini-swe-agent/bin/run_agent.py
RUN chmod +x /opt/mini-swe-agent/bin/run_agent.py

FROM scratch
COPY --from=builder /opt/mini-swe-agent /opt/mini-swe-agent
```

**最终镜像内容：**

```
/opt/mini-swe-agent/
├── bin/
│   ├── python                # Python 3.12 interpreter
│   ├── pip
│   └── run_agent.py          # Agent entrypoint (in-sandbox runner script)
├── lib/
│   └── python3.12/
│       └── site-packages/    # minisweagent, litellm, jinja2, etc.
└── pyvenv.cfg
```

镜像预期大小：200-400MB。

**版本管理：** 镜像 tag 对应 minisweagent 版本，如 `reg.antgroup-inc.cn/<repo>/mini-swe-agent-tool:2.2.8`。

### 2.2 In-Sandbox Runner Script: `run_agent.py`

该脚本运行在沙箱内部，使用 `LocalEnvironment`（本地 bash 执行）替代 `DockerEnvironment`（docker exec），使用 `LitellmModel` 对接注入的 gateway URL。

**接口：**

```bash
/opt/mini-swe-agent/bin/python \
    /opt/mini-swe-agent/bin/run_agent.py \
    --task-file /tmp/task.json \
    --result-file /tmp/agent_result.json
```

**task.json 格式：**

```json
{
  "task": "Consider the following PR description:\n...",
  "gateway_url": "http://172.17.0.1:8000/v1",
  "agent": {"step_limit": 250, "cost_limit": 0},
  "environment": {"cwd": "/testbed", "timeout": 60},
  "model": {"model_name": "openai/default"}
}
```

`gateway_url` 由 runner 在沙箱外部根据 `session.base_url` + 沙箱类型计算好（见 §4），随 task config 一起写入沙箱。不使用环境变量注入。

**agent_result.json 格式：**

```json
{
  "exit_status": "Submitted",
  "submission": "diff --git a/...",
  "model_stats": {"instance_cost": 0.0, "api_calls": 15}
}
```

**核心逻辑（伪代码）：**

```python
#!/opt/mini-swe-agent/bin/python
"""Run mini-swe-agent inside the sandbox using LocalEnvironment."""

import json
import os
import sys

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

def main():
    task_file = parse_arg("--task-file")
    result_file = parse_arg("--result-file")

    config = json.load(open(task_file))
    gateway_url = config["gateway_url"]

    # LocalEnvironment: commands run via local bash (subprocess.run, shell=True)
    env = LocalEnvironment(**config.get("environment", {}))

    # LitellmModel: LLM calls go to injected gateway
    model = LitellmModel(
        model_name=config.get("model", {}).get("model_name", "openai/default"),
        model_kwargs={
            "api_base": gateway_url,
            "api_key": "not-needed",
            "drop_params": True,
        },
        cost_tracking="ignore_errors",
    )

    # DefaultAgent with step_limit / cost_limit from config
    agent_cfg = config.get("agent", {})
    step_limit = int(os.environ.get("SWE_AGENT_MAX_TURNS", str(agent_cfg.get("step_limit", 250))))
    agent = DefaultAgent(model, env, step_limit=step_limit, cost_limit=0, **other_cfg)

    info = agent.run(task=config["task"])

    # Write result JSON for external collection
    with open(result_file, "w") as f:
        json.dump(info, f)

if __name__ == "__main__":
    main()
```

**关键设计决策：**

- `LocalEnvironment` 使用 `subprocess.run(command, shell=True)` 执行命令，对 agent 而言完全等同于本地执行
- `swebench.yaml` 的 `system_template` / `instance_template` / `observation_template` 等需要内嵌到 `run_agent.py` 中（因为 `minisweagent.config.builtin_config_dir` 在 sidecar venv 内可用）
- `gateway_url` 从 `task.json` 的 `gateway_url` 字段读取，由 runner 根据沙箱类型计算后写入

### 2.3 Unified Sandbox Command Interface

统一不同沙箱类型的命令执行接口，替换当前的 `DockerEnvironment` / `DockerEnvForReward`。

**Protocol：**

```python
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int


@runtime_checkable
class SandboxCommands(Protocol):
    """Unified interface for executing commands in a sandbox.
    
    Implementations only need to provide run() + cleanup().
    write_file / read_file are provided by SandboxCommandsBase mixin.
    """

    async def run(self, cmd: str, *, timeout: int = 60) -> CommandResult:
        """Execute a command and return the result. The ONLY primitive."""
        ...

    async def cleanup(self) -> None:
        """Destroy sandbox resources."""
        ...


class SandboxCommandsBase:
    """Convenience file I/O built on top of run().
    
    Subclasses implement run() + cleanup(); get write_file / read_file for free.
    """

    async def write_file(self, path: str | Path, content: str) -> None:
        encoded = base64.b64encode(content.encode()).decode()
        r = await self.run(f"echo {encoded} | base64 -d > {path}", timeout=10)
        if r.exit_code != 0:
            raise RuntimeError(f"write_file {path} failed: {r.stderr}")

    async def read_file(self, path: str | Path) -> str:
        r = await self.run(f"cat {path}", timeout=10)
        if r.exit_code != 0:
            raise RuntimeError(f"read_file {path} failed: {r.stderr}")
        return r.stdout
```

#### 2.3.1 DockerSandboxCommands（Local Docker）

适用于本地 Docker 环境。sidecar 通过 `docker cp` 注入。

```python
class DockerSandboxCommands:
    """Command execution via local Docker (docker exec)."""

    @classmethod
    async def create(
        cls,
        *,
        image: str,
        sidecar_image: str,
        env: dict[str, str] | None = None,
        cwd: str = "/",
        container_timeout: str = "2h",
    ) -> "DockerSandboxCommands":
        # 1. Extract sidecar to host cache path (one-time, cached)
        tool_dir = await _extract_sidecar(sidecar_image)

        # 2. Start sandbox container with bind mount + env
        env_flags = [f"-e {k}={v}" for k, v in (env or {}).items()]
        container_id = subprocess.run(
            ["docker", "run", "-d", "-w", cwd,
             "-v", f"{tool_dir}:/opt/mini-swe-agent:ro",
             *env_flags, "--rm", image, "sleep", container_timeout],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

        return cls(container_id=container_id, cwd=cwd)

    async def run(self, cmd: str, *, timeout: int = 60) -> CommandResult:
        result = await asyncio.to_thread(
            subprocess.run,
            ["docker", "exec", "-w", self._cwd, self._container_id,
             "bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return CommandResult(stdout=result.stdout, stderr=result.stderr,
                             exit_code=result.returncode)
```

**Sidecar 注入方式（Local Docker）：**

```bash
# 1. 首次使用时从 tool image 提取到 host 缓存
docker create --name _tool_cache mini-swe-agent-tool
docker cp _tool_cache:/opt/mini-swe-agent /tmp/.mini-swe-agent-tool
docker rm _tool_cache

# 2. 启动沙箱时 bind mount（只读）
docker run -d \
    -v /tmp/.mini-swe-agent-tool:/opt/mini-swe-agent:ro \
    swebench-image sleep 2h
```

#### 2.3.2 YRSandboxCommands（OpenYuanRong）

适用于远程沙箱。sidecar 通过 `akernel_sdk.Mount` 挂载。

```python
class YRSandboxCommands:
    """Command execution via akernel_sdk sandbox."""

    @classmethod
    async def create(
        cls,
        *,
        image: str,
        sidecar_image: str,
        env: dict[str, str] | None = None,
        cwd: str = "/",
        cpu: int = 4000,
        memory: int = 8192,
        **sandbox_kwargs,
    ) -> "YRSandboxCommands":
        from akernel_sdk import Mount, Sandbox

        sandbox = await asyncio.to_thread(
            lambda: Sandbox(
                image=image,
                cwd=cwd,
                cpu=cpu,
                memory=memory,
                env=env,
                mounts=[
                    Mount(target="/opt/mini-swe-agent", image_url=sidecar_image),
                ],
                **sandbox_kwargs,
            )
        )
        return cls(sandbox=sandbox)

    async def run(self, cmd: str, *, timeout: int = 60) -> CommandResult:
        result = await asyncio.to_thread(
            self._sandbox.commands.run, cmd, timeout=timeout
        )
        return CommandResult(stdout=result.stdout, stderr=result.stderr,
                             exit_code=result.exit_code)
```

**Sidecar 注入方式（OpenYuanRong）：**

```python
from akernel_sdk import Mount, Sandbox

Sandbox(
    image="swebench-image",
    mounts=[
        Mount(target="/opt/mini-swe-agent", image_url="mini-swe-agent-tool:2.2.8"),
    ],
)
```

#### 2.3.3 SandboxFactory

统一工厂，根据配置创建对应类型的 `SandboxCommands` 实现：

```python
MINI_SWE_AGENT_IMAGE = "reg.antgroup-inc.cn/<repo>/mini-swe-agent-tool:latest"

class SandboxFactory:
    @staticmethod
    async def create(
        *,
        image: str,
        sandbox_type: str = "local",  # "local" | "openyuanrong"
        sidecar_image: str = MINI_SWE_AGENT_IMAGE,
        env: dict[str, str] | None = None,
        cwd: str = "/",
        **kwargs,
    ) -> SandboxCommands:
        if sandbox_type == "local":
            return await DockerSandboxCommands.create(
                image=image, sidecar_image=sidecar_image,
                env=env, cwd=cwd, **kwargs,
            )
        elif sandbox_type == "openyuanrong":
            return await YRSandboxCommands.create(
                image=image, sidecar_image=sidecar_image,
                env=env, cwd=cwd, **kwargs,
            )
        else:
            raise ValueError(f"Unknown sandbox_type: {sandbox_type}")
```

### 2.4 Reward Evaluation Adapter

Reward 评估保持在外部执行（由 rollouter 通过 sandbox 命令接口远程执行），但使用统一的 `SandboxCommands` 接口替代当前的 `DockerEnvForReward`。

```python
class SandboxEnvForReward:
    """Adapts SandboxCommands to async env interface for reward specs.

    Drop-in replacement for DockerEnvForReward, works with any SandboxCommands implementation.
    """

    def __init__(self, sandbox: SandboxCommands):
        self._sandbox = sandbox

    async def communicate(self, input: str, timeout=60, check="ignore", error_msg="Command failed") -> str:
        result = await self._sandbox.run(input, timeout=int(timeout))
        if check == "raise" and result.exit_code != 0:
            raise RuntimeError(f"{error_msg}: {result.stdout[:200]}")
        return result.stdout

    async def write_file(self, path: str | Path, content: str) -> None:
        encoded = base64.b64encode(content.encode()).decode()
        await self.communicate(f"echo {encoded} | base64 -d > {path}", check="raise")

    async def read_file(self, path: str | Path, **_) -> str:
        return await self.communicate(f"cat {path}")
```

## 3. Data Flow

### 3.1 Complete Execution Flow

```
1. mini_swe_agent_runner(raw_prompt, session, ...)
   │
   ├── 2. Resolve gateway_url for sandbox type
   │     └── e.g., session.base_url → "http://172.17.0.1:8000/v1" (local docker)
   │
   ├── 3. SandboxFactory.create(image, sandbox_type)
   │     ├── Local Docker: extract sidecar → docker run -v ... bind mount
   │     └── YR: Sandbox(image, mounts=[Mount(...)])
   │
   ├── 4. sandbox.write_file("/tmp/task.json", {task, gateway_url, agent_config, ...})
   │
   ├── 5. [Optional] sandbox.run(post_setup_cmd)
   │
   ├── 6. sandbox.run("/opt/mini-swe-agent/bin/python /opt/mini-swe-agent/bin/run_agent.py \
   │                    --task-file /tmp/task.json --result-file /tmp/agent_result.json",
   │                    timeout=1800)
   │     │
   │     └── [Inside Sandbox - run_agent.py]
   │           ├── Read task.json → extract gateway_url
   │           ├── LocalEnvironment(cwd="/testbed")
   │           ├── LitellmModel(api_base=gateway_url)
   │           ├── DefaultAgent(env, model).run(task)
   │           │     └── loop: LLM → bash → LLM → bash → ... → submit
   │           └── Write agent_result.json
   │
   ├── 7. sandbox.read_file("/tmp/agent_result.json") → agent_info
   │
   ├── 8. SandboxEnvForReward(sandbox) → evaluate_in_env()
   │     └── sandbox.run("bash /tmp/eval_script_xxx.sh", timeout=600)
   │
   ├── 9. session_runtime.complete_session(reward_info)
   │
   └── 10. sandbox.cleanup()
```

### 3.2 Timing & Timeout

| Phase | Timeout | Notes |
|-------|---------|-------|
| Sandbox creation | 120s | YR sandbox startup |
| Task config write | 10s | Small file |
| Agent execution | 1800s (30min) | `SWE_AGENT_MAX_TURNS` × ~4min/turn |
| Reward evaluation | 600s | `SWE_AGENT_EVAL_TIMEOUT` |
| Total per sample | ~2700s | |

## 4. Gateway URL Resolution

gateway URL 由 runner 在沙箱外部根据 `session.base_url` + 沙箱类型计算，写入 `task.json` 的 `gateway_url` 字段。`run_agent.py` 从 config 文件读取，无需环境变量。

不同沙箱类型的 URL 计算方式：

| Sandbox Type | URL Calculation | Example |
|---|---|---|
| Local Docker | Host IP via docker bridge or `host.docker.internal` | `http://172.17.0.1:8000/v1` |
| OpenYuanRong | Port forwarding URL via `sandbox.get_port_url()` | `https://<id>.yr.antgroup-inc.cn:8000/v1` |
| Modal | Internal service networking | TBD |

```python
def resolve_gateway_url(session: SessionHandle, sandbox_type: str) -> str:
    """Compute the gateway URL reachable from inside the sandbox."""
    base = session.base_url  # e.g., http://127.0.0.1:8000

    if sandbox_type == "local":
        # Inside Docker container → reach host via bridge IP
        host_ip = _get_docker_host_ip()  # 172.17.0.1 or host.docker.internal
        return base.replace("127.0.0.1", host_ip).replace("localhost", host_ip)
    elif sandbox_type == "openyuanrong":
        # YR sandbox → use port forwarding URL
        return base  # or sandbox.get_port_url(...) if gateway is port-forwarded
    else:
        return base
```

**注意：** YR 场景下 gateway 的可达性取决于部署拓扑（port forwarding 或内网直达），需在集成时根据实际网络环境调整。

## 5. Configuration

### 5.1 YAML Config

在现有 `swe_agent_blackbox.yaml` 的 `agent_framework` 下增加 `sandbox` 配置：

```yaml
actor_rollout_ref:
  rollout:
    custom:
      agent_framework:
        agent_runner_fqn: examples.swe_agent_blackbox.mini_swe_agent_runner.mini_swe_agent_runner
        # ... existing config ...

        # NEW: sandbox configuration
        sandbox:
          type: local  # "local" | "openyuanrong"
          sidecar_image: "reg.antgroup-inc.cn/<repo>/mini-swe-agent-tool:2.2.8"
          container_timeout: "2h"
          # type-specific options
          openyuanrong:
            cpu: 4000
            memory: 8192
          local:
            docker_executable: docker
```

### 5.2 Deployment-Specific Override

通过环境变量覆盖 sandbox 类型（类似现有的 `OPENYUANRONG_SERVER_ADDRESS` 模式）：

```bash
# Local Docker (default)
SWE_AGENT_SANDBOX_TYPE=local

# OpenYuanRong
SWE_AGENT_SANDBOX_TYPE=openyuanrong \
OPENYUANRONG_SERVER_ADDRESS=6.2.179.37:8888 \
OPENYUANRONG_TOKEN=xxx
```

## 6. Files to Create/Modify

### 6.1 New Files

| File | Description |
|------|-------------|
| `examples/swe_agent_blackbox/sandbox/` | Sandbox abstraction package |
| `examples/swe_agent_blackbox/sandbox/__init__.py` | Exports `SandboxCommands`, `SandboxFactory` |
| `examples/swe_agent_blackbox/sandbox/protocol.py` | `SandboxCommands` protocol + `CommandResult` |
| `examples/swe_agent_blackbox/sandbox/docker_sandbox.py` | `DockerSandboxCommands` implementation |
| `examples/swe_agent_blackbox/sandbox/yr_sandbox.py` | `YRSandboxCommands` implementation |
| `examples/swe_agent_blackbox/sandbox/factory.py` | `SandboxFactory` + sidecar extraction logic |
| `examples/swe_agent_blackbox/sandbox/reward_env.py` | `SandboxEnvForReward` adapter |
| `examples/swe_agent_blackbox/run_agent.py` | In-sandbox runner script |
| `examples/swe_agent_blackbox/Dockerfile.mini-swe-agent-tool` | Tool image build file |

### 6.2 Modified Files

| File | Change |
|------|--------|
| `mini_swe_agent_runner.py` | Rewrite to use `SandboxFactory` + `run_agent.py` 代替 `_FixedCmdDockerEnvironment` |
| YAML config files | 增加 `sandbox` 配置段 |

### 6.3 Removed Code

| Code | Reason |
|------|--------|
| `_FixedCmdDockerEnvironment` | 被 `DockerSandboxCommands` 替代 |
| `DockerEnvForReward` | 被 `SandboxEnvForReward` 替代 |
| 直接 import `minisweagent.*` in runner | Agent 逻辑移入 `run_agent.py`，runner 不再需要 minisweagent 依赖 |

## 7. Migration Path

### Phase 1: Infrastructure（无行为变化）
1. 创建 `SandboxCommands` protocol
2. 实现 `DockerSandboxCommands`（封装现有 docker exec 逻辑）
3. 实现 `SandboxEnvForReward`
4. 构建 tool image + `run_agent.py`

### Phase 2: Runner Rewrite
5. 重写 `mini_swe_agent_runner` 使用 `SandboxFactory` + `run_agent.py`
6. 验证 local Docker 路径功能等价

### Phase 3: Remote Sandbox
7. 实现 `YRSandboxCommands`
8. 端到端 YR 沙箱测试

### Phase 4: Production
9. 性能调优（sidecar 缓存、并发控制）
10. 端到端异步训练验证

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Python venv 跨镜像兼容性 | sidecar 中的 Python 可能与 sandbox 基础镜像的 glibc 版本不兼容 | tool image 使用 `python:3.12-slim`（Debian based），与 SWE-bench 镜像兼容。测试时验证 glibc 版本 |
| Tool image 体积过大 | 200-400MB 增加沙箱创建时间 | 使用多阶段构建精简；YR 场景下镜像缓存在 registry |
| `run_agent.py` 执行超时 | agent loop 可能耗时很长 | 使用足够大的 timeout（1800s+）；runner 支持超时信号传递 |
| Gateway URL 在 YR 沙箱内不可达 | 网络拓扑可能阻止沙箱 → gateway 的连接 | 测试时验证连通性；备选方案：gateway 也通过 port forwarding 暴露 |
| `LocalEnvironment` 的 `shell=True` 安全性 | 不构成额外风险，因为沙箱本身就是隔离环境 | 无需额外处理 |

## 9. Alternative Approaches Considered

### 9.1 pip install in Sandbox (Rejected)

沙箱启动后通过 `post_setup_cmd` 执行 `pip install minisweagent`。

**优点：** 无需维护 tool image，最简实现。
**缺点：** 每次启动都需要网络下载，增加延迟 30-120s；依赖网络可用性，不适合离线环境。
**结论：** 不可靠，不适合生产环境。

### 9.2 Agent Runs Outside, Command Interface Unified (Rejected)

保持 mini-swe-agent 在外部运行，只统一命令执行接口（用 `SandboxCommands` 替代 `DockerEnvironment`）。

**优点：** 改动最小，agent 逻辑不变。
**缺点：** 外部 agent 仍需维护 minisweagent 依赖；agent → sandbox 的每次命令都经过网络（延迟更高）；不符合用户"agent 在沙箱内部"的设计目标。
**结论：** 不符合设计目标。
