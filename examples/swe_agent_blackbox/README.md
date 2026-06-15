# Mini-SWE-Agent In-Sandbox Execution — 使用指南

## 概述

mini-swe-agent 和 Claude Code 均以 sidecar 工具镜像的形式挂载到沙箱内部运行。Agent 在沙箱内通过
`LocalEnvironment`（本地 bash）执行命令，LLM 调用走 stdin 传入的 gateway URL。
外部 runner 创建沙箱、触发 agent 执行、评估 reward。

mini_swe 工具镜像使用 [python-build-standalone](https://github.com/astral-sh/python-build-standalone) 构建独立 Python 环境；Claude Code 工具镜像使用 Node builder 安装 npm 包。两者都通过 `FROM scratch` 保持最终 sidecar 镜像最小化，不依赖沙箱基础镜像预装对应运行时。

**支持的 runner：**

| runner | 说明 |
|--------|------|
| `uniagent` | 原 SWE-agent runner |
| `mini_swe` | mini-swe-agent sidecar runner |
| `claude_code` | Claude Code sidecar runner，reward 通过 `complete_session(reward_info)` 返回，不额外落盘 reward JSON |

**支持的沙箱类型：**

| 类型 | 说明 |
|------|------|
| 本地 Docker (`"local"`) | `docker exec` + `docker cp` 从工具镜像加载 sidecar |
| OpenYuanRong (`"openyuanrong"`) | `akernel_sdk.Mount` + `sandbox.commands.run()` |

两种沙箱类型都直接依赖所选 runner 的工具镜像，无需预先提取到 host 目录。

## 架构

```
[Rollouter Host: mini_swe_agent_runner]
  │
  ├── _create_sandbox(image, sandbox_type, sidecar_image)
  │     ├── local:     docker run + docker cp from sidecar image
  │     └── openyuanrong: Sandbox(mounts=[Mount(target="/opt/mini-swe-agent", ...)])
  │
  ├── sandbox.run("echo <b64_config> | base64 -d | /opt/.../python run_agent.py")
  │     └── [Inside Sandbox]
  │           /opt/mini-swe-agent/bin/python3.12  ← 独立 Python，不影响沙箱原有版本
  │           stdin ← task config JSON (task, gateway_url, agent)
  │           LocalEnvironment + LitellmModel(gateway_url) → DefaultAgent
  │           stdout → result JSON (exit_status, submission, model_stats)
  │
  ├── _parse_agent_result(stdout)
  ├── SandboxEnvForReward(sandbox) → evaluate_in_env()
  └── session_runtime.complete_session(reward_info)
```

## 前置条件

1. **Docker** — 本地模式需要
2. **runner tool image** — 需预先构建（见下文）

## 1. 构建 Tool Image

`mini_swe` 和 `claude_code` 都通过 sidecar tool image 注入到 SWE-bench 沙箱，但两者的镜像内容、挂载目录和加速源参数不同。统一使用 `build_tool.sh` 构建，通过 `--tool` 或 `TOOL_KIND` 选择目标 runner。

| runner | 默认 tool image | Dockerfile | 沙箱内目录 | 镜像内容 | 加速源 |
|--------|-----------------|------------|------------|----------|--------|
| `mini_swe` | `mini-swe-agent-tool:latest` | `Dockerfile.mini-swe-agent-tool` | `/opt/mini-swe-agent` | 独立 Python 3.12 + `mini-swe-agent` + `litellm` + `run_agent.py` | `--pip-index` / `PIP_INDEX_URL` |
| `claude_code` | `claude-code-tool:latest` | `Dockerfile.claude-code-tool` | `/opt/claude-code` | Node 20 构建出的 Claude Code npm 全局安装目录 | `--npm-registry` / `NPM_REGISTRY` |

### mini_swe tool image

默认构建的是 `mini_swe`：

```bash
# 默认 PyPI 源
bash examples/swe_agent_blackbox/build_tool.sh

# 使用国内 PyPI 镜像加速
bash examples/swe_agent_blackbox/build_tool.sh --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/

# 推送到远程仓库
bash examples/swe_agent_blackbox/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

mini_swe 镜像使用 `python-build-standalone` 构建独立 Python 环境，最终 `FROM scratch` 镜像只包含 `/opt/mini-swe-agent` 运行所需文件。它不依赖沙箱基础镜像里的 Python 版本。

推送到远程仓库后，运行时通过 `SWE_AGENT_TOOL_IMAGE` 指向该镜像：

```bash
SWE_AGENT_TOOL_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest \
RUNNER=mini_swe \
bash examples/swe_agent_blackbox/scripts/run_infer.sh
```

### Claude Code tool image

Claude Code 需要显式选择 `--tool claude_code`：

```bash
# 默认 npm registry
bash examples/swe_agent_blackbox/build_tool.sh --tool claude_code

# 使用国内 npm registry 加速
bash examples/swe_agent_blackbox/build_tool.sh \
    --tool claude_code \
    --npm-registry https://registry.npmmirror.com

# 指定 Claude Code npm 包版本
bash examples/swe_agent_blackbox/build_tool.sh \
    --tool claude_code \
    --tool-version latest

# 构建并推送 Claude Code sidecar
bash examples/swe_agent_blackbox/build_tool.sh \
    --tool claude_code \
    --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

Claude Code 镜像使用 `node:20-bookworm-slim` 作为 builder，安装 `@anthropic-ai/claude-code` 到 `/opt/claude-code`，最终同样输出 `FROM scratch` 镜像。运行时 runner 会把该镜像加载到沙箱的 `/opt/claude-code`，并调用 `/opt/claude-code/bin/claude`。

推送到远程仓库后，运行时同样通过 `SWE_AGENT_TOOL_IMAGE` 指向该镜像：

```bash
SWE_AGENT_TOOL_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest \
RUNNER=claude_code \
bash examples/swe_agent_blackbox/scripts/run_infer.sh
```

### 组合参数

`--tool`、镜像 tag、加速源和 registry 可以组合使用：

```bash
bash examples/swe_agent_blackbox/build_tool.sh \
    --tool mini_swe \
    --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

脚本会：
1. 按 `--tool` 选择 Dockerfile 和默认镜像名：
   - `mini_swe` → `mini-swe-agent-tool:latest`
   - `claude_code` → `claude-code-tool:latest`
2. 如果指定 `--registry`，自动 tag + push 到远程仓库

两个 tool image 都是 sidecar 运行时依赖，不是 SWE-bench 任务基础镜像。`mini_swe` 的 Python 与沙箱容器的 Python 完全隔离；`claude_code` 的 Node/npm 依赖也只存在于 `/opt/claude-code` sidecar 中，不要求沙箱基础镜像预装 Node。

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TOOL_IMAGE` | `mini-swe-agent-tool` / `claude-code-tool` | 镜像名，默认值随 `TOOL_KIND` 变化 |
| `TOOL_TAG` | `latest` | 镜像 tag |
| `TOOL_VERSION` | `latest` | 工具包版本；`claude_code` 构建时用于 `@anthropic-ai/claude-code` npm 包版本 |
| `PIP_INDEX_URL` | (空，使用 PyPI) | pip 镜像源（也可通过 `--pip-index` 传入） |
| `TOOL_KIND` | `mini_swe` | 工具类型：`mini_swe` 或 `claude_code` |
| `NPM_REGISTRY` | (空，使用 npm 默认源) | npm 镜像源（也可通过 `--npm-registry` 传入） |

## 2. 推理（本地 Docker 沙箱）

### 使用 run_infer.sh

```bash
cd /home/dyp/recipe/uni-agent

RUNNER=mini_swe \
SWE_AGENT_TOOL_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest \
MODEL_PATH=$HOME/models/Qwen3.5-9B \
DATA_PATH=$HOME/data/swe_agent/r2e_gym.parquet \
MAX_SAMPLES=1 \
TP=1 \
bash examples/swe_agent_blackbox/scripts/run_infer.sh
```

### 使用 Python 直接调用

```bash
python examples/swe_agent_blackbox/parallel_infer.py \
    --model-path ~/models/Qwen3.5-9B \
    --data-path ~/data/swe_agent/r2e_gym.parquet \
    --max-samples 1 \
    --runner mini_swe \
    --max-turns 100 \
    --tensor-parallel-size 1
```

### Claude Code runner

```bash
RUNNER=claude_code \
SWE_AGENT_TOOL_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest \
SWE_AGENT_SANDBOX_TYPE=openyuanrong \
SWE_AGENT_MAX_TURNS=50 \
SWE_AGENT_RUN_TIMEOUT=7200 \
MAX_SAMPLES=1 \
N=1 \
TP=4 \
GATEWAY_COUNT=1 \
MAX_CONCURRENT_SESSIONS=1 \
bash examples/swe_agent_blackbox/scripts/run_infer.sh
```

Claude Code 执行后直接在同一个 sandbox 内跑 reward，并通过 `session_runtime.complete_session(..., reward_info=...)` 回传；`parallel_infer.py` 从 TQ 汇总 `per_sample_scores`。

## 3. 推理（OpenYuanRong 远程沙箱）

### 环境变量

```bash
export OPENYUANRONG_SERVER_ADDRESS="6.2.179.37:8888"
export OPENYUANRONG_TOKEN="<your-token>"
export DEPLOYMENT=openyuanrong
```

### 运行

```bash
RUNNER=mini_swe \
OPENYUANRONG_SERVER_ADDRESS="6.2.179.37:8888" \
OPENYUANRONG_TOKEN="<token>" \
DEPLOYMENT=openyuanrong \
SWE_AGENT_TOOL_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest \
bash examples/swe_agent_blackbox/scripts/run_infer.sh
```

## 4. 训练

### 同步训练（main_ppo_sync）

在 YAML 配置中指定 `mini_swe_agent_runner`：

```yaml
actor_rollout_ref:
  rollout:
    custom:
      agent_framework:
        agent_runner_fqn: examples.swe_agent_blackbox.mini_swe_agent_runner.mini_swe_agent_runner
        gateway_count: 1
        completion_timeout_seconds: 600
```

```bash
bash examples/swe_agent_blackbox/scripts/run_train.sh
```

### 全异步训练（TQ 路径）

```bash
bash examples/swe_agent_blackbox/scripts/run_train_megatron_async.sh
```

## 5. 配置参数

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SWE_AGENT_SANDBOX_TYPE` | `"openyuanrong"` | 沙箱类型：`"local"` 或 `"openyuanrong"` |
| `SWE_AGENT_MAX_TURNS` | `250` | Agent 最大步数 |
| `SWE_AGENT_RUN_TIMEOUT` | `7200` | Agent 主进程超时（秒） |
| `SWE_AGENT_EVAL_TIMEOUT` | `600` | Reward 评估超时（秒） |
| `SWE_AGENT_TOOL_IMAGE` | runner 默认镜像 | sidecar 工具镜像；`mini_swe` 默认 `mini-swe-agent-tool`，`claude_code` 默认 `claude-code-tool` |
| `N_GPUS_PER_NODE` | `8` | 每节点 GPU 数量 |
| `DEBUG_MODE` | (unset) | 设为任意值开启 DEBUG 日志 |

### 数据集 tools_kwargs.env 字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `image` | (必填) | 沙箱基础镜像（SWE-bench dataset image） |
| `post_setup_cmd` | `""` | 可选的沙箱初始化命令 |

## 6. 文件结构

```
examples/swe_agent_blackbox/
├── sandbox/                          # 沙箱实现
│   ├── __init__.py                   # 导出 CommandResult
│   ├── docker_sandbox.py             # 本地 Docker 实现（docker cp 加载 sidecar）
│   └── yr_sandbox.py                 # OpenYuanRong 实现
├── build_tool.sh                     # 工具镜像构建脚本
├── run_agent.py                      # 沙箱内 agent 入口（stdin → stdout）
├── Dockerfile.mini-swe-agent-tool    # 工具镜像 Dockerfile（python-build-standalone + FROM scratch）
├── mini_swe_agent_runner.py          # Runner（外部触发 + reward 评估）
├── dataset.py                        # 数据集 + extract_image()
├── reward.py                         # Reward 计算
├── parallel_infer.py                 # 并行推理入口
├── config/                           # YAML 配置
│   ├── swe_agent_blackbox.yaml
│   ├── swe_agent_blackbox_megatron.yaml
│   ├── swe_agent_blackbox_megatron_async.yaml
│   ├── agent_config.yaml
│   └── parallel_infer.yaml
└── scripts/                          # 运行脚本
    ├── run_infer.sh
    ├── run_train.sh
    └── run_train_megatron_async.sh
```

## 7. 故障排查

### 构建时网络慢

构建过程涉及三个下载源，国内网络可能较慢，可使用镜像加速：

**apt 包（Dockerfile builder 阶段）：**
在 Dockerfile 的 `FROM debian:bullseye-slim` 之后添加：
```dockerfile
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
```

**python-build-standalone 下载：**
修改 Dockerfile 中的 wget URL，将 `github.com` 替换为 NJU 镜像：
```
原: https://github.com/astral-sh/python-build-standalone/releases/download/...
换: https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/...
```

**pip 包：**
使用 `--pip-index` 参数：
```bash
bash examples/swe_agent_blackbox/build_tool.sh --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/
```

### sidecar Python 无法运行（glibc 报错）

如果出现 `GLIBC_2.xx not found` 错误，说明工具镜像的 Python 与沙箱容器的 glibc 不兼容。
确保使用 `python-build-standalone` 构建工具镜像（当前默认），它基于较旧的 glibc 编译，
向前兼容。

### 工具镜像不存在

```
docker create: Error: No such image: mini-swe-agent-tool:latest
```

运行构建脚本：
```bash
bash examples/swe_agent_blackbox/build_tool.sh
```

### YR sandbox 创建失败

确认环境变量已设置：
```bash
echo $OPENYUANRONG_SERVER_ADDRESS
echo $OPENYUANRONG_TOKEN
```

### YR tunnel 连接超时

当 `OPENYUANRONG_SERVER_ADDRESS` 使用 443 端口（HTTPS）时，tunnel WebSocket 连接可能超时。
需要修改 `akernel_sdk` 本地代码，将 `ws://` 改为 `wss://`：

```bash
# 编辑文件
vi /usr/local/lib/python3.12/dist-packages/akernel_sdk/sandbox_api.py

# 修改以下行：
# 原: tunnel_ws_url = f"ws://{gateway}/{safe_id}/{tunnel_port}"
# 改: tunnel_ws_url = f"wss://{gateway}/{safe_id}/{tunnel_port}"
```
