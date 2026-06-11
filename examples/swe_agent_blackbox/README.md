# Mini-SWE-Agent In-Sandbox Execution — 使用指南

## 概述

mini-swe-agent 以 sidecar 工具镜像的形式挂载到沙箱内部运行。Agent 在沙箱内通过
`LocalEnvironment`（本地 bash）执行命令，LLM 调用走 stdin 传入的 gateway URL。
外部 runner 创建沙箱、触发 agent 执行、评估 reward。

工具镜像使用 [python-build-standalone](https://github.com/astral-sh/python-build-standalone) 构建独立 Python 环境，不依赖沙箱容器内的 Python 版本，通过 `FROM scratch` 保持镜像最小化。

**支持的沙箱类型：**

| 类型 | 说明 |
|------|------|
| 本地 Docker (`"local"`) | `docker exec` + `docker cp` 从工具镜像加载 sidecar |
| OpenYuanRong (`"openyuanrong"`) | `akernel_sdk.Mount` + `sandbox.commands.run()` |

两种沙箱类型都直接依赖 `mini-swe-agent-tool` 工具镜像，无需预先提取到 host 目录。

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
2. **mini-swe-agent tool image** — 需预先构建（见下文）

## 1. 构建 Tool Image

使用 `build_tool.sh` 一次性构建：

```bash
# 默认 PyPI 源
bash examples/swe_agent_blackbox/build_tool.sh

# 使用国内 PyPI 镜像加速
bash examples/swe_agent_blackbox/build_tool.sh --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/

# 推送到远程仓库
bash examples/swe_agent_blackbox/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong

# 组合使用
bash examples/swe_agent_blackbox/build_tool.sh \
    --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

脚本会：
1. `docker build` 构建工具镜像 `mini-swe-agent-tool:latest`
   - 基于 `python-build-standalone`（独立 Python 3.12，兼容不同 glibc 版本）
   - `FROM scratch` 最终镜像，仅包含 Python + mini-swe-agent + litellm
2. 如果指定 `--registry`，自动 tag + push 到远程仓库

工具镜像内的 Python 与沙箱容器的 Python 完全隔离，互不影响。

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TOOL_IMAGE` | `mini-swe-agent-tool` | 镜像名 |
| `TOOL_TAG` | `latest` | 镜像 tag |
| `PIP_INDEX_URL` | (空，使用 PyPI) | pip 镜像源（也可通过 `--pip-index` 传入） |

## 2. 推理（本地 Docker 沙箱）

### 使用 run_infer.sh

```bash
cd /home/dyp/recipe/uni-agent

RUNNER=mini_swe \
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
MINI_SWE_AGENT_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest \
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
| `SWE_AGENT_EVAL_TIMEOUT` | `600` | Reward 评估超时（秒） |
| `MINI_SWE_AGENT_IMAGE` | `swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest` | sidecar 工具镜像 |
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
