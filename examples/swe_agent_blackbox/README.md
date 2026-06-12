# Mini-SWE-Agent In-Sandbox Execution

## Overview

Mini-swe-agent runs inside an OpenYuanRong remote sandbox as a sidecar tool image mount. The agent executes commands via `LocalEnvironment` (local bash) inside the sandbox, and calls the LLM through a gateway URL piped via stdin. The external runner creates the sandbox, triggers agent execution, and evaluates the reward.

The tool image uses [python-build-standalone](https://github.com/astral-sh/python-build-standalone) to build a self-contained Python environment, independent of the sandbox container's Python version, with a minimal `FROM scratch` final image.

## Architecture

```
[Rollouter Host: mini_swe_agent_runner]
  │
  ├── YRSandbox.create(image, sidecar_image)
  │     └── OpenYuanRong: Sandbox(mounts=[Mount(target="/opt/mini-swe-agent", ...)])
  │
  ├── sandbox.run("echo <b64_config> | base64 -d | /opt/.../python run_agent.py")
  │     └── [Inside Sandbox]
  │           /opt/mini-swe-agent/bin/python  ← standalone Python, isolated from sandbox
  │           stdin ← task config JSON (task, gateway_url, agent)
  │           LocalEnvironment + LitellmModel(gateway_url) → DefaultAgent
  │           stdout → result JSON (exit_status, submission, model_stats)
  │
  ├── _parse_agent_result(stdout)
  ├── SandboxEnvForReward(sandbox) → evaluate_in_env()
  └── session_runtime.complete_session(reward_info)
```

## Prerequisites

1. **OpenYuanRong** — `OPENYUANRONG_SERVER_ADDRESS` and `OPENYUANRONG_TOKEN` must be set
2. **Tool image** — Build and push to a remote registry (see below)

## 1. Build Tool Image

The tool image contains a standalone Python environment with mini-swe-agent, mounted as a sidecar into the OpenYuanRong sandbox. Build once, push to a registry, and reuse across runs.

```bash
bash examples/swe_agent_blackbox/build_tool.sh --registry <your-registry>
```

Options:
- `--pip-index` — Use a custom pip mirror for faster downloads

The output is a minimal `FROM scratch` image containing only Python + mini-swe-agent + litellm.
The runner references it via the `MINI_SWE_AGENT_IMAGE` environment variable.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TOOL_IMAGE` | `mini-swe-agent-tool` | Image name |
| `TOOL_TAG` | `latest` | Image tag |

## 2. Dataset

Use `data_preprocess` to generate training/inference data in parquet format.

```bash
DEPLOYMENT=openyuanrong python examples/data_preprocess/swe_bench_verified.py --local-save-dir ~/data/swe_agent
```

Output: `~/data/swe_agent/swe_bench_verified_openyuanrong.parquet`

| Field | Description |
|-------|-------------|
| `prompt` | Task description (chat message list with system + user) |
| `agent_name` | Agent type identifier (e.g. `swe_agent`) |
| `extra_info.tools_kwargs.env.deployment` | Sandbox config (contains `image` for the container) |
| `extra_info.tools_kwargs.reward` | Reward evaluation config (FAIL_TO_PASS, PASS_TO_PASS, etc.) |

## 3. Inference

### Environment Variables

```bash
export OPENYUANRONG_SERVER_ADDRESS="6.2.179.37:8888"
export OPENYUANRONG_TOKEN="<your-token>"
```

### Run

```bash
RUNNER=mini_swe \
OPENYUANRONG_SERVER_ADDRESS="6.2.179.37:8888" \
OPENYUANRONG_TOKEN="<token>" \
bash examples/swe_agent_blackbox/scripts/run_infer.sh
```

## 4. Training (Fully Async)

```bash
OPENYUANRONG_SERVER_ADDRESS="6.2.179.37:8888" \
OPENYUANRONG_TOKEN="<token>" \
MODEL_PATH=~/models/Qwen3.5-9B \
bash examples/swe_agent_blackbox/scripts/run_train_megatron_async.sh
```

## 5. Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SWE_AGENT_MAX_TURNS` | `100` | Max agent steps |
| `MINI_SWE_AGENT_IMAGE` | `swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest` | Sidecar tool image |
| `DEBUG_MODE` | (unset) | Set to 1 to enable debug logging |
