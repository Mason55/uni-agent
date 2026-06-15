# Claude Code SWE-Verified Example

This directory builds a Claude Code sidecar tool image and runs Claude Code on
SWE-Verified style task sandboxes.

The runtime model matches the mini-swe-agent sidecar pattern:

- `image=<SWE task image>` starts the task sandbox.
- `Mount(target="/opt/claude-code", image_url=<claude-code-tool image>)` mounts this tool image into the task sandbox.
- Claude Code runs inside the task sandbox from `/opt/claude-code/bin/claude`.
- Claude Code talks to the Uni-Agent gateway through the Anthropic Messages API:
  `/sessions/{session_id}/v1/messages`.

The shared gateway now supports both protocols:

- OpenAI route: `/sessions/{session_id}/v1/chat/completions`
- Anthropic route: `/sessions/{session_id}/v1/messages`

The SWE blackbox example is not modified for Claude Code. This directory owns
the Claude-specific runner, sandbox mount target, and inference entrypoint.

Build locally:

```bash
bash build_tool.sh
```

Build with an npm mirror:

```bash
bash build_tool.sh --npm-registry https://registry.npmmirror.com
```

Build and push for remote sandbox use:

```bash
bash build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

Pin Claude Code:

```bash
bash build_tool.sh --claude-code-version 2.1.118
```

Use after mount:

```bash
export PATH=/opt/claude-code/bin:$PATH
export DISABLE_AUTOUPDATER=1
export IS_SANDBOX=1
export ANTHROPIC_BASE_URL=http://127.0.0.1:38197/sessions/<session_id>
export ANTHROPIC_API_KEY=not-needed
claude -p "fix the issue" --model default --max-turns 3 --permission-mode bypassPermissions
```

Authentication is not baked into the image. Pass credentials at runtime, for example `ANTHROPIC_API_KEY`, Bedrock, Vertex, or Claude Code account auth state, depending on your deployment.

The runner rewrites SWE-bench's long task prompt into a Claude Code prompt:

- edit source files only;
- run the relevant tests from reward metadata;
- do not use a `submit` tool;
- do not create extra edge-case test files after relevant tests pass;
- print a one-line summary and exit.

Claude Code is also launched with auto-compaction and disabled web/slash paths by default:

```bash
CLAUDE_CODE_AUTO_COMPACT_WINDOW=60000
CLAUDE_CODE_DISABLE_WEB_TOOLS=1
CLAUDE_CODE_DISABLE_SLASH_COMMANDS=1
```

Run one OpenYuanRong sample:

```bash
bash examples/claude_code_swe/run.sh
```

Or call the inference script directly:

```bash
CLAUDE_CODE_IMAGE=claude-code-tool:latest \
CLAUDE_CODE_SANDBOX_TYPE=openyuanrong \
MODEL_PATH=/data1/models/Qwen/Qwen3.5-4B \
DATA_PATH=/home/datasets/swe_bench_verified_openyuanrong.parquet \
MAX_SAMPLES=1 \
TP=2 \
N_GPUS_PER_NODE=2 \
bash examples/claude_code_swe/run_infer.sh
```

Key files:

- `claude_code_runner.py`: creates sandbox, runs Claude Code, evaluates reward.
- `sandbox.py`: local Docker and OpenYuanRong sandbox helpers with `/opt/claude-code` sidecar target.
- `framework.py`: small framework subclass that passes Anthropic-style `base_url` to the runner.
- `parallel_infer.py`: minimal inference entrypoint for Claude Code.
