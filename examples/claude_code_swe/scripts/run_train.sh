#!/usr/bin/env bash
# Training launch script for Claude Code SWE recipe.
#
# Uses GRPO + AgentFrameworkRolloutAdapter with reward computed in sandbox
# by Claude Code runner, then surfaced through reward worker compute_score.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-Coder-30B-A3B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/swe_agent/swe_bench_verified.parquet}"
VAL_DATA="${VAL_DATA:-$HOME/data/swe_agent/swe_bench_verified.parquet}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-131072}"
ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-$((PROMPT_LENGTH + RESPONSE_LENGTH + 1024))}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"

ENGINE="${ENGINE:-vllm}"
TP="${TP:-4}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
N="${N:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-32}"

MAX_TURNS="${MAX_TURNS:-100}"
COMPLETION_TIMEOUT="${COMPLETION_TIMEOUT:-7200}"

PROJECT_NAME="${PROJECT_NAME:-claude_code_swe}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-claude_code_$(date +%Y%m%d_%H%M)}"
VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"

export CLAUDE_CODE_IMAGE="${CLAUDE_CODE_IMAGE:-claude-code-tool:latest}"
export CLAUDE_CODE_SANDBOX_TYPE="${CLAUDE_CODE_SANDBOX_TYPE:-openyuanrong}"
export CLAUDE_CODE_TIMEOUT="${CLAUDE_CODE_TIMEOUT:-7200}"
export CLAUDE_CODE_MODEL="${CLAUDE_CODE_MODEL:-default}"
export CLAUDE_CODE_PERMISSION_MODE="${CLAUDE_CODE_PERMISSION_MODE:-bypassPermissions}"
export CLAUDE_CODE_CONDA_ENV="${CLAUDE_CODE_CONDA_ENV:-testbed}"
export CLAUDE_CODE_AUTO_COMPACT_WINDOW="${CLAUDE_CODE_AUTO_COMPACT_WINDOW:-60000}"
export CLAUDE_CODE_DISABLE_WEB_TOOLS="${CLAUDE_CODE_DISABLE_WEB_TOOLS:-1}"
export CLAUDE_CODE_DISABLE_SLASH_COMMANDS="${CLAUDE_CODE_DISABLE_SLASH_COMMANDS:-1}"
export SWE_AGENT_EVAL_TIMEOUT="${SWE_AGENT_EVAL_TIMEOUT:-600}"
export VERL_LOGGING_LEVEL

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

echo "=== Claude Code SWE Training ==="
echo "Model:       ${MODEL_PATH}"
echo "Train data:  ${TRAIN_DATA}"
echo "Val data:    ${VAL_DATA}"
echo "Engine:      ${ENGINE} (TP=${TP})"
echo "Batch size:  ${TRAIN_BATCH_SIZE}, N=${N}"
echo "Actor tok:   ${ACTOR_MAX_TOKEN_LEN_PER_GPU}"
echo "Epochs:      ${TOTAL_EPOCHS}"
echo "Sandbox:     ${CLAUDE_CODE_SANDBOX_TYPE}"
echo "Tool image:  ${CLAUDE_CODE_IMAGE}"
echo "================================"

python3 -m verl.trainer.main_ppo_sync \
    --config-name=claude_code_swe \
    --config-path="${REPO_ROOT}/examples/claude_code_swe/config" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_files="['${TRAIN_DATA}']" \
    data.val_files="['${VAL_DATA}']" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${PROMPT_LENGTH} \
    data.max_response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.max_model_len=$((PROMPT_LENGTH + RESPONSE_LENGTH + 1024)) \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
    actor_rollout_ref.rollout.custom.agent_framework.completion_timeout_seconds=${COMPLETION_TIMEOUT} \
    actor_rollout_ref.rollout.custom.agent_framework.max_concurrent_sessions=${MAX_CONCURRENT_SESSIONS} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.rollout.nnodes=${NNODES} \
    actor_rollout_ref.rollout.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    "$@"
