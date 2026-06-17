#!/usr/bin/env bash
# Fully async Megatron training launcher for Claude Code SWE recipe.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-Coder-30B-A3B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/swe_agent/swe_bench_verified.parquet}"
VAL_DATA="${VAL_DATA:-$HOME/data/swe_agent/swe_bench_verified.parquet}"

NNODES_TRAIN="${NNODES_TRAIN:-1}"
NNODES_ROLLOUT="${NNODES_ROLLOUT:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"

PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-131072}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-0}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-100}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

N="${N:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-32}"
MAX_TURNS="${MAX_TURNS:-100}"
COMPLETION_TIMEOUT="${COMPLETION_TIMEOUT:-7200}"

TRAIN_TP="${TRAIN_TP:-4}"
TRAIN_CP="${TRAIN_CP:-1}"
TRAIN_PP="${TRAIN_PP:-1}"
TRAIN_VPP="${TRAIN_VPP:-null}"
TRAIN_EP="${TRAIN_EP:-1}"
TRAIN_ETP="${TRAIN_ETP:-1}"
GEN_TP="${GEN_TP:-4}"
GEN_DP="${GEN_DP:-1}"
GEN_EP="${GEN_EP:-1}"

STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-0.1}"
TRIGGER_PARAMETER_SYNC_STEP="${TRIGGER_PARAMETER_SYNC_STEP:-4}"
REQUIRE_BATCHES="${REQUIRE_BATCHES:-1}"
PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-True}"

PROJECT_NAME="${PROJECT_NAME:-claude_code_swe_async}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-claude_code_async_$(date +%Y%m%d_%H%M)}"
VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"

TOTAL_TOKENS=$((PROMPT_LENGTH + RESPONSE_LENGTH))
ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-$((TOTAL_TOKENS / TRAIN_CP))}"
LOGPROB_MAX_TOKEN_LEN_PER_GPU="${LOGPROB_MAX_TOKEN_LEN_PER_GPU:-$((TOTAL_TOKENS / TRAIN_CP))}"

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

echo "=== Claude Code SWE Async Megatron Training ==="
echo "Model:          ${MODEL_PATH}"
echo "Train data:     ${TRAIN_DATA}"
echo "Val data:       ${VAL_DATA}"
echo "Train nodes:    ${NNODES_TRAIN}"
echo "Rollout nodes:  ${NNODES_ROLLOUT}"
echo "GPUs/node:      ${NGPUS_PER_NODE}"
echo "Train parallel: TP=${TRAIN_TP} CP=${TRAIN_CP} PP=${TRAIN_PP} EP=${TRAIN_EP} ETP=${TRAIN_ETP}"
echo "Rollout para:   TP=${GEN_TP} DP=${GEN_DP} EP=${GEN_EP}"
echo "Rollout N:      ${N}"
echo "Rollout steps:  ${TOTAL_ROLLOUT_STEPS}"
echo "Sandbox:        ${CLAUDE_CODE_SANDBOX_TYPE}"
echo "Tool image:     ${CLAUDE_CODE_IMAGE}"
echo "==============================================="

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name=claude_code_swe_megatron_async \
    --config-path="${REPO_ROOT}/examples/claude_code_swe/config" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_files="['${TRAIN_DATA}']" \
    data.val_files="['${VAL_DATA}']" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.gen_batch_size=${GEN_BATCH_SIZE} \
    data.max_prompt_length=${PROMPT_LENGTH} \
    data.max_response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${TRAIN_VPP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${TRAIN_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${TRAIN_ETP} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${TRAIN_CP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${TRAIN_VPP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${TRAIN_EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${TRAIN_ETP} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.data_parallel_size=${GEN_DP} \
    actor_rollout_ref.rollout.expert_parallel_size=${GEN_EP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.max_model_len=$((PROMPT_LENGTH + RESPONSE_LENGTH + 1024)) \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
    actor_rollout_ref.rollout.custom.agent_framework.completion_timeout_seconds=${COMPLETION_TIMEOUT} \
    actor_rollout_ref.rollout.custom.agent_framework.max_concurrent_sessions=${MAX_CONCURRENT_SESSIONS} \
    trainer.nnodes=${NNODES_TRAIN} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.total_epochs=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    rollout.nnodes=${NNODES_ROLLOUT} \
    rollout.n_gpus_per_node=${NGPUS_PER_NODE} \
    rollout.n=${N} \
    rollout.total_rollout_steps=${TOTAL_ROLLOUT_STEPS} \
    async_training.staleness_threshold=${STALENESS_THRESHOLD} \
    async_training.trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP} \
    async_training.require_batches=${REQUIRE_BATCHES} \
    async_training.partial_rollout=${PARTIAL_ROLLOUT} \
    "$@"
