#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.5-9B}"
DATA_PATH="${DATA_PATH:-$HOME/data/swe_agent/swe_bench_verified.parquet}"

MAX_SAMPLES="${MAX_SAMPLES:--1}"
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-65536}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
N="${N:-1}"
ENGINE="${ENGINE:-vllm}"
TP="${TP:-4}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

export CLAUDE_CODE_IMAGE="${CLAUDE_CODE_IMAGE:-claude-code-tool:latest}"
export CLAUDE_CODE_MAX_TURNS="${CLAUDE_CODE_MAX_TURNS:-100}"
export CLAUDE_CODE_SANDBOX_TYPE="${CLAUDE_CODE_SANDBOX_TYPE:-openyuanrong}"
export CLAUDE_CODE_TIMEOUT="${CLAUDE_CODE_TIMEOUT:-7200}"
export SWE_AGENT_EVAL_TIMEOUT="${SWE_AGENT_EVAL_TIMEOUT:-600}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"

cd "$(dirname "$0")/../.."

python examples/claude_code_swe/parallel_infer.py \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --max-samples "${MAX_SAMPLES}" \
    --prompt-length "${PROMPT_LENGTH}" \
    --response-length "${RESPONSE_LENGTH}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --n "${N}" \
    --engine "${ENGINE}" \
    --tensor-parallel-size "${TP}" \
    --max-turns "${CLAUDE_CODE_MAX_TURNS}" \
    --n-gpus-per-node "${N_GPUS_PER_NODE}"
