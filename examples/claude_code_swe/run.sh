#!/bin/bash
export TUNNEL_SSL_VERIFY=0

cd /data1/lmy/uni-agent && \
  OPENYUANRONG_SERVER_ADDRESS="124.70.166.142:443" \
  OPENYUANRONG_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE4MTU4NTMwNTAsInJvbGUiOiJkZXZlbG9wZXIiLCJzdWIiOiJkZWZhdWx0In0.NGY4ZjkzZWZhYmE4YzkxOGIwNTdkN2VmZTQ5MTdiZWQ2MjhlMTMwYzA0OTU3NjRlMWNmNDNjZDUzYTMxNjliYw" \
  DEPLOYMENT=openyuanrong \
  PYTHONPATH=/data1/lmy/uni-agent:$PYTHONPATH \
  CLAUDE_CODE_IMAGE="${CLAUDE_CODE_IMAGE:-swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest}" \
  CLAUDE_CODE_SANDBOX_TYPE="${CLAUDE_CODE_SANDBOX_TYPE:-openyuanrong}" \
  MODEL_PATH="${MODEL_PATH:-/data1/models/Qwen/Qwen3.5-9B}" \
  DATA_PATH="${DATA_PATH:-/home/datasets/swe_bench_verified_openyuanrong.parquet}" \
  MAX_SAMPLES="${MAX_SAMPLES:-1}" \
  TP="${TP:-4}" \
  N="${N:-1}" \
  N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}" \
  PROMPT_LENGTH="${PROMPT_LENGTH:-65536}" \
  RESPONSE_LENGTH="${RESPONSE_LENGTH:-4096}" \
  CLAUDE_CODE_MAX_TURNS="${CLAUDE_CODE_MAX_TURNS:-30}" \
  GATEWAY_AGENT_LOG_PATH="${GATEWAY_AGENT_LOG_PATH:-/tmp/claude_code_gateway_agent.jsonl}" \
  DEBUG_MODE=1 \
  bash examples/claude_code_swe/run_infer.sh
