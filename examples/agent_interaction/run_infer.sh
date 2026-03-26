ray job submit --no-wait \
    --runtime-env examples/agent_interaction/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/agent_interaction/parallel_infer.py \
    --data-path ~/data/swe_agent/swe_bench_verified.parquet \
    --model-path ~/models/Qwen3-Coder-30B-A3B-Instruct \
    --agent-config-path examples/agent_interaction/agent_config.yaml \
    --nnodes 4 \
    --n-gpus-per-node 8 \
    --n 4 \
