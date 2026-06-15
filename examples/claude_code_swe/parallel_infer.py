"""Parallel inference entry for Claude Code on SWE-Verified style data."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

import numpy as np
import ray
from tensordict import TensorDict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.utils import hf_tokenizer
from verl.utils.transferqueue_utils import tq as _tq_mock
from verl.workers.rollout.llm_server import LLMServerManager

from uni_agent.trainer.gateway.runtime import GatewayServingRuntime

from examples.claude_code_swe.claude_code_runner import claude_code_runner
from examples.claude_code_swe.framework import ClaudeCodeSWEFramework
from examples.swe_agent_blackbox.parallel_infer import _init_hydra_config, load_swe_dataset

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=os.getenv("VERL_LOGGING_LEVEL", "INFO"),
    force=True,
)
logger = logging.getLogger(__name__)


class _MockReplayBuffer:
    def add(self, partition_id, items):
        pass


def run_inference(
    *,
    model_path: str,
    data_path: str,
    prompt_length: int = 4096,
    response_length: int = 65536,
    temperature: float = 0.8,
    top_p: float = 0.9,
    n: int = 1,
    max_samples: int = -1,
    engine: str = "vllm",
    nnodes: int = 1,
    n_gpus_per_node: int = 8,
    tensor_parallel_size: int = 4,
    gateway_count: int = 1,
    completion_timeout: float = 600.0,
    tool_parser: str | None = None,
) -> dict[str, Any]:
    if not ray.is_initialized():
        ray.init()

    config = _init_hydra_config(
        model_path=model_path,
        engine=engine,
        prompt_length=prompt_length,
        response_length=response_length,
        temperature=temperature,
        top_p=top_p,
        n=n,
        nnodes=nnodes,
        n_gpus_per_node=n_gpus_per_node,
        tensor_parallel_size=tensor_parallel_size,
    )

    samples = load_swe_dataset(data_path, max_samples=max_samples)
    if not samples:
        raise ValueError("No samples to process")

    logger.info("Initializing LLM server manager...")
    llm_server_manager = LLMServerManager.create(config=config)
    llm_client = llm_server_manager.get_client()
    gateway_actor_kwargs = {
        "tokenizer": hf_tokenizer(os.path.expanduser(model_path)),
        "base_sampling_params": {"temperature": temperature, "top_p": top_p, "max_tokens": response_length},
    }
    if tool_parser:
        gateway_actor_kwargs["tool_parser_name"] = tool_parser

    gateway_runtime = GatewayServingRuntime(
        llm_client=llm_client,
        gateway_count=gateway_count,
        gateway_actor_kwargs=gateway_actor_kwargs,
    )

    from verl.experimental.reward_loop.reward_loop import RewardLoopWorker

    reward_worker = ray.remote(RewardLoopWorker).remote(config, None)
    framework = ClaudeCodeSWEFramework(
        session_runtime=gateway_runtime,
        agent_runner=claude_code_runner,
        replay_buffer=_MockReplayBuffer(),
        rollout_config={"n": n, "val_kwargs": {"n": n}},
        completion_timeout=completion_timeout,
        wait_for_completion_after_agent_run=True,
        max_concurrent_sessions=2,
        reward_loop_worker_handles=[reward_worker],
    )

    tools_kwargs_list = []
    for sample in samples:
        tk = (sample.get("extra_info") or {}).get("tools_kwargs", {})
        tk["model_path"] = os.path.expanduser(model_path)
        tools_kwargs_list.append(tk)

    from verl.utils import tensordict_utils as _tu

    raw_prompts = [sample["prompt"] for sample in samples]
    uids = [str(uuid4()) for _ in samples]
    td = TensorDict({"uid": uids, "global_steps": [0] * len(samples)}, batch_size=[len(samples)])
    _tu.assign_non_tensor_stack(td, "raw_prompt", raw_prompts)
    _tu.assign_non_tensor_stack(td, "tools_kwargs", tools_kwargs_list)
    _tu.assign_non_tensor_stack(td, "data_source", [sample["data_source"] for sample in samples])
    _tu.assign_non_tensor_stack(td, "reward_model", [sample["reward_model"] for sample in samples])

    batch = DataProto(batch=td, meta_info={}).repeat(n)
    batch_padded, _ = pad_dataproto_to_divisor(batch, gateway_count)

    tq_store: dict[str, Any] = {}

    async def _dummy_kv_put(key, partition_id=None, tag=None, **kwargs):
        tq_store[key] = tag

    async def _dummy_kv_batch_put(keys=None, fields=None, tags=None, partition_id=None, **kwargs):
        for i, key in enumerate(keys):
            tq_store[key] = {"fields": fields, "tag": tags[i] if tags else None}

    _tq_mock.async_kv_put = _dummy_kv_put
    _tq_mock.async_kv_batch_put = _dummy_kv_batch_put

    async def _generate():
        return await framework.generate_sequences(batch_padded.batch)

    try:
        stats = asyncio.run(_generate())
    except RuntimeError as exc:
        logger.warning("generate_sequences failed: %s", exc)
        stats = {}

    uid_to_sample_idx = {uid: i for i, uid in enumerate(uids)}
    per_sample_scores = [0.0] * len(samples)
    sample_trajectory_counts = [0] * len(samples)
    for key, value in tq_store.items():
        if not isinstance(value, dict) or "fields" not in value:
            continue
        fields = value["fields"]
        rm_scores = fields.get("rm_scores")
        if rm_scores is None:
            continue
        uid = key.rsplit("_", 2)[0]
        sample_idx = uid_to_sample_idx.get(uid)
        if sample_idx is None:
            continue
        per_sample_scores[sample_idx] += float(rm_scores.float()[-1, -1].item())
        sample_trajectory_counts[sample_idx] += 1

    for i in range(len(samples)):
        if sample_trajectory_counts[i] > 0:
            per_sample_scores[i] /= sample_trajectory_counts[i]

    asyncio.run(gateway_runtime.shutdown())
    return {
        "stats": stats,
        "mean_score": float(np.mean(per_sample_scores)) if per_sample_scores else 0.0,
        "per_sample_scores": per_sample_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Claude Code SWE-Verified inference")
    parser.add_argument("--data-path", type=str, default="~/data/swe_agent/swe_bench_verified.parquet")
    parser.add_argument("--model-path", "--model", type=str, default="~/models/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--response-length", type=int, default=65536)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--n-gpus-per-node", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=4)
    parser.add_argument("--tool-parser", type=str, default="qwen3_coder")
    args = parser.parse_args()

    os.environ["CLAUDE_CODE_MAX_TURNS"] = str(args.max_turns)
    run_inference(
        model_path=args.model_path,
        data_path=args.data_path,
        prompt_length=args.prompt_length,
        response_length=args.response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        max_samples=args.max_samples,
        engine=args.engine,
        nnodes=args.nnodes,
        n_gpus_per_node=args.n_gpus_per_node,
        tensor_parallel_size=args.tensor_parallel_size,
        tool_parser=args.tool_parser,
    )


if __name__ == "__main__":
    main()
