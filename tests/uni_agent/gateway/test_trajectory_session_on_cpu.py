import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from tests.uni_agent.support import FakeProcessor, FakeTokenizer, fake_vision_info_extractor
from uni_agent.gateway.session import (
    CapturedGeneration,
    CaptureReceipt,
    MessageCodec,
    SessionHandle,
    SessionLifecycleError,
    TrajectorySession,
)


async def _video_metadata_extractor(messages, image_patch_size, config=None):
    images, videos = await fake_vision_info_extractor(
        messages,
        image_patch_size=image_patch_size,
        config=config,
    )
    if videos is not None:
        videos = [(video, {"url": video}) for video in videos]
    return images, videos


def test_trajectory_session_constructs_without_codec_backend_or_transport():
    session = TrajectorySession(SessionHandle(session_id="domain-only"))

    assert session.snapshot_state() == {
        "session_id": "domain-only",
        "phase": "ACTIVE",
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "num_trajectories": 0,
        "has_active_trajectory": False,
        "num_active_chains": 0,
        "active_chain_ids": [],
        "active_chain_tip_hashes": {},
    }


def test_trajectory_session_public_import_does_not_load_runtime_modules():
    script = """
import sys
from uni_agent.gateway.session import TrajectorySession

forbidden_modules = {
    "fastapi",
    "ray",
    "transfer_queue",
    "uni_agent.gateway.gateway",
    "uni_agent.gateway.manager",
    "uni_agent.framework.framework",
    "verl.utils.transferqueue_utils",
}
forbidden_prefixes = ("verl.trainer",)
loaded = sorted(
    name
    for name in sys.modules
    if name in forbidden_modules or name.startswith(forbidden_prefixes)
)
if loaded:
    raise SystemExit(f"runtime modules loaded: {loaded}")
assert TrajectorySession.__name__ == "TrajectorySession"
"""

    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.asyncio
async def test_trajectory_session_public_lifecycle_is_transport_independent():
    session = TrajectorySession(SessionHandle(session_id="lifecycle"))

    await session.set_reward_info({"score": 1.0})
    assert await session.finalize() == []
    assert session.snapshot_state()["phase"] == "FINALIZED"

    with pytest.raises(SessionLifecycleError, match="finalized"):
        await session.set_reward_info({"score": 0.0})
    with pytest.raises(SessionLifecycleError, match="finalized"):
        await session.abort()


@pytest.mark.asyncio
async def test_trajectory_session_abort_is_idempotent_and_prevents_finalize():
    session = TrajectorySession(SessionHandle(session_id="aborted"))

    await session.abort()
    await session.abort()

    assert session.snapshot_state()["phase"] == "ABORTED"
    with pytest.raises(SessionLifecycleError, match="aborted"):
        await session.finalize()


@pytest.mark.asyncio
async def test_single_turn_passive_capture_returns_receipt_and_finalized_trajectory():
    session = TrajectorySession(
        SessionHandle(session_id="passive-single-turn"),
        MessageCodec(FakeTokenizer()),
        response_length=8,
        sampling_params={"temperature": 0.2},
    )
    request = {
        "messages": [{"role": "user", "content": "Say OK"}],
        "tools": None,
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "custom": {"adapter": "base"},
        },
    }
    expected_prompt_ids = tuple(ord(char) for char in "user:Say OK\nassistant:")
    routed_experts = {"layers": [[[1, 2], [3, 4]]]}

    transaction = await session.capture(request)

    assert transaction.context_ids == expected_prompt_ids
    assert transaction.sampling_params == {
        "temperature": 0.7,
        "top_p": 0.9,
        "custom": {"adapter": "base"},
        "max_tokens": 8,
    }
    assert transaction.image_data is None
    assert transaction.video_data is None

    receipt = await transaction.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "OK"},
            prompt_ids=expected_prompt_ids,
            completion_ids=(79, 75),
            completion_logprobs=(-0.1, -0.2),
            stop_reason="completed",
            routed_experts=routed_experts,
            routing_metadata={"model": "fixture-model"},
        )
    )

    assert isinstance(receipt, CaptureReceipt)
    assert receipt.prompt_ids == expected_prompt_ids
    assert receipt.response_ids == (79, 75)
    assert receipt.response_mask == (1, 1)
    assert receipt.response_logprobs == (-0.1, -0.2)
    assert receipt.chain_id == 1
    assert receipt.turn_id == 1
    assert receipt.usage.prompt_tokens == len(expected_prompt_ids)
    assert receipt.usage.completion_tokens == 2
    assert receipt.assistant_message == {"role": "assistant", "content": "OK"}
    assert receipt.finish_reason == "stop"
    assert receipt.routed_experts == {"layers": (((1, 2), (3, 4)),)}
    assert receipt.routing_metadata == {"model": "fixture-model"}

    routed_experts["layers"][0][0][0] = 99
    request["messages"][0]["content"] = "caller-mutated"
    request["sampling_params"]["custom"]["adapter"] = "caller-mutated"
    exposed_sampling_params = transaction.sampling_params
    exposed_sampling_params["custom"]["adapter"] = "transaction-mutated"
    assert transaction.sampling_params["custom"] == {"adapter": "base"}

    [trajectory] = await session.finalize()

    assert trajectory.prompt_ids == list(receipt.prompt_ids)
    assert trajectory.response_ids == list(receipt.response_ids)
    assert trajectory.response_mask == list(receipt.response_mask)
    assert trajectory.response_logprobs == list(receipt.response_logprobs)
    assert trajectory.routed_experts == {"layers": [[[1, 2], [3, 4]]]}
    trajectory.routed_experts["layers"][0][0][0] = 77
    assert receipt.routed_experts == {"layers": (((1, 2), (3, 4)),)}


@pytest.mark.asyncio
async def test_single_turn_capture_prepares_multimodal_inputs():
    session = TrajectorySession(
        SessionHandle(session_id="passive-multimodal"),
        MessageCodec(
            FakeTokenizer(),
            processor=FakeProcessor(),
            vision_info_extractor=_video_metadata_extractor,
        ),
    )
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "image://fixture"}},
                    {"type": "video_url", "video_url": {"url": "video://fixture"}},
                    {"type": "text", "text": "describe"},
                ],
            }
        ],
        "tools": None,
        "sampling_params": {},
    }

    transaction = await session.capture(request)

    assert transaction.image_data == ("image://fixture",)
    assert transaction.video_data == (("video://fixture", {"url": "video://fixture"}),)
    assert transaction.context_ids[-2:] == (FakeProcessor.image_token_id, FakeProcessor.video_token_id)


def test_capture_records_are_immutable():
    assistant_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "search", "arguments": {"query": "original"}}}],
    }
    generation = CapturedGeneration(
        assistant_message=assistant_message,
        prompt_ids=[1, 2],
        completion_ids=[3],
        completion_logprobs=[-0.1],
        stop_reason="completed",
        routed_experts={"layers": [[1, 2]]},
        routing_metadata={"route": {"adapter": "base"}},
    )

    assistant_message["tool_calls"][0]["function"]["arguments"]["query"] = "caller-mutated"
    assert generation.prompt_ids == (1, 2)
    assert generation.assistant_message["tool_calls"][0]["function"]["arguments"]["query"] == "original"
    with pytest.raises(FrozenInstanceError):
        generation.stop_reason = "length"
    with pytest.raises(TypeError):
        generation.assistant_message["tool_calls"][0]["function"]["arguments"]["query"] = "record-mutated"
    with pytest.raises(TypeError):
        generation.routing_metadata["route"]["adapter"] = "changed"
    with pytest.raises(TypeError):
        generation.routed_experts["layers"][0][0] = 9
