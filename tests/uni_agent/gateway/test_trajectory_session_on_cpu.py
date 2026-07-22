import asyncio
import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tests.uni_agent.support import FakeProcessor, FakeTokenizer, fake_vision_info_extractor
from uni_agent.gateway.session import (
    CapturedGeneration,
    CaptureErrorCode,
    CaptureReceipt,
    CaptureTransactionError,
    CaptureValidationError,
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
    routed_experts = [[[1, 2]]] * (len(expected_prompt_ids) + 2)

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
    assert len(receipt.routed_experts) == len(expected_prompt_ids) + 2
    assert receipt.routed_experts[0] == ((1, 2),)
    assert receipt.routing_metadata == {"model": "fixture-model"}

    routed_experts[0][0][0] = 99
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
    assert len(trajectory.routed_experts) == len(expected_prompt_ids) + 2
    assert trajectory.routed_experts[0] == [[1, 2]]
    trajectory.routed_experts[0][0][0] = 77
    assert receipt.routed_experts[0] == ((1, 2),)


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


@pytest.mark.asyncio
async def test_passive_capture_linear_continuation_accumulates_context_and_model_tokens():
    session = TrajectorySession(
        SessionHandle(session_id="passive-continuation"),
        MessageCodec(FakeTokenizer()),
        sampling_params={"logprobs": True},
    )
    first_messages = [{"role": "user", "content": "first"}]
    first_transaction = await session.capture(
        {"messages": first_messages, "tools": None, "sampling_params": {}}
    )
    first_receipt = await first_transaction.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "ONE"},
            prompt_ids=first_transaction.context_ids,
            completion_ids=(79, 78, 69),
            completion_logprobs=(-0.1, -0.2, -0.3),
            stop_reason="completed",
            routed_experts=[[[1]]] * (len(first_transaction.context_ids) + 3),
        )
    )
    continuation_messages = [
        *first_messages,
        {"role": "assistant", "content": "ONE"},
        {"role": "user", "content": "next"},
    ]

    continuation = await session.capture(
        {"messages": continuation_messages, "tools": None, "sampling_params": {}}
    )

    incremental_ids = tuple(ord(char) for char in "user:next\nassistant:")
    assert continuation.context_ids == (
        *first_receipt.prompt_ids,
        *first_receipt.response_ids,
        *incremental_ids,
    )
    second_receipt = await continuation.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "TWO"},
            prompt_ids=continuation.context_ids,
            completion_ids=(84, 87, 79),
            completion_logprobs=(-0.4, -0.5, -0.6),
            stop_reason="completed",
            routed_experts=[[[2]]] * (len(continuation.context_ids) + 3),
        )
    )

    assert second_receipt.chain_id == first_receipt.chain_id
    assert second_receipt.turn_id == 2
    assert second_receipt.prompt_ids == continuation.context_ids
    assert second_receipt.response_ids == (84, 87, 79)
    assert second_receipt.response_mask == (1, 1, 1)
    assert second_receipt.response_logprobs == (-0.4, -0.5, -0.6)
    assert len(second_receipt.routed_experts) == len(continuation.context_ids) + 3
    assert second_receipt.routed_experts[0] == ((2,),)

    [trajectory] = await session.finalize()
    assert trajectory.response_ids == [*first_receipt.response_ids, *incremental_ids, *second_receipt.response_ids]
    assert trajectory.response_mask == [1, 1, 1, *([0] * len(incremental_ids)), 1, 1, 1]
    assert trajectory.response_logprobs == [
        -0.1,
        -0.2,
        -0.3,
        *([0.0] * len(incremental_ids)),
        -0.4,
        -0.5,
        -0.6,
    ]
    assert len(trajectory.routed_experts) == len(continuation.context_ids) + 3
    assert trajectory.routed_experts[0] == [[2]]


@pytest.mark.asyncio
async def test_passive_capture_zero_token_turn_keeps_continuation_logprobs_aligned():
    session = TrajectorySession(
        SessionHandle(session_id="passive-zero-token-logprobs"),
        MessageCodec(FakeTokenizer()),
        sampling_params={"logprobs": True},
    )
    first_messages = [{"role": "user", "content": "first"}]
    first = await session.capture({"messages": first_messages, "tools": None, "sampling_params": {}})
    await first.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": ""},
            prompt_ids=first.context_ids,
            completion_ids=(),
            completion_logprobs=(),
            stop_reason="length",
        )
    )
    continuation_messages = [
        *first_messages,
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "next"},
    ]
    continuation = await session.capture(
        {"messages": continuation_messages, "tools": None, "sampling_params": {}}
    )
    await continuation.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "X"},
            prompt_ids=continuation.context_ids,
            completion_ids=(88,),
            completion_logprobs=(-0.1,),
            stop_reason="completed",
        )
    )

    [trajectory] = await session.finalize()
    assert trajectory.response_logprobs is not None
    assert len(trajectory.response_logprobs) == len(trajectory.response_ids)
    assert trajectory.response_logprobs[:-1] == [0.0] * (len(trajectory.response_ids) - 1)
    assert trajectory.response_logprobs[-1] == -0.1


@pytest.mark.asyncio
async def test_passive_capture_tool_continuation_reuses_chain_and_preserves_tools():
    session = TrajectorySession(
        SessionHandle(session_id="passive-tool-continuation"),
        MessageCodec(FakeTokenizer()),
    )
    tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    tool_call = {
        "id": "call_fixture",
        "type": "function",
        "function": {"name": "search", "arguments": '{"query":"weather"}'},
    }
    first_messages = [{"role": "user", "content": "weather"}]
    first = await session.capture({"messages": first_messages, "tools": tools, "sampling_params": {}})
    first_receipt = await first.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "", "tool_calls": [tool_call]},
            prompt_ids=first.context_ids,
            completion_ids=(84,),
            completion_logprobs=(-0.1,),
            stop_reason="completed",
        )
    )
    continuation_messages = [
        *first_messages,
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call_fixture", "content": "sunny"},
    ]

    continuation = await session.capture(
        {"messages": continuation_messages, "tools": tools, "sampling_params": {}}
    )
    second_receipt = await continuation.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "Sunny."},
            prompt_ids=continuation.context_ids,
            completion_ids=(83,),
            completion_logprobs=(-0.2,),
            stop_reason="completed",
        )
    )

    assert second_receipt.chain_id == first_receipt.chain_id
    assert second_receipt.turn_id == 2
    [trajectory] = await session.finalize()
    assert trajectory.response_ids[0] == 84
    assert trajectory.response_ids[-1] == 83
    assert 0 in trajectory.response_mask


@pytest.mark.asyncio
async def test_passive_capture_multimodal_continuation_accumulates_new_media():
    session = TrajectorySession(
        SessionHandle(session_id="passive-multimodal-continuation"),
        MessageCodec(
            FakeTokenizer(),
            processor=FakeProcessor(),
            vision_info_extractor=_video_metadata_extractor,
        ),
    )
    first_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "image://first"}},
                {"type": "text", "text": "inspect"},
            ],
        }
    ]
    first = await session.capture({"messages": first_messages, "tools": None, "sampling_params": {}})
    await first.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "seen"},
            prompt_ids=first.context_ids,
            completion_ids=(65,),
            completion_logprobs=(-0.1,),
            stop_reason="completed",
        )
    )
    continuation_messages = [
        *first_messages,
        {"role": "assistant", "content": "seen"},
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "video://second"}},
                {"type": "text", "text": "compare"},
            ],
        },
    ]

    continuation = await session.capture(
        {"messages": continuation_messages, "tools": None, "sampling_params": {}}
    )

    assert continuation.image_data == ("image://first",)
    assert continuation.video_data == (("video://second", {"url": "video://second"}),)
    await continuation.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "compared"},
            prompt_ids=continuation.context_ids,
            completion_ids=(66,),
            completion_logprobs=(-0.2,),
            stop_reason="completed",
        )
    )
    [trajectory] = await session.finalize()
    assert trajectory.multi_modal_data == {
        "images": ["image://first"],
        "videos": [("video://second", {"url": "video://second"})],
    }


@pytest.mark.asyncio
async def test_passive_capture_response_budget_clamps_then_closes_exhausted_chain():
    session = TrajectorySession(
        SessionHandle(session_id="passive-length-exhaustion"),
        MessageCodec(FakeTokenizer()),
        response_length=24,
    )
    first_messages = [{"role": "user", "content": "first"}]
    first = await session.capture(
        {
            "messages": first_messages,
            "tools": None,
            "sampling_params": {"max_tokens": 99},
        }
    )
    assert first.sampling_params["max_tokens"] == 24
    first_receipt = await first.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "ONE"},
            prompt_ids=first.context_ids,
            completion_ids=(79, 78, 69),
            completion_logprobs=(-0.1, -0.2, -0.3),
            stop_reason="completed",
        )
    )
    continuation_messages = [
        *first_messages,
        {"role": "assistant", "content": "ONE"},
        {"role": "user", "content": "next"},
    ]

    clamped = await session.capture(
        {"messages": continuation_messages, "tools": None, "sampling_params": {"max_tokens": 99}}
    )
    assert clamped.length_exhausted is False
    assert clamped.sampling_params["max_tokens"] == 1
    second_receipt = await clamped.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "X"},
            prompt_ids=clamped.context_ids,
            completion_ids=(88,),
            completion_logprobs=(-0.4,),
            stop_reason="length",
        )
    )
    exhausted_messages = [
        *continuation_messages,
        {"role": "assistant", "content": "X"},
        {"role": "user", "content": "again"},
    ]

    exhausted = await session.capture(
        {"messages": exhausted_messages, "tools": None, "sampling_params": {"max_tokens": 99}}
    )

    assert exhausted.length_exhausted is True
    assert exhausted.sampling_params == {}
    receipt = await exhausted.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "ignored"},
            prompt_ids=exhausted.context_ids,
            completion_ids=(),
            completion_logprobs=(),
            stop_reason="length",
        )
    )
    assert receipt.chain_id == first_receipt.chain_id
    assert receipt.turn_id == 3
    assert receipt.response_ids == ()
    assert receipt.response_mask == ()
    assert receipt.response_logprobs == ()
    assert receipt.finish_reason == "length"
    assert receipt.assistant_message == {"role": "assistant", "content": ""}

    [trajectory] = await session.finalize()
    assert trajectory.response_ids[-1] == second_receipt.response_ids[-1]
    assert trajectory.response_mask.count(1) == 4
    assert trajectory.extra_fields == {"materialization_reason": "max_response_length"}


@pytest.mark.asyncio
async def test_passive_capture_concurrent_siblings_reserve_distinct_chains():
    session = TrajectorySession(
        SessionHandle(session_id="passive-reserved-siblings"),
        MessageCodec(FakeTokenizer()),
    )
    prompt = [{"role": "user", "content": "same"}]
    seed_receipts = []
    for response_token in (65, 66, 67):
        transaction = await session.capture({"messages": prompt, "tools": None, "sampling_params": {}})
        seed_receipts.append(
            await transaction.commit(
                CapturedGeneration(
                    assistant_message={"role": "assistant", "content": chr(response_token)},
                    prompt_ids=transaction.context_ids,
                    completion_ids=(response_token,),
                    completion_logprobs=(-0.1,),
                    stop_reason="completed",
                )
            )
        )
    continuations = [
        [*prompt, {"role": "assistant", "content": chr(token)}, {"role": "user", "content": "next"}]
        for token in (67, 66, 65)
    ]
    transactions = [
        await session.capture({"messages": messages, "tools": None, "sampling_params": {}})
        for messages in continuations
    ]
    fallback = await session.capture(
        {"messages": continuations[0], "tools": None, "sampling_params": {}}
    )

    receipts = []
    for transaction, token in [
        (fallback, 90),
        (transactions[0], 88),
        (transactions[2], 86),
        (transactions[1], 87),
    ]:
        receipts.append(
            await transaction.commit(
                CapturedGeneration(
                    assistant_message={"role": "assistant", "content": chr(token)},
                    prompt_ids=transaction.context_ids,
                    completion_ids=(token,),
                    completion_logprobs=(-0.2,),
                    stop_reason="completed",
                )
            )
        )

    assert {receipt.chain_id for receipt in receipts} == {
        *(receipt.chain_id for receipt in seed_receipts),
        4,
    }
    assert [receipt.chain_id for receipt in receipts] == [4, 3, 1, 2]
    trajectories = await session.finalize()
    assert len(trajectories) == 4


@pytest.mark.asyncio
async def test_passive_capture_cleanup_releases_chain_for_every_exit_path():
    session = TrajectorySession(
        SessionHandle(session_id="passive-capture-cleanup"),
        MessageCodec(FakeTokenizer()),
    )
    first_messages = [{"role": "user", "content": "first"}]
    first = await session.capture({"messages": first_messages, "tools": None, "sampling_params": {}})
    first_receipt = await first.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "ONE"},
            prompt_ids=first.context_ids,
            completion_ids=(79,),
            completion_logprobs=(-0.1,),
            stop_reason="completed",
        )
    )
    continuation_messages = [
        *first_messages,
        {"role": "assistant", "content": "ONE"},
        {"role": "user", "content": "next"},
    ]
    continuation_request = {
        "messages": continuation_messages,
        "tools": None,
        "sampling_params": {},
    }

    rolled_back = await session.capture(continuation_request)
    await rolled_back.rollback()
    await rolled_back.rollback()

    with pytest.raises(RuntimeError, match="backend boom"):
        async with await session.capture(continuation_request):
            raise RuntimeError("backend boom")

    with pytest.raises(ValueError, match="caller boom"):
        async with await session.capture(continuation_request):
            raise ValueError("caller boom")

    entered = asyncio.Event()
    never = asyncio.Event()

    async def cancelled_rollout():
        async with await session.capture(continuation_request):
            entered.set()
            await never.wait()

    cancelled_task = asyncio.create_task(cancelled_rollout())
    await entered.wait()
    cancelled_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_task

    failed_commit = await session.capture(continuation_request)
    with pytest.raises(TypeError, match="JSON serializable"):
        await failed_commit.commit(
            CapturedGeneration(
                assistant_message={"role": "assistant", "content": object()},
                prompt_ids=failed_commit.context_ids,
                completion_ids=(70,),
                completion_logprobs=(-0.4,),
                stop_reason="completed",
            )
        )

    retry = await session.capture(continuation_request)
    retry_receipt = await retry.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "TWO"},
            prompt_ids=retry.context_ids,
            completion_ids=(84,),
            completion_logprobs=(-0.2,),
            stop_reason="completed",
        )
    )
    assert retry_receipt.chain_id != first_receipt.chain_id

    third_messages = [
        *continuation_messages,
        {"role": "assistant", "content": "TWO"},
        {"role": "user", "content": "again"},
    ]
    third = await session.capture({"messages": third_messages, "tools": None, "sampling_params": {}})
    third_receipt = await third.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "THREE"},
            prompt_ids=third.context_ids,
            completion_ids=(72,),
            completion_logprobs=(-0.3,),
            stop_reason="completed",
        )
    )
    assert third_receipt.chain_id == retry_receipt.chain_id
    [trajectory] = await session.finalize()
    assert trajectory.response_ids[-1] == 72


@pytest.mark.asyncio
async def test_passive_capture_repeated_cancellation_cannot_cancel_reservation_cleanup():
    class BlockingCodec(MessageCodec):
        def __init__(self):
            super().__init__(FakeTokenizer())
            self.block_next = False
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def extract_multi_modal_data(self, messages):
            if self.block_next:
                self.block_next = False
                self.entered.set()
                await self.release.wait()
            return await super().extract_multi_modal_data(messages)

    codec = BlockingCodec()
    session = TrajectorySession(
        SessionHandle(session_id="passive-double-cancel-cleanup"),
        codec,
    )
    first_messages = [{"role": "user", "content": "first"}]
    first = await session.capture({"messages": first_messages, "tools": None, "sampling_params": {}})
    first_receipt = await first.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "ONE"},
            prompt_ids=first.context_ids,
            completion_ids=(79,),
            completion_logprobs=(-0.1,),
            stop_reason="completed",
        )
    )
    continuation_messages = [
        *first_messages,
        {"role": "assistant", "content": "ONE"},
        {"role": "user", "content": "next"},
    ]
    continuation_request = {
        "messages": continuation_messages,
        "tools": None,
        "sampling_params": {},
    }
    reserved = await session.capture(continuation_request)

    codec.block_next = True
    lock_holder = asyncio.create_task(
        session.capture(
            {
                "messages": [{"role": "user", "content": "hold session lock"}],
                "tools": None,
                "sampling_params": {},
            }
        )
    )
    await codec.entered.wait()
    cleanup = asyncio.create_task(reserved.rollback())
    await asyncio.sleep(0)
    cleanup.cancel()
    await asyncio.sleep(0)
    cleanup.cancel()
    codec.release.set()

    with pytest.raises(asyncio.CancelledError):
        await cleanup
    holder_transaction = await lock_holder
    await holder_transaction.rollback()

    retry = await session.capture(continuation_request)
    retry_receipt = await retry.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "TWO"},
            prompt_ids=retry.context_ids,
            completion_ids=(84,),
            completion_logprobs=(-0.2,),
            stop_reason="completed",
        )
    )
    assert retry_receipt.chain_id == first_receipt.chain_id


@pytest.mark.asyncio
async def test_passive_capture_main_and_subagent_commit_to_reserved_chains_out_of_order():
    session = TrajectorySession(
        SessionHandle(session_id="passive-main-subagent"),
        MessageCodec(FakeTokenizer()),
    )
    main_messages = [
        {"role": "system", "content": "main"},
        {"role": "user", "content": "work"},
    ]
    sub_messages = [
        {"role": "system", "content": "subagent"},
        {"role": "user", "content": "research"},
    ]
    seed_receipts = []
    for messages, token in ((main_messages, 77), (sub_messages, 83)):
        transaction = await session.capture({"messages": messages, "tools": None, "sampling_params": {}})
        seed_receipts.append(
            await transaction.commit(
                CapturedGeneration(
                    assistant_message={"role": "assistant", "content": chr(token)},
                    prompt_ids=transaction.context_ids,
                    completion_ids=(token,),
                    completion_logprobs=(-0.1,),
                    stop_reason="completed",
                )
            )
        )
    main_continuation = [
        *main_messages,
        {"role": "assistant", "content": "M"},
        {"role": "user", "content": "continue"},
    ]
    sub_continuation = [
        *sub_messages,
        {"role": "assistant", "content": "S"},
        {"role": "user", "content": "continue"},
    ]
    main = await session.capture(
        {"messages": main_continuation, "tools": None, "sampling_params": {}}
    )
    subagent = await session.capture(
        {"messages": sub_continuation, "tools": None, "sampling_params": {}}
    )

    sub_receipt = await subagent.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "s"},
            prompt_ids=subagent.context_ids,
            completion_ids=(115,),
            completion_logprobs=(-0.2,),
            stop_reason="completed",
        )
    )
    main_receipt = await main.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "m"},
            prompt_ids=main.context_ids,
            completion_ids=(109,),
            completion_logprobs=(-0.3,),
            stop_reason="completed",
        )
    )

    assert main_receipt.chain_id == seed_receipts[0].chain_id
    assert sub_receipt.chain_id == seed_receipts[1].chain_id
    trajectories = await session.finalize()
    assert len(trajectories) == 2
    assert {trajectory.response_ids[-1] for trajectory in trajectories} == {109, 115}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("generation_fields", "error_code"),
    [
        ({"prompt_ids": None}, CaptureErrorCode.MISSING_PROMPT_IDS),
        ({"prompt_ids": (999,)}, CaptureErrorCode.PROMPT_MISMATCH),
        ({"completion_ids": None}, CaptureErrorCode.MISSING_COMPLETION_IDS),
        ({"completion_logprobs": None}, CaptureErrorCode.MISSING_LOGPROBS),
        ({"completion_logprobs": ()}, CaptureErrorCode.LOGPROB_LENGTH_MISMATCH),
        ({"completion_logprobs": ("bad",)}, CaptureErrorCode.MALFORMED_LOGPROBS),
        ({"completion_logprobs": 0.1}, CaptureErrorCode.MALFORMED_LOGPROBS),
        ({"completion_logprobs": (float("nan"),)}, CaptureErrorCode.MALFORMED_LOGPROBS),
        ({"completion_logprobs": (float("inf"),)}, CaptureErrorCode.MALFORMED_LOGPROBS),
        (
            {"completion_ids": (), "completion_logprobs": (), "stop_reason": "completed"},
            CaptureErrorCode.EMPTY_COMPLETION,
        ),
    ],
)
async def test_passive_capture_rejects_invalid_rollout_truth_without_materializing(
    generation_fields,
    error_code,
):
    session = TrajectorySession(
        SessionHandle(session_id=f"passive-validation-{error_code.value}"),
        MessageCodec(FakeTokenizer()),
    )
    transaction = await session.capture(
        {
            "messages": [{"role": "user", "content": "validate"}],
            "tools": None,
            "sampling_params": {},
        }
    )
    fields = {
        "assistant_message": {"role": "assistant", "content": "X"},
        "prompt_ids": transaction.context_ids,
        "completion_ids": (88,),
        "completion_logprobs": (-0.1,),
        "stop_reason": "completed",
    }
    fields.update(generation_fields)

    with pytest.raises(CaptureValidationError) as error:
        await transaction.commit(CapturedGeneration(**fields))

    assert error.value.code is error_code
    assert await session.finalize() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("routed_experts", "error_code"),
    [
        ([[1], [2]], CaptureErrorCode.ROUTING_SHAPE_MISMATCH),
        ([[[1]], [[2, 3]]], CaptureErrorCode.ROUTING_SHAPE_MISMATCH),
        ([[[1]]], CaptureErrorCode.ROUTING_LENGTH_MISMATCH),
        (np.zeros((1, 1, 1)), CaptureErrorCode.ROUTING_LENGTH_MISMATCH),
    ],
)
async def test_passive_capture_rejects_misaligned_routed_experts(routed_experts, error_code):
    session = TrajectorySession(
        SessionHandle(session_id=f"passive-routing-{error_code.value}"),
        MessageCodec(FakeTokenizer()),
    )
    transaction = await session.capture(
        {
            "messages": [{"role": "user", "content": "route"}],
            "tools": None,
            "sampling_params": {},
        }
    )

    with pytest.raises(CaptureValidationError) as error:
        await transaction.commit(
            CapturedGeneration(
                assistant_message={"role": "assistant", "content": "X"},
                prompt_ids=transaction.context_ids,
                completion_ids=(88,),
                completion_logprobs=(-0.1,),
                stop_reason="completed",
                routed_experts=routed_experts,
            )
        )

    assert error.value.code is error_code
    assert await session.finalize() == []


@pytest.mark.asyncio
async def test_passive_capture_accepts_aligned_ndarray_routed_experts():
    session = TrajectorySession(
        SessionHandle(session_id="passive-routing-ndarray"),
        MessageCodec(FakeTokenizer()),
    )
    transaction = await session.capture(
        {
            "messages": [{"role": "user", "content": "route"}],
            "tools": None,
            "sampling_params": {},
        }
    )
    routed_experts = np.zeros((len(transaction.context_ids) + 1, 2, 1), dtype=np.int64)

    receipt = await transaction.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "X"},
            prompt_ids=transaction.context_ids,
            completion_ids=(88,),
            completion_logprobs=(-0.1,),
            stop_reason="completed",
            routed_experts=routed_experts,
        )
    )

    assert receipt.routed_experts.shape == routed_experts.shape


@pytest.mark.asyncio
async def test_passive_capture_validation_failure_drops_chain_and_allows_independent_recovery():
    session = TrajectorySession(
        SessionHandle(session_id="passive-validation-isolation"),
        MessageCodec(FakeTokenizer()),
    )
    first_messages = [{"role": "user", "content": "first"}]
    first = await session.capture({"messages": first_messages, "tools": None, "sampling_params": {}})
    first_receipt = await first.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "ONE"},
            prompt_ids=first.context_ids,
            completion_ids=(79,),
            completion_logprobs=(-0.1,),
            stop_reason="completed",
        )
    )
    continuation_messages = [
        *first_messages,
        {"role": "assistant", "content": "ONE"},
        {"role": "user", "content": "next"},
    ]
    continuation_request = {
        "messages": continuation_messages,
        "tools": None,
        "sampling_params": {},
    }
    invalid = await session.capture(continuation_request)

    with pytest.raises(CaptureValidationError) as error:
        await invalid.commit(
            CapturedGeneration(
                assistant_message={"role": "assistant", "content": "BAD"},
                prompt_ids=(999,),
                completion_ids=(66,),
                completion_logprobs=(-0.2,),
                stop_reason="completed",
            )
        )
    assert error.value.code is CaptureErrorCode.PROMPT_MISMATCH
    assert session.snapshot_state()["num_active_chains"] == 0

    recovery = await session.capture(continuation_request)
    recovery_receipt = await recovery.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "GOOD"},
            prompt_ids=recovery.context_ids,
            completion_ids=(71,),
            completion_logprobs=(-0.3,),
            stop_reason="completed",
        )
    )

    assert recovery_receipt.chain_id != first_receipt.chain_id
    [trajectory] = await session.finalize()
    assert trajectory.response_ids == [71]


@pytest.mark.asyncio
async def test_passive_capture_commit_failure_drops_selected_chain():
    session = TrajectorySession(
        SessionHandle(session_id="passive-commit-failure-isolation"),
        MessageCodec(FakeTokenizer()),
    )
    first_messages = [{"role": "user", "content": "first"}]
    first = await session.capture({"messages": first_messages, "tools": None, "sampling_params": {}})
    await first.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "ONE"},
            prompt_ids=first.context_ids,
            completion_ids=(79,),
            completion_logprobs=(-0.1,),
            stop_reason="completed",
        )
    )
    continuation_messages = [
        *first_messages,
        {"role": "assistant", "content": "ONE"},
        {"role": "user", "content": "next"},
    ]
    failed = await session.capture(
        {"messages": continuation_messages, "tools": None, "sampling_params": {}}
    )

    with pytest.raises(TypeError, match="JSON serializable"):
        await failed.commit(
            CapturedGeneration(
                assistant_message={"role": "assistant", "content": object()},
                prompt_ids=failed.context_ids,
                completion_ids=(66,),
                completion_logprobs=(-0.2,),
                stop_reason="completed",
            )
        )

    assert session.snapshot_state()["num_active_chains"] == 0
    assert await session.finalize() == []


@pytest.mark.asyncio
async def test_passive_capture_transaction_and_session_lifecycle_errors_are_classified():
    session = TrajectorySession(
        SessionHandle(session_id="passive-classified-lifecycle"),
        MessageCodec(FakeTokenizer()),
    )
    request = {
        "messages": [{"role": "user", "content": "first"}],
        "tools": None,
        "sampling_params": {},
    }
    committed = await session.capture(request)
    generation = CapturedGeneration(
        assistant_message={"role": "assistant", "content": "ONE"},
        prompt_ids=committed.context_ids,
        completion_ids=(79,),
        completion_logprobs=(-0.1,),
        stop_reason="completed",
    )
    await committed.commit(generation)

    with pytest.raises(CaptureTransactionError) as duplicate:
        await committed.commit(generation)
    assert duplicate.value.code is CaptureErrorCode.DUPLICATE_COMMIT

    continuation_request = {
        "messages": [
            *request["messages"],
            {"role": "assistant", "content": "ONE"},
            {"role": "user", "content": "next"},
        ],
        "tools": None,
        "sampling_params": {},
    }
    rolled_back = await session.capture(continuation_request)
    await rolled_back.rollback()
    with pytest.raises(CaptureTransactionError) as closed:
        await rolled_back.commit(
            CapturedGeneration(
                assistant_message={"role": "assistant", "content": "TWO"},
                prompt_ids=rolled_back.context_ids,
                completion_ids=(84,),
                completion_logprobs=(-0.2,),
                stop_reason="completed",
            )
        )
    assert closed.value.code is CaptureErrorCode.TRANSACTION_CLOSED
    with pytest.raises(CaptureTransactionError) as closed_context:
        async with rolled_back:
            pass
    assert closed_context.value.code is CaptureErrorCode.TRANSACTION_CLOSED

    assert await session.finalize() == []
    with pytest.raises(SessionLifecycleError) as phase_error:
        await session.capture(request)
    assert phase_error.value.code is CaptureErrorCode.INVALID_SESSION_PHASE

    commit_after_finalize_session = TrajectorySession(
        SessionHandle(session_id="passive-commit-after-finalize"),
        MessageCodec(FakeTokenizer()),
    )
    seed = await commit_after_finalize_session.capture(request)
    await seed.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "ONE"},
            prompt_ids=seed.context_ids,
            completion_ids=(79,),
            completion_logprobs=(-0.1,),
            stop_reason="completed",
        )
    )
    pending = await commit_after_finalize_session.capture(continuation_request)
    [stable_trajectory] = await commit_after_finalize_session.finalize()
    assert stable_trajectory.response_ids == [79]
    with pytest.raises(SessionLifecycleError) as commit_phase_error:
        await pending.commit(
            CapturedGeneration(
                assistant_message={"role": "assistant", "content": "LATE"},
                prompt_ids=pending.context_ids,
                completion_ids=(76,),
                completion_logprobs=(-0.3,),
                stop_reason="completed",
            )
        )
    assert commit_phase_error.value.code is CaptureErrorCode.INVALID_SESSION_PHASE
