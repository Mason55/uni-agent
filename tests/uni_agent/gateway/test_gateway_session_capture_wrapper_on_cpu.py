import pytest
from fastapi import HTTPException

from tests.uni_agent.support import FakeTokenizer
from uni_agent.gateway.session import (
    CapturedGeneration,
    GatewaySession,
    MessageCodec,
    SessionHandle,
    TrajectorySession,
)
from verl.workers.rollout.replica import TokenOutput


class _DeterministicBackend:
    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        assert sampling_params["logprobs"] is True
        completion_ids = [79, 75]
        return TokenOutput(
            token_ids=completion_ids,
            log_probs=[-0.1, -0.2],
            routed_experts=[[[index % 2]] for index in range(len(prompt_ids) + len(completion_ids))],
            stop_reason="completed",
        )


class _SequencedParityBackend:
    def __init__(self, steps):
        self.steps = list(steps)

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        assert sampling_params["logprobs"] is True
        text, expert_id = self.steps.pop(0)
        completion_ids = [ord(char) for char in text]
        return TokenOutput(
            token_ids=completion_ids,
            log_probs=[-0.1] * len(completion_ids),
            routed_experts=[[[expert_id]]] * (len(prompt_ids) + len(completion_ids)),
            stop_reason="completed",
        )


async def _commit_passive_turn(session, request, text, expert_id):
    transaction = await session.capture(request)
    completion_ids = tuple(ord(char) for char in text)
    routed_experts = tuple(((expert_id,),) for _ in range(len(transaction.context_ids) + len(completion_ids)))
    return await transaction.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": text},
            prompt_ids=transaction.context_ids,
            completion_ids=completion_ids,
            completion_logprobs=(-0.1,) * len(completion_ids),
            stop_reason="completed",
            routed_experts=routed_experts,
        )
    )


@pytest.mark.asyncio
async def test_native_wrapper_matches_passive_capture_receipt_and_trajectory():
    request = {
        "messages": [{"role": "user", "content": "Say OK"}],
        "tools": None,
        "sampling_params": {"temperature": 0.2},
    }
    native = GatewaySession(
        SessionHandle(session_id="native-parity"),
        MessageCodec(FakeTokenizer()),
    )
    passive = TrajectorySession(
        SessionHandle(session_id="passive-parity"),
        MessageCodec(FakeTokenizer()),
    )

    native_outcome = await native.run_generation(request, _DeterministicBackend())
    passive_transaction = await passive.capture(request)
    completion_ids = (79, 75)
    routed_experts = tuple(
        ((index % 2,),) for index in range(len(passive_transaction.context_ids) + len(completion_ids))
    )
    passive_receipt = await passive_transaction.commit(
        CapturedGeneration(
            assistant_message={"role": "assistant", "content": "OK"},
            prompt_ids=passive_transaction.context_ids,
            completion_ids=completion_ids,
            completion_logprobs=(-0.1, -0.2),
            stop_reason="completed",
            routed_experts=routed_experts,
        )
    )

    assert native_outcome.assistant_msg == {"role": "assistant", "content": "OK"}
    assert native_outcome.finish_reason == "stop"
    assert native_outcome.prompt_tokens == len(passive_transaction.context_ids)
    assert native_outcome.completion_tokens == 2
    assert native_outcome.capture_receipt == passive_receipt
    assert await native.finalize() == await passive.finalize()


@pytest.mark.asyncio
async def test_native_wrapper_matches_passive_continuation_and_replaces_full_context_routing():
    first_request = {
        "messages": [{"role": "user", "content": "first"}],
        "tools": None,
        "sampling_params": {},
    }
    continuation_request = {
        "messages": [
            *first_request["messages"],
            {"role": "assistant", "content": "ONE"},
            {"role": "user", "content": "next"},
        ],
        "tools": None,
        "sampling_params": {},
    }
    native = GatewaySession(
        SessionHandle(session_id="native-continuation-parity"),
        MessageCodec(FakeTokenizer()),
    )
    passive = TrajectorySession(
        SessionHandle(session_id="passive-continuation-parity"),
        MessageCodec(FakeTokenizer()),
    )
    backend = _SequencedParityBackend([("ONE", 1), ("TWO", 2)])

    first_native = await native.run_generation(first_request, backend)
    first_passive = await _commit_passive_turn(passive, first_request, "ONE", 1)
    second_native = await native.run_generation(continuation_request, backend)
    second_passive = await _commit_passive_turn(passive, continuation_request, "TWO", 2)

    assert first_native.capture_receipt == first_passive
    assert second_native.capture_receipt == second_passive
    native_trajectories = await native.finalize()
    passive_trajectories = await passive.finalize()
    assert native_trajectories == passive_trajectories
    assert native_trajectories[0].routed_experts[0] == [[2]]
    assert len(native_trajectories[0].routed_experts) == len(native_trajectories[0].prompt_ids) + len(
        native_trajectories[0].response_ids
    )


@pytest.mark.asyncio
async def test_native_wrapper_rejects_zero_token_abort_without_fabricating_rollout_truth():
    class AbortedBackend:
        async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
            assert sampling_params["logprobs"] is True
            return TokenOutput(token_ids=[], log_probs=None, stop_reason="aborted")

    session = GatewaySession(
        SessionHandle(session_id="native-abort"),
        MessageCodec(FakeTokenizer()),
    )
    request = {
        "messages": [{"role": "user", "content": "abort"}],
        "tools": None,
        "sampling_params": {},
    }

    with pytest.raises(HTTPException, match="upstream completion log-probabilities are required") as error:
        await session.run_generation(request, AbortedBackend())

    assert error.value.status_code == 500
    assert await session.finalize() == []
