import pytest

from uni_agent.gateway.session import (
    SessionHandle,
    SessionLifecycleError,
    TrajectorySession,
)


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
