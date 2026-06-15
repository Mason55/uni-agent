"""Framework glue for Claude Code SWE example."""

from __future__ import annotations

from dataclasses import replace

from examples.swe_agent_blackbox.framework import SWEAgentFramework


class ClaudeCodeSWEFramework(SWEAgentFramework):
    async def _run_session(
        self,
        *,
        prompts,
        raw_prompt,
        sample_index: int,
        session_id: str | None = None,
        runner_kwargs: dict | None = None,
    ):
        """Run one session with Anthropic-compatible base_url for Claude Code."""
        session_id = session_id or None
        sample_fields = self._extract_sample_fields(prompts=prompts, sample_index=sample_index)
        if session_id is None:
            from uuid import uuid4

            session_id = f"session-{sample_index}-0-{uuid4().hex}"

        session = await self.session_runtime.create_session(session_id)
        session = replace(session, base_url=session.base_url.removesuffix("/v1"))

        try:
            await self.agent_runner(
                raw_prompt=raw_prompt,
                session=session,
                sample_index=sample_index,
                session_runtime=self.session_runtime,
                **(runner_kwargs or {}),
            )
            if self.wait_for_completion_after_agent_run:
                await self.session_runtime.wait_for_completion(session_id, timeout=self.completion_timeout)
            session_trajectories = await self.session_runtime.finalize_session(session_id)
        except Exception:
            await self.session_runtime.abort_session(session_id)
            raise

        if not self.reward_loop_worker_handles or not session_trajectories:
            return session_trajectories, sample_fields

        annotations = await self._score_trajectories(session_trajectories, sample_fields)
        scored_trajectories = []
        for traj, (score, extra) in zip(session_trajectories, annotations, strict=True):
            from dataclasses import replace as dataclass_replace

            scored_trajectories.append(
                dataclass_replace(
                    traj,
                    reward_score=score,
                    extra_fields={**traj.extra_fields, "reward_extra_info": extra},
                )
            )
        return scored_trajectories, sample_fields
