"""
AgentRLHermesHook — Session lifecycle hook for real-time learning.

Attach this to Hermes session end to:
  1. Extract trajectory from session messages
  2. Mine corrections
  3. Update policy via OnlinePolicyUpdater
  4. Save updated recommender + policy

Usage:
    hook = AgentRLHermesHook(data_dir="/opt/agentrl/data")
    hook.on_session_end(session_id="abc", messages=[...])
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentrl.extractors.trajectory import TrajectoryBuilder
from agentrl.models import TaskTrajectory
from agentrl.policy.action_recommender import ActionRecommender
from agentrl.policy.online_updater import OnlinePolicyUpdater
from agentrl.llm.kimi_client import KimiClient

logger = logging.getLogger(__name__)


class AgentRLHermesHook:
    """Lifecycle hook that learns from every Hermes session."""

    def __init__(self, data_dir: str = "/opt/agentrl/data", use_kimi_judge: bool = True) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.recommender_path = self.data_dir / "agentrl_recommender.json"
        self.policy_path = self.data_dir / "agentrl_policy.json"
        self.trajectory_db = self.data_dir / "trajectories.jsonl"
        self.use_kimi_judge = use_kimi_judge

        # Load or initialize
        self.recommender = ActionRecommender.load(str(self.recommender_path))
        self.updater = OnlinePolicyUpdater(self.recommender.policy)
        self.builder = TrajectoryBuilder()
        self.kimi = KimiClient()

    def on_session_end(self, session_id: str, messages: list[dict[str, Any]], backend: str = "hermes") -> dict[str, Any]:
        """
        Main entrypoint. Call this when a Hermes session finishes.

        Args:
            session_id: Hermes session identifier
            messages: Full message list from the session (JSONL-style dicts)
            backend: Backend name tag (default "hermes")

        Returns:
            Diagnostics dict with learning stats.
        """
        logger.info(f"[agentRL] Processing session {session_id} ({len(messages)} messages)")

        # 1. Build trajectory
        traj = self.builder.build(session_id=session_id, backend=backend, messages=messages)

        # 2. LLM Judge: resolve unknown outcomes
        agent_steps = [s for s in traj.steps if s.action_type != "user_input"]
        if self.use_kimi_judge and traj.final_outcome == "unknown" and len(agent_steps) >= 3:
            try:
                judged, confidence = self.kimi.judge_outcome(traj.goal, messages)
                if judged != "unknown":
                    logger.info(f"[agentRL] LLM judged outcome: {judged} (conf={confidence:.0%})")
                    traj.final_outcome = judged
            except Exception as e:
                logger.debug(f"[agentRL] LLM judge failed: {e}")

        # 3. Store raw trajectory for offline batch training later
        self._store_trajectory(traj)

        # 4. Skip if too short or no signal
        if len(traj.steps) < 2:
            return {"status": "skipped", "reason": "too_few_steps", "session_id": session_id}

        # 5. Learn into recommender (transitions, first-actions, corrections)
        self.recommender.learn_from_trajectory(traj)

        # 6. Online policy update (REINFORCE)
        diagnostics = self.updater.update_from_trajectory(traj)

        # 7. Persist
        self.recommender.save(str(self.recommender_path))

        logger.info(
            f"[agentRL] Learned: reward={diagnostics['trajectory_reward']:.2f} "
            f"corrections={traj.num_corrections} steps={len(traj.steps)} outcome={traj.final_outcome}"
        )

        return {
            "status": "learned",
            "session_id": session_id,
            "goal": traj.goal[:80],
            "final_outcome": traj.final_outcome,
            "num_steps": len(traj.steps),
            "num_corrections": traj.num_corrections,
            "has_early_mistake": traj.has_early_mistake,
            **diagnostics,
        }

    def get_recommendation(self, context: str, current_action: str = "start", history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """
        Ask agentRL for a recommendation mid-session.
        Can be called by a Hermes tool or memory provider.
        """
        if current_action == "start":
            return self.recommender.recommend_first_action(context)
        return self.recommender.recommend_next_action(context, current_action, history or [])

    def get_first_action(self, context: str) -> dict[str, Any]:
        """Convenience wrapper for first-action recommendation."""
        return self.recommender.recommend_first_action(context)

    def _store_trajectory(self, traj: TaskTrajectory) -> None:
        """Append trajectory summary to JSONL database."""
        record = {
            "trajectory_id": traj.trajectory_id,
            "session_id": traj.session_id,
            "backend": traj.backend,
            "goal": traj.goal,
            "final_outcome": traj.final_outcome,
            "num_steps": len(traj.steps),
            "num_corrections": traj.num_corrections,
            "has_early_mistake": traj.has_early_mistake,
            "step_types": [s.action_type for s in traj.steps],
            "correction_types": [traj.steps[i].correction_type for i in traj.correction_points],
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.trajectory_db, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
