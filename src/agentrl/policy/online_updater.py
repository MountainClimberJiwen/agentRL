"""
OnlinePolicyUpdater — Incremental policy updates from a single session.

Uses REINFORCE with baseline to update the SoftmaxPolicy after each
user interaction.  Can run in real-time at the end of a session.
"""

from __future__ import annotations

from agentrl.models import TaskTrajectory
from agentrl.policy.network import SoftmaxPolicy
from agentrl.policy.nn_policy import MLPPolicy, MiniTransformerPolicy, StateEncoder


class OnlinePolicyUpdater:
    """Update policy network from a single trajectory."""

    def __init__(self, policy: SoftmaxPolicy | MLPPolicy | MiniTransformerPolicy, learning_rate: float = 0.1, baseline_decay: float = 0.9) -> None:
        self.policy = policy
        self.lr = learning_rate
        self.baseline_decay = baseline_decay
        self.baseline = 0.0
        self.state_encoder = StateEncoder()
        self._action_to_idx = {
            "read_file": 0, "terminal": 1, "browser": 2, "search": 3,
            "llm_response": 4, "execute_code": 5, "user_input": 6,
        }

    def update_from_trajectory(self, traj: TaskTrajectory) -> dict[str, float]:
        """
        Perform one policy-gradient update using a full trajectory.

        Returns diagnostics:
            {
                "trajectory_reward": float,
                "advantage": float,
                "baseline_before": float,
                "baseline_after": float,
                "num_steps_updated": int,
                "num_corrected_steps": int,
            }
        """
        reward = self._compute_trajectory_reward(traj)
        advantage = reward - self.baseline
        baseline_before = self.baseline
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward

        num_updated = 0
        num_corrected = 0
        intent = self._detect_intent(traj.goal)

        for i, step in enumerate(traj.steps):
            is_corrected = i in traj.correction_points
            step_reward = reward - 0.5 if is_corrected else reward

            # Scale update by advantage magnitude (REINFORCE)
            scaled_lr = self.lr * (1.0 + abs(advantage))

            if isinstance(self.policy, (MLPPolicy, MiniTransformerPolicy)):
                state_str = f"{intent}:{step.action_type}:{i}"
                state_vec = self.state_encoder.encode(state_str)
                action_idx = self._action_to_idx.get(step.action_type, 0)
                self.policy.update(state_vec, action_idx, step_reward, lr=scaled_lr)
            else:
                state = f"{intent}:{step.action_type}:{i}"
                self.policy.update(state, step.action_type, step_reward, lr=scaled_lr)

            num_updated += 1
            if is_corrected:
                num_corrected += 1

        return {
            "trajectory_reward": reward,
            "advantage": advantage,
            "baseline_before": baseline_before,
            "baseline_after": self.baseline,
            "num_steps_updated": num_updated,
            "num_corrected_steps": num_corrected,
        }

    def _compute_trajectory_reward(self, traj: TaskTrajectory) -> float:
        """Composite reward: outcome + correction penalty + step efficiency."""
        base = {"approved": 1.0, "success": 1.0, "corrected": 0.3, "failed": -1.0, "unknown": 0.0}.get(traj.final_outcome, 0.0)

        # Correction penalty
        correction_penalty = len(traj.correction_points) * -0.2

        # Step efficiency (shorter successful paths are better)
        # Normalize: 10 steps = 0 penalty, more steps = penalty
        step_penalty = max(0, (len(traj.steps) - 10)) * -0.02

        return base + correction_penalty + step_penalty

    @staticmethod
    def _detect_intent(goal: str) -> str:
        """Simple keyword-based intent detection."""
        text = goal.lower()
        keywords = {
            "coding": ["code", "implement", "write", "modify", "fix", "bug", "refactor", "function", "class", "实现", "写", "修改", "修复"],
            "doc": ["doc", "readme", "documentation", "explain", "说明", "文档"],
            "deploy": ["deploy", "server", "host", "上线", "部署", "服务器", "nginx", "docker"],
            "config": ["config", "setup", "configure", "settings", "配置", "设置"],
            "debug": ["debug", "troubleshoot", "error", "exception", "调试", "排查", "错误"],
            "test": ["test", "unit test", "spec", "测试"],
            "git": ["git", "commit", "branch", "merge", "pr", "pull request"],
        }
        scores = {intent: sum(1 for k in kws if k in text) for intent, kws in keywords.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"
