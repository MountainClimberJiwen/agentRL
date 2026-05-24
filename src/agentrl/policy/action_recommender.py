"""
ActionRecommender — Recommend next actions based on historical trajectories.

Learns from past sessions to predict:
  1. What first action to take for a new task
  2. What next action to take after a given action
  3. What corrections to avoid repeating
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentrl.models import ActionStep, TaskTrajectory
from agentrl.policy.network import SoftmaxPolicy


class ActionRecommender:
    """Recommends actions using transition statistics + learned policy."""

    def __init__(self, policy: SoftmaxPolicy | None = None) -> None:
        self.policy = policy or SoftmaxPolicy()

        # Transition counts: current_action -> next_action -> count
        self.transition_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Action rewards: action -> list of rewards
        self.action_rewards: dict[str, list[float]] = defaultdict(list)

        # First-action map: intent -> {action, success_rate, count}
        self.first_action_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"action": "", "successes": 0, "total": 0})

        # Correction patterns: (wrong_action, correction_type, context) -> fix_info
        self.correction_fixes: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn_from_trajectory(self, traj: TaskTrajectory) -> None:
        """Ingest a single trajectory and update all statistics."""
        # 1. Learn state transitions
        for i in range(len(traj.steps) - 1):
            current = traj.steps[i].action_type
            next_a = traj.steps[i + 1].action_type
            self.transition_counts[current][next_a] += 1

        # 2. Learn first action for intent
        intent = self._detect_intent(traj.goal)
        if traj.steps:
            first = traj.steps[0].action_type
            stats = self.first_action_stats[intent]
            if stats["total"] == 0:
                stats["action"] = first
            stats["total"] += 1
            if traj.final_outcome in ("approved", "success"):
                stats["successes"] += 1

        # 3. Learn correction patterns
        for idx in traj.correction_points:
            if idx < len(traj.steps):
                wrong_step = traj.steps[idx]
                self.correction_fixes.append({
                    "wrong_action": wrong_step.action_type,
                    "wrong_target": wrong_step.target,
                    "correction_type": wrong_step.correction_type,
                    "context": traj.goal,
                    "intent": intent,
                })

        # 4. Record action rewards
        reward = self._outcome_to_reward(traj.final_outcome)
        for step in traj.steps:
            self.action_rewards[step.action_type].append(reward)

        # 5. Update policy network
        self._update_policy(traj, intent)

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend_first_action(self, context: str) -> dict[str, Any]:
        """Recommend the first action for a new task."""
        intent = self._detect_intent(context)

        # Lookup learned first-action stats
        stats = self.first_action_stats.get(intent)
        if stats and stats["total"] > 0:
            success_rate = stats["successes"] / stats["total"]
            return {
                "recommended_action": stats["action"],
                "confidence": min(0.95, success_rate),
                "reason": f"Learned from {stats['total']} past '{intent}' tasks",
                "alternatives": [],
                "warning": None,
            }

        # Fallback to policy network
        state = f"{intent}:start:0"
        available = ["read_file", "terminal", "browser", "search", "llm_response"]
        best = self.policy.get_best(state, available)
        return {
            "recommended_action": best,
            "confidence": 0.5,
            "reason": f"Policy fallback for '{intent}'",
            "alternatives": [],
            "warning": None,
        }

    def recommend_next_action(self, context: str, current_action: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Recommend the next action given current state."""
        intent = self._detect_intent(context)
        step_idx = len(history) if history else 0
        state = f"{intent}:{current_action}:{step_idx}"

        # 1. Transition statistics
        transitions = dict(self.transition_counts.get(current_action, {}))
        total_trans = sum(transitions.values()) if transitions else 0

        # 2. Policy network
        available = list(transitions.keys()) if transitions else ["read_file", "terminal", "browser", "search", "llm_response", "llm_response"]
        best_policy = self.policy.get_best(state, available)

        # 3. Merge: prefer high-frequency transition, but warn if policy disagrees
        if transitions:
            best_trans = max(transitions, key=transitions.get)
            confidence = transitions[best_trans] / total_trans
            warning = None
            if best_trans != best_policy and confidence < 0.7:
                warning = f"Policy suggests '{best_policy}' but only {confidence:.0%} transition confidence"
            recommended = best_trans if confidence > 0.5 else best_policy
        else:
            recommended = best_policy
            confidence = 0.3
            warning = "No historical transitions; using policy fallback"

        # 4. Check correction warnings
        corr_warning = self._check_correction_warning(current_action, intent, context)
        if corr_warning:
            warning = corr_warning

        alternatives = sorted(transitions.items(), key=lambda x: -x[1])[:3]

        return {
            "recommended_action": recommended,
            "confidence": confidence,
            "reason": f"Based on {total_trans} transitions from '{current_action}'",
            "alternative_actions": alternatives,
            "warning": warning,
        }

    def get_correction_warning(self, action: str, target: str, context: str) -> str | None:
        """Check if an action+target combination has been corrected before."""
        intent = self._detect_intent(context)
        for fix in self.correction_fixes:
            if fix["wrong_action"] == action and fix["intent"] == intent:
                if target and fix["wrong_target"] and target in fix["wrong_target"]:
                    return f"Previously corrected: {fix['correction_type']} in '{context[:60]}...'"
                if not target or not fix["wrong_target"]:
                    return f"Previously corrected ({fix['correction_type']}) in similar '{intent}' tasks"
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "ActionRecommender":
        """Load recommender from JSON file."""
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p) as f:
            data = json.load(f)
        rec = cls()
        if "policy" in data:
            rec.policy = SoftmaxPolicy.load(str(p))  # policy saved inline or same dir
        rec.transition_counts = defaultdict(lambda: defaultdict(int))
        for k, v in data.get("transition_counts", {}).items():
            rec.transition_counts[k] = defaultdict(int, v)
        rec.first_action_stats = defaultdict(lambda: {"action": "", "successes": 0, "total": 0})
        rec.first_action_stats.update(data.get("first_action_stats", {}))
        rec.correction_fixes = data.get("correction_fixes", [])
        rec.action_rewards = defaultdict(list)
        rec.action_rewards.update({k: v for k, v in data.get("action_rewards", {}).items()})
        return rec

    def save(self, path: str) -> None:
        """Save recommender state to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Save policy alongside
        policy_path = p.with_suffix(".policy.json")
        self.policy.save(str(policy_path))

        data = {
            "transition_counts": {k: dict(v) for k, v in self.transition_counts.items()},
            "first_action_stats": dict(self.first_action_stats),
            "correction_fixes": self.correction_fixes,
            "action_rewards": {k: v for k, v in self.action_rewards.items()},
            "policy_path": str(policy_path),
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_policy(self, traj: TaskTrajectory, intent: str) -> None:
        """Update policy network with trajectory reward."""
        reward = self._outcome_to_reward(traj.final_outcome)
        for i, step in enumerate(traj.steps):
            state = f"{intent}:{step.action_type}:{i}"
            is_corrected = i in traj.correction_points
            step_reward = reward - 0.5 if is_corrected else reward
            self.policy.update(state, step.action_type, step_reward)

    def _check_correction_warning(self, current_action: str, intent: str, context: str) -> str | None:
        """Check if current action has been corrected in similar contexts."""
        for fix in self.correction_fixes:
            if fix["wrong_action"] == current_action and fix["intent"] == intent:
                return f"History: '{current_action}' corrected ({fix['correction_type']}) in '{context[:60]}...'"
        return None

    @staticmethod
    def _outcome_to_reward(outcome: str) -> float:
        return {"approved": 1.0, "success": 1.0, "corrected": 0.3, "failed": -1.0, "unknown": 0.0}.get(outcome, 0.0)

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
