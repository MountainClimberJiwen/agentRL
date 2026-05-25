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
from agentrl.policy.nn_policy import MLPPolicy, MiniTransformerPolicy, StateEncoder
from agentrl.llm.kimi_client import KimiClient
from agentrl.sync.mem0_sync import Mem0PolicySync

# Action mapping for neural policies
_ACTION_TO_IDX = {
    "read_file": 0, "terminal": 1, "browser": 2, "search": 3,
    "llm_response": 4, "execute_code": 5, "user_input": 6,
}


class ActionRecommender:
    """Recommends actions + targets using transition statistics + learned policy + LLM fallback."""

    def __init__(self, policy: SoftmaxPolicy | MLPPolicy | MiniTransformerPolicy | None = None) -> None:
        self.policy = policy or SoftmaxPolicy()
        self.state_encoder = StateEncoder()
        self._action_to_idx = _ACTION_TO_IDX

        # Transition counts: current_action -> next_action -> count
        self.transition_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Action rewards: action -> list of rewards
        self.action_rewards: dict[str, list[float]] = defaultdict(list)

        # First-action map: intent -> {action, success_rate, count}
        self.first_action_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"action": "", "successes": 0, "total": 0})

        # Correction patterns: (wrong_action, correction_type, context) -> fix_info
        self.correction_fixes: list[dict[str, Any]] = []

        # Target learning: action_type -> target -> count
        self.target_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # LLM client for target inference
        self.kimi = KimiClient()

        # Mem0 sync for multi-agent sharing
        self.mem0_sync = Mem0PolicySync()
        self._agent_steps_total = 0

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn_from_trajectory(self, traj: TaskTrajectory) -> None:
        """Ingest a single trajectory and update all statistics."""
        # Filter out user_input steps for action learning
        agent_steps = [s for s in traj.steps if s.action_type != "user_input"]
        if not agent_steps:
            return

        self._agent_steps_total += len(agent_steps)
        intent = self._detect_intent(traj.goal)

        # 1. Learn state transitions (agent actions only)
        for i in range(len(agent_steps) - 1):
            current = agent_steps[i].action_type
            next_a = agent_steps[i + 1].action_type
            self.transition_counts[current][next_a] += 1

        # 2. Learn targets
        for step in agent_steps:
            if step.target:
                self.target_counts[step.action_type][step.target] += 1

        # 3. Learn first action for intent (first agent action)
        first = agent_steps[0].action_type
        first_target = agent_steps[0].target
        stats = self.first_action_stats[intent]
        if stats["total"] == 0:
            stats["action"] = first
            stats["target"] = first_target
        stats["total"] += 1
        if traj.final_outcome in ("approved", "success"):
            stats["successes"] += 1

        # 4. Learn correction patterns (map correction index to agent step)
        for idx in traj.correction_points:
            if idx < len(traj.steps):
                wrong_step = traj.steps[idx]
                # Skip if correction was on user_input itself; map to nearest preceding agent step
                if wrong_step.action_type == "user_input":
                    # Find nearest preceding agent step
                    preceding = [s for s in traj.steps[:idx] if s.action_type != "user_input"]
                    if preceding:
                        wrong_step = preceding[-1]
                    else:
                        continue
                self.correction_fixes.append({
                    "wrong_action": wrong_step.action_type,
                    "wrong_target": wrong_step.target,
                    "correction_type": wrong_step.correction_type,
                    "context": traj.goal,
                    "intent": intent,
                })

        # 5. Record action rewards (agent actions only)
        reward = self._outcome_to_reward(traj.final_outcome)
        for step in agent_steps:
            self.action_rewards[step.action_type].append(reward)

        # 6. Update policy network (agent actions only)
        self._update_policy(traj, intent, agent_steps)

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend_first_action(self, context: str, use_llm: bool = True) -> dict[str, Any]:
        """Recommend the first action + target for a new task."""
        intent = self._detect_intent(context)

        # Lookup learned first-action stats
        stats = self.first_action_stats.get(intent)
        if stats and stats["total"] > 0 and stats["action"]:
            success_rate = stats["successes"] / stats["total"]
            target = stats.get("target", "")
            if not target and use_llm:
                target = self.kimi.infer_target(context, "start", stats["action"])
            return {
                "recommended_action": stats["action"],
                "recommended_target": target,
                "confidence": min(0.95, success_rate),
                "reason": f"Learned from {stats['total']} past '{intent}' tasks",
                "alternatives": [],
                "warning": None,
            }

        # Fallback to policy network
        state_str = f"{intent}:start:0"
        available = ["read_file", "terminal", "browser", "search", "llm_response"]

        if isinstance(self.policy, (MLPPolicy, MiniTransformerPolicy)):
            state_vec = self.state_encoder.encode(state_str, self._build_state_stats(intent, "start"))
            best = self.policy.get_best(state_vec, available, self._action_to_idx)
        else:
            best = self.policy.get_best(state_str, available)
        target = ""
        if use_llm:
            target = self.kimi.infer_target(context, "start", best)
        return {
            "recommended_action": best,
            "recommended_target": target,
            "confidence": 0.5,
            "reason": f"Policy fallback for '{intent}'",
            "alternatives": [],
            "warning": None,
        }

    def recommend_next_action(self, context: str, current_action: str, history: list[dict[str, Any]] | None = None, use_llm: bool = True) -> dict[str, Any]:
        """Recommend the next action + target given current state."""
        intent = self._detect_intent(context)
        step_idx = len(history) if history else 0
        state_str = f"{intent}:{current_action}:{step_idx}"

        # 1. Transition statistics
        transitions = dict(self.transition_counts.get(current_action, {}))
        total_trans = sum(transitions.values()) if transitions else 0

        # 2. Policy network
        available = list(transitions.keys()) if transitions else ["read_file", "terminal", "browser", "search", "llm_response", "execute_code"]

        if isinstance(self.policy, (MLPPolicy, MiniTransformerPolicy)):
            state_vec = self.state_encoder.encode(state_str, self._build_state_stats(intent, current_action))
            best_policy = self.policy.get_best(state_vec, available, self._action_to_idx)
        else:
            best_policy = self.policy.get_best(state_str, available)

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

        # 4. Target recommendation
        target = self._get_target_recommendation(recommended, context, current_action, history, use_llm)

        # 5. Check correction warnings
        corr_warning = self._check_correction_warning(current_action, intent, context)
        if corr_warning:
            warning = corr_warning

        alternatives = sorted(transitions.items(), key=lambda x: -x[1])[:3]

        return {
            "recommended_action": recommended,
            "recommended_target": target,
            "confidence": confidence,
            "reason": f"Based on {total_trans} transitions from '{current_action}'",
            "alternative_actions": alternatives,
            "warning": warning,
        }

    def _get_target_recommendation(
        self,
        action: str,
        context: str,
        current_action: str,
        history: list[dict[str, Any]] | None,
        use_llm: bool,
    ) -> str:
        """Recommend a concrete target for an action."""
        # 1. Check historical target counts
        targets = dict(self.target_counts.get(action, {}))
        if targets:
            best_target = max(targets, key=targets.get)
            if targets[best_target] >= 2:
                return best_target

        # 2. Check correction warnings for target
        for fix in self.correction_fixes:
            if fix["wrong_action"] == action and fix.get("wrong_target"):
                return f"AVOID: {fix['wrong_target']}"

        # 3. LLM fallback
        if use_llm:
            return self.kimi.infer_target(context, current_action, action)

        return ""

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
    def load(cls, path: str, pull_from_mem0: bool = True) -> "ActionRecommender":
        """Load recommender from JSON file, optionally merging shared Mem0 policy."""
        p = Path(path)
        rec = cls()

        # 1. Load local file
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            if "policy" in data:
                rec.policy = SoftmaxPolicy.load(str(p))
            rec.transition_counts = defaultdict(lambda: defaultdict(int))
            for k, v in data.get("transition_counts", {}).items():
                rec.transition_counts[k] = defaultdict(int, v)
            rec.first_action_stats = defaultdict(lambda: {"action": "", "successes": 0, "total": 0})
            rec.first_action_stats.update(data.get("first_action_stats", {}))
            rec.correction_fixes = data.get("correction_fixes", [])
            rec.action_rewards = defaultdict(list)
            rec.action_rewards.update({k: v for k, v in data.get("action_rewards", {}).items()})
            rec.target_counts = defaultdict(lambda: defaultdict(int))
            for k, v in data.get("target_counts", {}).items():
                rec.target_counts[k] = defaultdict(int, v)
            rec._agent_steps_total = data.get("agent_steps_total", 0)

        # 2. Pull shared policy from Mem0
        if pull_from_mem0:
            try:
                shared = rec.mem0_sync.pull_policy()
                if shared:
                    # Merge first actions
                    for intent, stats in shared.get("first_action_stats", {}).items():
                        local = rec.first_action_stats[intent]
                        if local["total"] == 0:
                            local["action"] = stats.get("action", "")
                        local["total"] += stats.get("total", 0)
                        local["successes"] += stats.get("successes", 0)

                    # Merge transitions
                    for cur, nexts in shared.get("transition_counts", {}).items():
                        for nxt, cnt in nexts.items():
                            rec.transition_counts[cur][nxt] += cnt

                    # Merge corrections
                    rec.correction_fixes.extend(shared.get("correction_fixes", []))

                    print(f"[agentRL] Merged shared policy from Mem0: {len(shared.get('first_action_stats', {}))} intents, {len(shared.get('transition_counts', {}))} transitions")
            except Exception as e:
                print(f"[agentRL] Mem0 pull skipped: {e}")

        return rec

    def save(self, path: str, push_to_mem0: bool = True) -> None:
        """Save recommender state to JSON file, optionally pushing to Mem0."""
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
            "target_counts": {k: dict(v) for k, v in self.target_counts.items()},
            "agent_steps_total": self._agent_steps_total,
            "policy_path": str(policy_path),
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Push to Mem0
        if push_to_mem0:
            try:
                ok = self.mem0_sync.push_policy(
                    first_action_stats=dict(self.first_action_stats),
                    transition_counts={k: dict(v) for k, v in self.transition_counts.items()},
                    correction_fixes=self.correction_fixes,
                    agent_steps_total=self._agent_steps_total,
                )
                if ok:
                    print("[agentRL] Policy snapshot pushed to Mem0")
            except Exception as e:
                print(f"[agentRL] Mem0 push skipped: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_policy(self, traj: TaskTrajectory, intent: str, agent_steps: list | None = None) -> None:
        """Update policy network with trajectory reward (agent actions only)."""
        steps = agent_steps if agent_steps is not None else [s for s in traj.steps if s.action_type != "user_input"]
        reward = self._outcome_to_reward(traj.final_outcome)

        # Neural policy update
        if isinstance(self.policy, (MLPPolicy, MiniTransformerPolicy)):
            for i, step in enumerate(steps):
                full_idx = traj.steps.index(step) if step in traj.steps else -1
                is_corrected = full_idx in traj.correction_points if full_idx >= 0 else False
                step_reward = reward - 0.5 if is_corrected else reward

                state_str = f"{intent}:{step.action_type}:{i}"
                stats = self._build_state_stats(intent, step.action_type)
                state_vec = self.state_encoder.encode(state_str, stats)

                action_idx = self._action_to_idx.get(step.action_type, 0)
                self.policy.update(state_vec, action_idx, step_reward)
            return

        # Legacy tabular policy update
        for i, step in enumerate(steps):
            state = f"{intent}:{step.action_type}:{i}"
            full_idx = traj.steps.index(step) if step in traj.steps else -1
            is_corrected = full_idx in traj.correction_points if full_idx >= 0 else False
            step_reward = reward - 0.5 if is_corrected else reward
            self.policy.update(state, step.action_type, step_reward)

    def _build_state_stats(self, intent: str, action: str) -> dict[str, float]:
        """Build extra statistics for state encoding."""
        stats = self.first_action_stats.get(intent, {})
        total = stats.get("total", 0)
        successes = stats.get("successes", 0)
        corrections = sum(1 for f in self.correction_fixes if f["wrong_action"] == action and f["intent"] == intent)
        return {
            "visit_count": total,
            "success_rate": successes / total if total > 0 else 0.5,
            "correction_rate": corrections / total if total > 0 else 0.0,
        }

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
