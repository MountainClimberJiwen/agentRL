"""
Trajectory Builder — Reconstruct fine-grained task trajectories from session logs.

Turns a flat JSONL session into a step-by-step ActionStep graph
with explicit correction attribution.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from agentrl.models import ActionStep, TaskTrajectory
from agentrl.patterns.correction_miner import CorrectionMiner, MinedCorrection
from agentrl.utils import extract_files_from_tools, parse_iso


class TrajectoryBuilder:
    """Build TaskTrajectory from raw Hermes-style JSONL messages."""

    def __init__(self) -> None:
        self.miner = CorrectionMiner()

    def build(
        self,
        session_id: str,
        backend: str,
        messages: list[dict[str, Any]],
    ) -> TaskTrajectory:
        """Build a TaskTrajectory from raw messages.

        Message roles expected:
          - "user": user input
          - "assistant": LLM response + tool_calls
          - "tool": tool execution result (optional, if present)
        """
        goal = self._extract_goal(messages)
        steps: list[ActionStep] = []
        corrections: list[int] = []  # indices into steps

        # We process in triplets: user -> assistant -> (tool results)
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")
            ts = parse_iso(msg.get("timestamp"))

            if role == "user":
                # Look ahead for assistant response
                assistant_msg = None
                for j in range(i + 1, len(messages)):
                    if messages[j].get("role") == "assistant":
                        assistant_msg = messages[j]
                        break

                # --- User message itself becomes a "user_input" step ---
                user_step = ActionStep(
                    step_id=f"u{len(steps)}",
                    timestamp=ts,
                    action_type="user_input",
                    target="",
                    content_preview=self._preview(msg.get("content", "")),
                )
                steps.append(user_step)

                # Mine correction from user text using what assistant just did as context
                context_desc = ""
                if steps and len(steps) > 1:
                    context_desc = steps[-2].action_type
                mined = self.miner.mine(msg.get("content", ""), context_desc)
                if mined:
                    user_step.user_reaction = "correct"
                    user_step.correction_type = mined.pattern_type
                    user_step.correction_detail = mined.raw_text
                    corrections.append(len(steps) - 1)

                # --- Parse assistant's internal tool steps ---
                if assistant_msg:
                    tool_calls = assistant_msg.get("tool_calls", [])
                    # Each tool call becomes its own ActionStep
                    for tidx, tc in enumerate(tool_calls):
                        step = self._tool_call_to_step(tc, len(steps), ts)
                        # If preceding user step was a correction, mark affected steps
                        if mined and self._is_affected_by_correction(step, mined):
                            step.user_reaction = "correct"
                            step.correction_type = mined.pattern_type
                        steps.append(step)

                    # --- LLM text response step ---
                    content = assistant_msg.get("content", "")
                    if content:
                        finish = assistant_msg.get("finish_reason", "")
                        reaction = "approve" if finish == "stop" else "silent"
                        steps.append(ActionStep(
                            step_id=f"r{len(steps)}",
                            timestamp=ts,
                            action_type="llm_response",
                            target="",
                            content_preview=self._preview(content, 200),
                            user_reaction=reaction,
                        ))

                i += 1
            elif role == "assistant":
                # Assistant without preceding user (shouldn't happen often)
                i += 1
            else:
                i += 1

        # Derive final outcome from last few steps
        final_outcome = self._derive_final_outcome(steps)
        traj = TaskTrajectory(
            trajectory_id=f"{backend}_{session_id}",
            session_id=session_id,
            backend=backend,
            goal=goal,
            steps=steps,
            correction_points=corrections,
            final_outcome=final_outcome,
        )
        return traj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_goal(self, messages: list[dict]) -> str:
        """First user message = task goal."""
        for m in messages:
            if m.get("role") == "user":
                return self._preview(m.get("content", ""), 300)
        return ""

    def _tool_call_to_step(
        self, tc: dict, idx: int, ts: datetime | None
    ) -> ActionStep:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        name = fn.get("name", "") if isinstance(fn, dict) else ""
        args = fn.get("arguments", {}) if isinstance(fn, dict) else {}
        if isinstance(args, str):
            try:
                import json
                args = json.loads(args)
            except Exception:
                args = {}

        target = args.get("path") or args.get("file_path") or args.get("filename", "")
        target = target or args.get("command", "") or args.get("url", "")

        return ActionStep(
            step_id=f"t{idx}",
            timestamp=ts,
            action_type=name or "tool_call",
            target=target,
            content_preview=self._preview(str(args), 300),
        )

    def _is_affected_by_correction(self, step: ActionStep, mined: MinedCorrection) -> bool:
        """Heuristic: does this step match what the user complained about?"""
        hint = mined.affected_step_hint
        if hint == "sequence":
            return True  # sequence errors affect the whole flow
        if hint == "tool" and step.action_type in mined.expected_action:
            return True
        if hint in ("file", "read_file", "write_file") and step.target in mined.expected_action:
            return True
        if hint == "scope":
            return True
        if hint == "content" and step.action_type in ("write_file", "patch", "llm_response"):
            return True
        return False

    def _derive_final_outcome(self, steps: list[ActionStep]) -> str:
        """Derive outcome from last user_reaction signals."""
        if not steps:
            return "unknown"
        # If last user_input step was a correction -> overall corrected
        user_steps = [s for s in steps if s.action_type == "user_input"]
        if user_steps and user_steps[-1].user_reaction == "correct":
            return "corrected"
        # If any correction in the trajectory
        if any(s.user_reaction == "correct" for s in steps):
            return "corrected"
        # Check last LLM response finish_reason proxy
        resp_steps = [s for s in steps if s.action_type == "llm_response"]
        if resp_steps:
            last = resp_steps[-1]
            if last.user_reaction == "approve":
                return "approved"
        return "unknown"

    @staticmethod
    def _preview(text: str, max_len: int = 120) -> str:
        text = text.replace("\n", " ")
        if len(text) > max_len:
            return text[:max_len] + "..."
        return text
