"""
Correction Miner — Extract structured correction signals from user messages.

Turns vague user pushback like "不对，你应该先看 README" into typed,
actionable patterns that agentRL can learn from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MinedCorrection:
    pattern_type: str           # sequence | tool | file | scope | content | reject | unclear
    confidence: float           # 0..1
    trigger_phrase: str         # matched substring
    expected_action: str        # what agent should do instead (human readable)
    affected_step_hint: str     # e.g. "read_file", "write_file", "terminal"
    raw_text: str = ""


class CorrectionMiner:
    """Rule-based + lightweight heuristic correction classifier.

    Designed to be replaced later by a small fine-tuned classifier
    once enough labelled data is collected.
    """

    # ------------------------------------------------------------------
    # Pattern definitions: (type, regex, expected_action_template, affected_hint)
    # ------------------------------------------------------------------
    _RULES = [
        # --- SEQUENCE ---
        (
            "sequence",
            r"(先|第一步|first)应该.*然后|先.*再|先.*然后|反过来|顺序错了|do .* first|first do|then do|reverse the order",
            "{actor} should perform steps in correct order: {expected}",
            "sequence",
        ),
        (
            "sequence",
            r"(不要.*先|不要.*再|别急着|等一下|暂时不要|not yet|hold on|wait before)",
            "{actor} should defer '{current}' until prerequisite '{expected}' is done",
            "sequence",
        ),

        # --- TOOL ---
        (
            "tool",
            r"(不要用|别用|改用|换成|改成|用.*代替|don't use|stop using|use .* instead|switch to)",
            "{actor} should use tool '{expected}' instead of '{current}'",
            "tool",
        ),
        (
            "tool",
            r"(为什么不用|怎么不用|why not use|why didn't you use)",
            "{actor} should consider using tool '{expected}'",
            "tool",
        ),

        # --- FILE ---
        (
            "file",
            r"(不要改|别改|别动|不用管|不要动|别碰|don't modify|don't touch|leave .* alone|ignore this file|skip this)",
            "{actor} should NOT modify file '{current}'",
            "write_file",
        ),
        (
            "file",
            r"(先看|先读|先检查|查看|看看.*文件|你看了没|read .* first|check .* first|look at .* first)",
            "{actor} should read file '{expected}' before proceeding",
            "read_file",
        ),
        (
            "file",
            r"(不是这个文件|文件错了|找错了|wrong file|not this file|wrong path)",
            "{actor} should operate on correct file '{expected}' instead of '{current}'",
            "read_file",
        ),

        # --- SCOPE ---
        (
            "scope",
            r"(范围太大|范围太小|只看|只关注|只需要|不需要看|不用管|too broad|narrow down|only look at|focus on|ignore the rest)",
            "{actor} should narrow scope to '{expected}' and ignore '{current}'",
            "scope",
        ),
        (
            "scope",
            r"(还有其他|也要看|也要改|缺了|missed|forgot|also need|you missed)",
            "{actor} should expand scope to include '{expected}'",
            "scope",
        ),

        # --- CONTENT ---
        (
            "content",
            r"(这里应该|这里是|改成|修改为|替换为|should be|change to|replace with|update to|use .* here)",
            "{actor} should change content to '{expected}'",
            "content",
        ),
        (
            "content",
            r"(不对|错了|不正确|有误|不是这样|incorrect|wrong|not right|that's not)",
            "{actor} should correct the output/content",
            "content",
        ),

        # --- REJECT / STOP ---
        (
            "reject",
            r"(算了|别做了|停止|不要继续|stop|abort|give up|forget it|never mind)",
            "{actor} should stop current task",
            "stop",
        ),
    ]

    def mine(self, user_text: str, current_context: str = "") -> Optional[MinedCorrection]:
        """Classify a user message into a structured correction.

        Args:
            user_text: raw user message
            current_context: brief description of what agent just did
        """
        if not user_text:
            return None

        text = user_text.lower().strip()
        best: Optional[MinedCorrection] = None

        for pat_type, regex, template, hint in self._RULES:
            m = re.search(regex, text, re.IGNORECASE)
            if not m:
                continue

            trigger = m.group(0)
            confidence = self._score_match(text, trigger, pat_type)

            expected = self._infer_expected_action(text, current_context, pat_type)
            action_str = template.format(actor="Agent", current=current_context, expected=expected)

            if best is None or confidence > best.confidence:
                best = MinedCorrection(
                    pattern_type=pat_type,
                    confidence=confidence,
                    trigger_phrase=trigger,
                    expected_action=action_str,
                    affected_step_hint=hint,
                    raw_text=user_text,
                )

        return best

    # ------------------------------------------------------------------
    # Scoring heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_match(full_text: str, trigger: str, pat_type: str) -> float:
        base = 0.6
        # Longer triggers tend to be more specific
        base += min(0.1, len(trigger) / 100.0)
        # Explicit imperative mood boosts confidence
        if any(w in full_text for w in ("应该", "要", "先", "别", "不要", "should", "must", "do")):
            base += 0.15
        # Reject patterns are strong signals but lower confidence if ambiguous
        if pat_type == "reject" and len(full_text) < 10:
            base -= 0.1
        return min(0.95, base)

    @staticmethod
    def _infer_expected_action(text: str, context: str, pat_type: str) -> str:
        """Naive extraction of what the user wants instead."""
        # Try to extract file paths
        file_pats = re.findall(r"[\w\-/\.]+\.(?:py|js|ts|json|md|txt|yaml|yml|sh)", text)
        if file_pats:
            return file_pats[-1]
        # Try to extract tool names
        tool_pats = re.findall(r"(?:read_file|write_file|patch|terminal|browser|search|git)", text, re.I)
        if tool_pats:
            return tool_pats[-1].lower()
        # Fallback
        return "(see user message)"

    # ------------------------------------------------------------------
    # Aggregate patterns across trajectories
    # ------------------------------------------------------------------

    def aggregate(
        self, corrections: list[MinedCorrection]
    ) -> list[dict[str, object]]:
        """Group mined corrections by type and surface frequent patterns."""
        from collections import Counter

        buckets: dict[str, list[MinedCorrection]] = {}
        for c in corrections:
            buckets.setdefault(c.pattern_type, []).append(c)

        results = []
        for ptype, items in buckets.items():
            triggers = Counter(c.trigger_phrase for c in items)
            top_trigger = triggers.most_common(1)[0][0] if triggers else ""
            results.append({
                "pattern_type": ptype,
                "frequency": len(items),
                "avg_confidence": round(sum(c.confidence for c in items) / len(items), 2),
                "top_trigger": top_trigger,
                "examples": [c.raw_text for c in items[:3]],
            })
        return results
