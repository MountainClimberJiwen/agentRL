from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentrl.models import UnifiedTurn


def accuracy_reward(outcome: str) -> float:
    return {
        "approved": 1.0,
        "completed": 0.5,
        "corrected_completed": 0.3,
        "corrected_exited": -0.7,
        "rejected": -0.8,
        "exited": -1.0,
        "timeout": -0.5,
        "unknown": 0.0,
        "corrected": -0.3,
    }.get(outcome, 0.0)


def grounding_reward(tool_calls: list[dict], correction_text: str) -> float:
    if not correction_text:
        return 0.0
    c = correction_text.lower()
    if any(k in c for k in ("文件", "file", "路径", "path", "不该", "不要看", "应该看")):
        return -0.5
    return -0.1


def temporal_reward(query: str, correction_text: str) -> float:
    if not correction_text:
        return 0.0
    c = correction_text.lower()
    markers = ["昨天", "前天", "上周", "时间", "不是", "才是", "日期", "day", "week", "month"]
    if any(m in c for m in markers):
        return -0.6
    return 0.0


def compute_all_rewards(turn: "UnifiedTurn") -> tuple[float, float, float]:
    r_acc = accuracy_reward(turn.outcome)
    r_ground = grounding_reward(turn.tool_calls, turn.raw_meta.get("correction_text", ""))
    r_temp = temporal_reward(turn.user_input or "", turn.raw_meta.get("correction_text", ""))
    return r_acc, r_ground, r_temp
