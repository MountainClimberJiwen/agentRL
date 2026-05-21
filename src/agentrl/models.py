from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class UnifiedTurn:
    backend: str
    session_id: str
    turn_id: str
    timestamp: datetime
    user_input: str
    assistant_response: str
    tool_calls: list[dict] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    outcome: str = "unknown"
    outcome_confidence: float = 0.0
    raw_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedSession:
    backend: str
    session_id: str
    created_at: Optional[datetime] = None
    turns: list[UnifiedTurn] = field(default_factory=list)
    raw_meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fine-grained Trajectory Models
# ---------------------------------------------------------------------------

@dataclass
class ActionStep:
    """A single atomic action inside an agent turn (tool call or thought)."""
    step_id: str
    timestamp: Optional[datetime]
    action_type: str  # e.g. "read_file", "write_file", "patch", "terminal", "browser", "thought", "llm_response"
    target: str       # file path, URL, command, etc.
    content_preview: str = ""
    result_preview: str = ""
    user_reaction: str = "silent"  # "approve", "correct", "reject", "silent"
    correction_type: str = ""      # "sequence", "tool", "file", "scope", "content", ""
    correction_detail: str = ""    # raw user text that triggered correction
    reward_delta: float = 0.0      # estimated step-level reward contribution


@dataclass
class CorrectionPattern:
    """A mined correction pattern from historical trajectories."""
    pattern_type: str           # "sequence", "tool", "file", "scope", "content"
    trigger_condition: str      # human-readable condition
    expected_action: str        # what the agent should have done instead
    frequency: int = 0          # how many times seen
    success_after_fix: float = 0.0  # avg reward after applying this fix
    contexts: list[str] = field(default_factory=list)


@dataclass
class TaskTrajectory:
    """Full task-level trajectory with step-by-step reconstruction."""
    trajectory_id: str
    session_id: str
    backend: str
    goal: str = ""                  # first user message = task goal
    steps: list[ActionStep] = field(default_factory=list)
    correction_points: list[int] = field(default_factory=list)  # indices into steps
    patterns_found: list[CorrectionPattern] = field(default_factory=list)
    final_outcome: str = "unknown"
    final_reward: float = 0.0
    raw_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_corrections(self) -> int:
        return len(self.correction_points)

    @property
    def has_early_mistake(self) -> bool:
        """True if first correction happens in first 30% of steps."""
        if not self.correction_points:
            return False
        return self.correction_points[0] < max(1, len(self.steps) * 0.3)
