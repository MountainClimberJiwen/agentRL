from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class UnifiedTurn:
    backend: str
    session_id: str
    turn_id: str
    timestamp: datetime
    user_input: str = ""
    assistant_response: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    outcome: str = "unknown"
    outcome_confidence: float = 0.0
    pending_approval: bool = False
    approval_resolved: Optional[bool] = None
    duration_ms: Optional[int] = None
    raw_meta: dict = field(default_factory=dict)


@dataclass
class UnifiedSession:
    backend: str
    session_id: str
    created_at: Optional[datetime] = None
    cwd: Optional[str] = None
    turns: list[UnifiedTurn] = field(default_factory=list)
    raw_meta: dict = field(default_factory=dict)
