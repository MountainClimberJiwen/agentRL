from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentrl.models import UnifiedTurn

DEFAULT_DB = Path(__file__).parent.parent.parent / "data" / "agentrl.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    backend TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    query TEXT,
    assistant_response TEXT,
    selected_sessions TEXT,
    selected_utterances TEXT,
    predicted_time_range TEXT,
    user_outcome TEXT NOT NULL,
    outcome_confidence REAL DEFAULT 1.0,
    last_event_type TEXT,
    last_event_at TEXT,
    exit_detected_at TEXT,
    next_session_id TEXT,
    next_session_gap_minutes REAL,
    correction_text TEXT,
    files_read TEXT,
    files_written TEXT,
    tool_calls TEXT,
    has_temporal_query INTEGER DEFAULT 0,
    computed_reward_accuracy REAL,
    computed_reward_grounding REAL,
    computed_reward_temporal REAL,
    raw_meta TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, turn_id, backend)
);
CREATE INDEX IF NOT EXISTS idx_memory_feedback_session ON memory_feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_feedback_outcome ON memory_feedback(user_outcome);
CREATE INDEX IF NOT EXISTS idx_memory_feedback_backend_created ON memory_feedback(backend, created_at);
CREATE INDEX IF NOT EXISTS idx_memory_feedback_temporal ON memory_feedback(has_temporal_query);
"""


def init_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def upsert_turn(
    conn: sqlite3.Connection,
    turn: "UnifiedTurn",
    reward_accuracy: float,
    reward_grounding: float,
    reward_temporal: float,
) -> bool:
    cursor = conn.cursor()
    has_temporal = 1 if turn.user_input and _has_temporal(turn.user_input) else 0
    raw_meta_json = json.dumps(turn.raw_meta, ensure_ascii=False)
    ts = turn.timestamp.isoformat() if turn.timestamp else None

    def _s(v) -> str | None:
        return str(v) if v is not None else None

    cursor.execute(
        """
        INSERT INTO memory_feedback (
            session_id, backend, turn_id, query, assistant_response,
            user_outcome, outcome_confidence, files_read, files_written,
            tool_calls, has_temporal_query, computed_reward_accuracy,
            computed_reward_grounding, computed_reward_temporal,
            raw_meta, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, turn_id, backend) DO UPDATE SET
            query = excluded.query,
            assistant_response = excluded.assistant_response,
            user_outcome = excluded.user_outcome,
            outcome_confidence = excluded.outcome_confidence,
            files_read = excluded.files_read,
            files_written = excluded.files_written,
            tool_calls = excluded.tool_calls,
            has_temporal_query = excluded.has_temporal_query,
            computed_reward_accuracy = excluded.computed_reward_accuracy,
            computed_reward_grounding = excluded.computed_reward_grounding,
            computed_reward_temporal = excluded.computed_reward_temporal,
            raw_meta = excluded.raw_meta,
            created_at = excluded.created_at
        """,
        (
            _s(turn.session_id),
            _s(turn.backend),
            _s(turn.turn_id),
            _s(turn.user_input),
            _s(turn.assistant_response),
            _s(turn.outcome),
            turn.outcome_confidence,
            json.dumps(turn.files_read, ensure_ascii=False),
            json.dumps(turn.files_written, ensure_ascii=False),
            json.dumps(turn.tool_calls, ensure_ascii=False),
            has_temporal,
            reward_accuracy,
            reward_grounding,
            reward_temporal,
            raw_meta_json,
            ts,
        ),
    )
    conn.commit()
    return cursor.rowcount > 0


def stats(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    print("\n=== memory_feedback stats ===")
    cur.execute("SELECT COUNT(*) FROM memory_feedback")
    print(f"Total rows: {cur.fetchone()[0]}")
    cur.execute(
        """
        SELECT backend, COUNT(*),
               SUM(CASE WHEN has_temporal_query=1 THEN 1 ELSE 0 END)
        FROM memory_feedback GROUP BY backend
        """
    )
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} turns, {row[2]} temporal")
    cur.execute(
        "SELECT user_outcome, COUNT(*) FROM memory_feedback GROUP BY user_outcome ORDER BY COUNT(*) DESC"
    )
    print("\nOutcomes:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")
    cur.execute(
        "SELECT AVG(computed_reward_accuracy), AVG(computed_reward_grounding), AVG(computed_reward_temporal) FROM memory_feedback"
    )
    a, g, t = cur.fetchone()
    print(f"\nAvg rewards: accuracy={a:.3f}, grounding={g:.3f}, temporal={t:.3f}")


def _has_temporal(text: str) -> bool:
    from agentrl.utils import has_temporal_keywords
    return has_temporal_keywords(text)
