"""Memos-local (SQLite) memory client."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from agentrl.memory.unified_client import UnifiedMemoryClient


class MemosLocalMemoryClient(UnifiedMemoryClient):
    """
    Local SQLite backend for memos-local-hermes-plugin.
    Directly reads/writes ~/.hermes/memos/memos.db without requiring Hermes runtime.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or self._default_db_path()
        self._agent_id = os.environ.get("AGENTRL_AGENT_ID", "agentrl")

    def _default_db_path(self) -> str:
        hermes_home = Path.home() / ".hermes"
        return str(hermes_home / "memos" / "memos.db")

    def is_available(self) -> bool:
        return os.path.exists(self.db_path)

    # ------------------------------------------------------------------
    # Policy sync
    # ------------------------------------------------------------------

    def push_policy(self, policy_data: dict[str, Any]) -> bool:
        """Store policy snapshot as a shared memory chunk."""
        if not self.is_available():
            return False

        content = json.dumps(policy_data, indent=2, ensure_ascii=False)
        chunk_id = f"agentrl_policy_{int(time.time() * 1000)}"

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute(
                    """
                    INSERT INTO chunks (id, sessionKey, turnId, seq, role, content, kind,
                                        owner, visibility, createdAt, updatedAt)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        "agentrl_sync",
                        "policy",
                        0,
                        "system",
                        content,
                        "agentrl_policy",
                        self._agent_id,
                        "shared",
                        int(time.time()),
                        int(time.time()),
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            print(f"[agentRL] memos push_policy failed: {e}")
            return False

    def pull_policy(self) -> dict[str, Any] | None:
        """Load the latest shared policy snapshot."""
        if not self.is_available():
            return None

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT content FROM chunks
                    WHERE kind = 'agentrl_policy' AND visibility = 'shared'
                    ORDER BY createdAt DESC
                    LIMIT 1
                    """
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row["content"])
                return None
        except Exception as e:
            print(f"[agentRL] memos pull_policy failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Generic memory
    # ------------------------------------------------------------------

    def write_memory(self, content: str, metadata: dict[str, Any] | None = None) -> bool:
        if not self.is_available():
            return False

        meta = metadata or {}
        chunk_id = meta.get("id", f"agentrl_mem_{int(time.time() * 1000)}")
        visibility = meta.get("visibility", "private")

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute(
                    """
                    INSERT INTO chunks (id, sessionKey, turnId, seq, role, content, kind,
                                        owner, visibility, createdAt, updatedAt)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        meta.get("sessionKey", "agentrl"),
                        meta.get("turnId", "memory"),
                        meta.get("seq", 0),
                        meta.get("role", "system"),
                        content,
                        meta.get("kind", "agentrl_memory"),
                        self._agent_id,
                        visibility,
                        int(time.time()),
                        int(time.time()),
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            print(f"[agentRL] memos write_memory failed: {e}")
            return False

    def search_memory(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        if not self.is_available():
            return []

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                # Simple LIKE search (memos-local has FTS5, but LIKE is safest fallback)
                cursor = conn.execute(
                    """
                    SELECT id, content, role, visibility, createdAt
                    FROM chunks
                    WHERE content LIKE ? AND (visibility = 'shared' OR owner = ?)
                    ORDER BY createdAt DESC
                    LIMIT ?
                    """,
                    (f"%{query}%", self._agent_id, limit),
                )
                rows = cursor.fetchall()
                return [
                    {
                        "id": r["id"],
                        "content": r["content"],
                        "role": r["role"],
                        "visibility": r["visibility"],
                        "created_at": r["createdAt"],
                    }
                    for r in rows
                ]
        except Exception as e:
            print(f"[agentRL] memos search_memory failed: {e}")
            return []
