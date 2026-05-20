"""Memory retrieval: coarse filtering and prompt-based selection.

All memory policy is expressed as PROMPT TEXT — no model weights are trained.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from agentrl.prompts.registry import PromptRegistry, RetrievalStrategy, SelectedEvidence
from agentrl.utils import has_temporal_keywords


@dataclass
class CandidateSession:
    """A session candidate after coarse filtering."""

    session_id: str
    turn_id: str
    backend: str
    query: str
    assistant_response: str
    user_outcome: str
    created_at: datetime
    computed_reward_accuracy: float | None = None
    files_read: list[str] = None
    files_written: list[str] = None
    tool_calls: list[dict] = None

    def __post_init__(self):
        if self.files_read is None:
            self.files_read = []
        if self.files_written is None:
            self.files_written = []
        if self.tool_calls is None:
            self.tool_calls = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "backend": self.backend,
            "query": (self.query or "")[:300],
            "assistant_response": (self.assistant_response or "")[:300],
            "user_outcome": self.user_outcome,
            "created_at": self.created_at.isoformat(),
            "computed_reward_accuracy": self.computed_reward_accuracy,
            "files_read": self.files_read[:5],
            "files_written": self.files_written[:5],
        }


def _parse_datetime(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _extract_project_from_query(query: str | None) -> str | None:
    """Heuristic: extract project/directory hint from query text."""
    if not query:
        return None
    # Look for patterns like "Active file: scripts/foo.py" or file paths
    m = re.search(r"Active file:\s*(\S+)", query)
    if m:
        path = m.group(1)
        parts = Path(path).parts
        if len(parts) > 1:
            return parts[0]  # e.g., "scripts" from "scripts/foo.py"
    # Look for directory paths
    m = re.search(r"under folder\s+(\S+)", query, re.IGNORECASE)
    if m:
        return m.group(1).strip("./")
    return None


def _extract_keywords(query: str | None) -> list[str]:
    """Simple keyword extraction for coarse filtering."""
    if not query:
        return []
    # Remove common stop words and punctuation
    text = re.sub(r"[^\w\s]", " ", query.lower())
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "ought", "used", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between",
        "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "and",
        "but", "if", "or", "because", "until", "while", "this", "that",
        "these", "those", "i", "me", "my", "myself", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "it", "its",
        "they", "them", "their", "what", "which", "who", "whom",
    }
    words = [w for w in text.split() if len(w) > 2 and w not in stop]
    # Return unique keywords, capped
    seen = set()
    result = []
    for w in words:
        if w not in seen:
            seen.add(w)
            result.append(w)
        if len(result) >= 10:
            break
    return result


class CoarseFilter:
    """Phase 1 of Memory-T1: prune the full memory into a candidate pool.

    This is purely rule-based — no LLM involved.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = db_path

    def filter(
        self,
        query: str,
        project_hint: str | None = None,
        time_scope: tuple[datetime, datetime] | None = None,
        keywords: list[str] | None = None,
        outcome_bias: str = "any",
        max_candidates: int = 50,
        exclude_session_ids: set[str] | None = None,
    ) -> list[CandidateSession]:
        """Return candidate sessions from the DB."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Build query dynamically
        conditions: list[str] = []
        params: list[Any] = []

        # Time filtering
        if time_scope:
            t_start, t_end = time_scope
            conditions.append("created_at >= ? AND created_at <= ?")
            params.extend([t_start.isoformat(), t_end.isoformat()])

        # Outcome bias
        if outcome_bias == "approved_only":
            conditions.append("user_outcome = 'approved'")
        elif outcome_bias == "avoid_exited":
            conditions.append("user_outcome NOT IN ('exited', 'rejected')")

        # Keyword filtering via LIKE (coarse)
        kw_list = keywords or _extract_keywords(query)
        if kw_list:
            like_clauses = " OR ".join(["query LIKE ?"] * len(kw_list))
            conditions.append(f"({like_clauses})")
            params.extend([f"%{k}%" for k in kw_list])

        # Exclude current session to avoid self-reference
        if exclude_session_ids:
            placeholders = ",".join(["?"] * len(exclude_session_ids))
            conditions.append(f"session_id NOT IN ({placeholders})")
            params.extend(list(exclude_session_ids))

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT session_id, turn_id, backend, query, assistant_response,
                   user_outcome, created_at, computed_reward_accuracy,
                   files_read, files_written, tool_calls
            FROM memory_feedback
            WHERE {where_clause}
            ORDER BY
                CASE user_outcome
                    WHEN 'approved' THEN 3
                    WHEN 'completed' THEN 2
                    WHEN 'corrected' THEN 1
                    WHEN 'unknown' THEN 0
                    ELSE -1
                END DESC,
                computed_reward_accuracy DESC NULLS LAST,
                created_at DESC
            LIMIT ?
        """
        params.append(max_candidates)

        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()

        candidates = []
        for row in rows:
            created = _parse_datetime(row["created_at"])
            if not created:
                continue
            candidates.append(
                CandidateSession(
                    session_id=row["session_id"],
                    turn_id=row["turn_id"],
                    backend=row["backend"],
                    query=row["query"] or "",
                    assistant_response=row["assistant_response"] or "",
                    user_outcome=row["user_outcome"],
                    created_at=created,
                    computed_reward_accuracy=row["computed_reward_accuracy"],
                    files_read=_parse_json_list(row["files_read"]),
                    files_written=_parse_json_list(row["files_written"]),
                    tool_calls=_parse_json_list(row["tool_calls"]),
                )
            )
        return candidates


def _parse_json_list(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


class MemoryRouter:
    """Phase 2a: Render a prompt that asks the LLM to produce a RetrievalStrategy.

    The LLM is NOT called here — the caller sends the rendered prompt to their
    frozen LLM and parses the response with parse_strategy().
    """

    def __init__(self, registry: PromptRegistry | None = None):
        self.registry = registry or PromptRegistry()

    def render(
        self,
        query: str,
        project: str = "",
        current_time: datetime | None = None,
        recent_sessions: list[str] | None = None,
    ) -> str:
        """Return the filled Memory Router prompt to send to the LLM."""
        template = self.registry.load("router")
        now = (current_time or datetime.now(timezone.utc)).isoformat()
        recent = ", ".join(recent_sessions or [])[:500]
        return template.format(
            project=project or "unknown",
            current_time=now,
            recent_sessions=recent or "none",
            query=query,
        )

    def parse_strategy(self, response: str) -> RetrievalStrategy:
        """Parse LLM response (expected JSON) into RetrievalStrategy."""
        # Try to extract JSON from the response
        text = response.strip()
        # Remove markdown fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to find JSON object in the text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        ts = data.get("time_scope", {})
        start = _parse_datetime(ts.get("start")) if ts else None
        end = _parse_datetime(ts.get("end")) if ts else None

        # If temporal keywords in query but no time scope parsed, do heuristic
        if not start and not end and has_temporal_keywords(query):
            start, end = _infer_time_scope_from_query(query)

        return RetrievalStrategy(
            time_scope={
                "start": start.isoformat() if start else None,
                "end": end.isoformat() if end else None,
                "description": ts.get("description", "") if ts else "",
            },
            project_filter=data.get("project_filter", "any"),
            keywords=data.get("keywords", []),
            outcome_bias=data.get("outcome_bias", "any"),
            max_sessions=min(max(data.get("max_sessions", 20), 1), 100),
            reasoning=data.get("reasoning", ""),
        )


class EvidenceSelector:
    """Phase 2b: Render a prompt that asks the LLM to select precise evidence.

    The LLM is NOT called here — the caller sends the rendered prompt to their
    frozen LLM and parses the response with parse_selection().
    """

    def __init__(self, registry: PromptRegistry | None = None):
        self.registry = registry or PromptRegistry()

    def render(
        self,
        query: str,
        candidates: list[CandidateSession],
        time_scope: dict[str, Any] | None = None,
        project: str = "",
        max_sessions: int = 10,
    ) -> str:
        """Return the filled Evidence Selection prompt to send to the LLM."""
        template = self.registry.load("selector")
        ts_str = json.dumps(time_scope, ensure_ascii=False) if time_scope else "null"

        # Format candidates for the prompt
        candidate_lines = []
        for i, c in enumerate(candidates, 1):
            candidate_lines.append(
                f"[{i}] Session: {c.session_id[:8]}... | Turn: {c.turn_id[:8]}...\n"
                f"    Time: {c.created_at.isoformat()} | Outcome: {c.user_outcome}\n"
                f"    Query: {c.query[:200]}{'...' if len(c.query) > 200 else ''}\n"
                f"    Response: {c.assistant_response[:200]}{'...' if len(c.assistant_response) > 200 else ''}\n"
                f"    Files: {', '.join(c.files_read[:3]) or 'none'}\n"
            )

        return template.format(
            query=query,
            time_scope=ts_str,
            project=project or "unknown",
            max_sessions=max_sessions,
            candidates="\n".join(candidate_lines),
        )

    def parse_selection(self, response: str) -> list[SelectedEvidence]:
        """Parse LLM response (expected JSON array) into SelectedEvidence list."""
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(data, list):
            return []

        results = []
        for item in data:
            if not isinstance(item, dict):
                continue
            results.append(
                SelectedEvidence(
                    session_id=item.get("session_id", ""),
                    turn_id=item.get("turn_id", ""),
                    reason=item.get("reason", ""),
                    relevance_score=float(item.get("relevance_score", 0.0)),
                )
            )
        return results


def _infer_time_scope_from_query(query: str) -> tuple[datetime | None, datetime | None]:
    """Heuristic fallback: infer time range from temporal keywords in query."""
    now = datetime.now(timezone.utc)
    q = (query or "").lower()

    if "昨天" in q or "yesterday" in q:
        start = now - timedelta(days=1)
        end = now
    elif "前天" in q:
        start = now - timedelta(days=2)
        end = now - timedelta(days=1)
    elif "上周" in q or "last week" in q:
        start = now - timedelta(days=14)
        end = now - timedelta(days=7)
    elif "三天前" in q or "3 days ago" in q:
        start = now - timedelta(days=4)
        end = now - timedelta(days=2)
    elif "一周前" in q or "a week ago" in q:
        start = now - timedelta(days=10)
        end = now - timedelta(days=5)
    else:
        return None, None

    return start.replace(hour=0, minute=0, second=0, microsecond=0), end
