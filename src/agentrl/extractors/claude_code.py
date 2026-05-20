from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from agentrl.models import UnifiedSession, UnifiedTurn
from agentrl.utils import extract_files_from_tools, parse_iso


class ClaudeCodeParser:
    """Parser for Anthropic Claude Code CLI sessions.

    Claude Code stores session histories as JSON files. Each session directory
    typically contains a `history.json` with a list of messages in Anthropic
    API format (role, content, optional tool_use / tool_result blocks).

    Expected layout::

        ~/.claude/sessions/
            <session_id>/
                history.json
                metadata.json   # optional
    """

    BASE_DIR = os.path.expanduser("~/.claude/sessions")
    BACKEND = "claude_code"

    def iter_sessions(self) -> Iterator[UnifiedSession]:
        base = Path(self.BASE_DIR)
        if not base.exists():
            return
        for sess_dir in sorted(base.iterdir()):
            if not sess_dir.is_dir():
                continue
            yield from self._parse_session_dir(sess_dir)

    def _parse_session_dir(self, sess_dir: Path) -> Iterator[UnifiedSession]:
        history_path = sess_dir / "history.json"
        meta_path = sess_dir / "metadata.json"

        if not history_path.exists():
            return

        meta: dict = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            return

        session_id = sess_dir.name
        created_at = parse_iso(meta.get("created_at"))

        session = UnifiedSession(
            backend=self.BACKEND,
            session_id=session_id,
            created_at=created_at,
            raw_meta={"dir": str(sess_dir), "meta": meta},
        )

        # Claude Code history is a flat list of messages.
        # We chunk them into turns: each user message starts a new turn,
        # followed by assistant message(s) and tool results.
        turns_data: list[dict[str, Any]] = []
        current_turn: dict[str, Any] | None = None

        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                if current_turn is not None:
                    self._finalize_turn(current_turn)
                    turns_data.append(current_turn)
                current_turn = {
                    "user_input": self._extract_text(content),
                    "assistant_response": "",
                    "tool_calls": [],
                    "outcome": "unknown",
                    "outcome_confidence": 0.0,
                    "timestamp": None,
                }

            elif role == "assistant" and current_turn is not None:
                text = self._extract_text(content)
                if text:
                    current_turn["assistant_response"] += text

                # Extract tool_use blocks
                for tc in self._extract_tool_calls(msg):
                    current_turn["tool_calls"].append(tc)

            elif role in ("tool", "tool_result") and current_turn is not None:
                # Tool results don't change the turn structure but may imply
                # the assistant attempted something. We record them loosely.
                pass

        if current_turn is not None:
            self._finalize_turn(current_turn)
            turns_data.append(current_turn)

        # Infer outcomes from the last turn and session metadata
        for i, data in enumerate(turns_data):
            turn = UnifiedTurn(
                backend=self.BACKEND,
                session_id=session_id,
                turn_id=str(i + 1),
                timestamp=data.get("timestamp") or datetime.now(timezone.utc),
                user_input=data.get("user_input", ""),
                assistant_response=data.get("assistant_response", ""),
                tool_calls=data.get("tool_calls", []),
                files_read=data.get("files_read", []),
                files_written=data.get("files_written", []),
                outcome=data.get("outcome", "unknown"),
                outcome_confidence=data.get("outcome_confidence", 0.0),
                pending_approval=False,
                raw_meta={},
            )
            session.turns.append(turn)

        # Simple outcome heuristics on the session level
        if session.turns:
            last = session.turns[-1]
            if not last.assistant_response and not last.tool_calls:
                last.outcome = "exited"
                last.outcome_confidence = 0.6
            elif meta.get("ended_by") == "user_abort":
                last.outcome = "exited"
                last.outcome_confidence = 0.9
            elif meta.get("ended_by") == "approval":
                last.outcome = "approved"
                last.outcome_confidence = 0.9

        if session.turns:
            yield session

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Normalize Anthropic-style content to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        texts.append(part.get("text", ""))
            return "\n".join(texts)
        return ""

    @staticmethod
    def _extract_tool_calls(msg: dict) -> list[dict]:
        """Extract tool calls from Anthropic API message format."""
        calls = []
        content = msg.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "tool_use":
                    calls.append({
                        "id": part.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": part.get("name", ""),
                            "arguments": json.dumps(part.get("input", {})),
                        },
                    })
        # Also handle OpenAI-compatible tool_calls field
        for tc in msg.get("tool_calls", []):
            calls.append(tc)
        return calls

    @staticmethod
    def _finalize_turn(turn: dict) -> None:
        reads, writes = extract_files_from_tools(turn.get("tool_calls", []))
        turn["files_read"] = reads
        turn["files_written"] = writes
        if turn.get("outcome") == "unknown":
            if turn.get("assistant_response") or turn.get("tool_calls"):
                turn["outcome"] = "completed"
                turn["outcome_confidence"] = 0.5
            else:
                turn["outcome"] = "exited"
                turn["outcome_confidence"] = 0.5
