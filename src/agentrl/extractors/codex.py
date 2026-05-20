from __future__ import annotations

import glob as glob_mod
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from agentrl.models import UnifiedSession, UnifiedTurn
from agentrl.utils import detect_outcome_from_correction, parse_iso


class CodexParser:
    GLOB = os.path.expanduser("~/.codex/sessions/**/*.jsonl")
    BACKEND = "codex"

    def iter_sessions(self) -> Iterator[UnifiedSession]:
        for p in sorted(glob_mod.glob(self.GLOB, recursive=True)):
            yield from self._parse_file(Path(p))

    def _parse_file(self, path: Path) -> Iterator[UnifiedSession]:
        events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not events:
            return

        session_id = ""
        created_at = None
        cwd = None
        for ev in events:
            if ev.get("type") == "session_meta":
                payload = ev.get("payload", {})
                session_id = payload.get("id", "")
                created_at = parse_iso(payload.get("timestamp"))
                cwd = payload.get("cwd", "")
                break

        if not session_id:
            session_id = path.stem

        session = UnifiedSession(
            backend=self.BACKEND,
            session_id=session_id,
            created_at=created_at,
            cwd=cwd,
            raw_meta={"file": str(path)},
        )

        turns_data: dict[str, dict[str, Any]] = {}
        pending_approvals: dict[str, dict] = {}
        last_event_per_turn: dict[str, dict] = {}

        for ev in events:
            ts = parse_iso(ev.get("timestamp"))
            ev_type = ev.get("type")
            payload = ev.get("payload", {})

            if ev_type == "event_msg":
                msg_type = payload.get("type")
                turn_id = payload.get("turn_id", "")
                last_event_per_turn[turn_id] = {
                    "type": msg_type, "timestamp": ts, "payload": payload,
                }

                if msg_type == "exec_command":
                    sandbox = payload.get("sandbox_permissions", "")
                    if sandbox == "require_escalated":
                        pending_approvals[turn_id] = {
                            "command": payload.get("command", ""),
                            "timestamp": ts,
                        }

                elif msg_type == "task_complete":
                    turns_data.setdefault(turn_id, {})
                    turns_data[turn_id]["outcome"] = "approved"
                    turns_data[turn_id]["outcome_confidence"] = 1.0
                    turns_data[turn_id]["assistant_response"] = payload.get("last_agent_message", "")
                    turns_data[turn_id]["completed_at"] = payload.get("completed_at")
                    turns_data[turn_id]["duration_ms"] = payload.get("duration_ms")

                elif msg_type == "turn_aborted":
                    turns_data.setdefault(turn_id, {})
                    reason = payload.get("reason", "").lower()
                    if reason == "interrupted":
                        turns_data[turn_id]["outcome"] = "exited"
                        turns_data[turn_id]["outcome_confidence"] = 0.8
                    else:
                        turns_data[turn_id]["outcome"] = "rejected"
                        turns_data[turn_id]["outcome_confidence"] = 0.9

                elif msg_type == "user_message":
                    turns_data.setdefault(turn_id, {})
                    content = payload.get("message", "")
                    turns_data[turn_id]["user_input"] = content
                    corr, conf = detect_outcome_from_correction(content)
                    if corr:
                        turns_data[turn_id]["correction_detected"] = True
                        turns_data[turn_id]["correction_text"] = content
                        if "outcome" not in turns_data[turn_id]:
                            turns_data[turn_id]["outcome"] = "corrected"
                            turns_data[turn_id]["outcome_confidence"] = conf

                elif msg_type == "agent_message":
                    turns_data.setdefault(turn_id, {})
                    content = payload.get("message", "") or payload.get("content", "")
                    corr, conf = detect_outcome_from_correction(content)
                    if corr:
                        turns_data[turn_id]["correction_detected"] = True
                        turns_data[turn_id]["correction_text"] = content
                        if "outcome" not in turns_data[turn_id]:
                            turns_data[turn_id]["outcome"] = "corrected"
                            turns_data[turn_id]["outcome_confidence"] = conf

            elif ev_type == "response_item":
                resp = payload.get("response", {})
                turn_id = resp.get("turn_id", "")
                content = ""
                for item in resp.get("content", []):
                    if item.get("type") == "text":
                        content += item.get("text", "")
                turns_data.setdefault(turn_id, {})
                turns_data[turn_id]["assistant_response"] = content
                if "outcome" not in turns_data[turn_id]:
                    turns_data[turn_id]["outcome"] = "completed"
                    turns_data[turn_id]["outcome_confidence"] = 0.5

            elif ev_type == "turn_context":
                turn_id = payload.get("turn_id", "")
                user_msg = ""
                for msg in payload.get("messages", []):
                    if msg.get("role") == "user":
                        user_msg = msg.get("content", "")
                        break
                turns_data.setdefault(turn_id, {})
                if user_msg:
                    turns_data[turn_id]["user_input"] = user_msg
                turns_data[turn_id]["timestamp"] = ts

        for turn_id, data in list(turns_data.items()):
            if turn_id in pending_approvals and data.get("outcome") not in ("approved", "rejected"):
                last_ev = last_event_per_turn.get(turn_id, {})
                if last_ev.get("type") in ("exec_command", "user_message"):
                    data["outcome"] = "exited"
                    data["outcome_confidence"] = 0.85
                    data["pending_approval"] = True

        for tid, data in sorted(turns_data.items()):
            turn = UnifiedTurn(
                backend=self.BACKEND,
                session_id=session_id,
                turn_id=tid or "unknown",
                timestamp=data.get("timestamp") or created_at or datetime.now(timezone.utc),
                user_input=data.get("user_input", ""),
                assistant_response=data.get("assistant_response", ""),
                outcome=data.get("outcome", "unknown"),
                outcome_confidence=data.get("outcome_confidence", 0.0),
                duration_ms=data.get("duration_ms"),
                pending_approval=data.get("pending_approval", False) or (tid in pending_approvals),
                approval_resolved=None if (tid in pending_approvals and data.get("outcome") == "exited")
                else (True if data.get("outcome") == "approved"
                      else False if data.get("outcome") == "rejected" else None),
                raw_meta={"correction": data.get("correction_text", "")},
            )
            session.turns.append(turn)

        if session.turns:
            yield session
