from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from agentrl.models import UnifiedSession, UnifiedTurn
from agentrl.utils import extract_files_from_tools, parse_iso


class KimiParser:
    BASE_DIR = os.path.expanduser("~/.kimi/sessions")
    BACKEND = "kimi"

    def iter_sessions(self) -> Iterator[UnifiedSession]:
        base = Path(self.BASE_DIR)
        if not base.exists():
            return
        for workspace_dir in sorted(base.iterdir()):
            if not workspace_dir.is_dir():
                continue
            for sess_dir in sorted(workspace_dir.iterdir()):
                if not sess_dir.is_dir():
                    continue
                yield from self._parse_session_dir(sess_dir)

    def _parse_session_dir(self, sess_dir: Path) -> Iterator[UnifiedSession]:
        state_path = sess_dir / "state.json"
        wire_path = sess_dir / "wire.jsonl"

        state: dict = {}
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except Exception:
                pass

        session_id = sess_dir.name
        created_at = None

        turns_data: dict[str, dict[str, Any]] = {}
        current_turn_id = 0
        current_turn: dict[str, Any] = {}
        tool_calls: list[dict] = []
        has_turn_end = False

        if wire_path.exists():
            with open(wire_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg = obj.get("message", {})
                    mtype = msg.get("type")
                    payload = msg.get("payload", {})
                    ts = parse_iso(obj.get("timestamp"))

                    if mtype == "TurnBegin":
                        if current_turn:
                            self._finalize(current_turn, tool_calls, has_turn_end)
                            turns_data[str(current_turn_id)] = current_turn
                        current_turn_id += 1
                        current_turn = {
                            "user_input": payload.get("user_input", ""),
                            "timestamp": ts,
                            "tool_calls": [],
                            "outcome": "unknown",
                            "outcome_confidence": 0.0,
                        }
                        tool_calls = []
                        has_turn_end = False

                    elif mtype == "ToolCall":
                        tc = payload if "function" in payload else payload.get("tool_call", {})
                        if tc:
                            tool_calls.append(tc)

                    elif mtype == "ContentPart":
                        part = payload
                        if part.get("type") == "text":
                            current_turn.setdefault("assistant_response", "")
                            current_turn["assistant_response"] += part.get("text", "")

                    elif mtype == "StepInterrupted":
                        current_turn["outcome"] = "exited"
                        current_turn["outcome_confidence"] = 0.9
                        has_turn_end = True

                    elif mtype == "TurnEnd":
                        has_turn_end = True
                        if current_turn.get("outcome") == "unknown":
                            if current_turn.get("assistant_response"):
                                current_turn["outcome"] = "approved"
                                current_turn["outcome_confidence"] = 0.7
                            else:
                                current_turn["outcome"] = "exited"
                                current_turn["outcome_confidence"] = 0.5

            if current_turn:
                self._finalize(current_turn, tool_calls, has_turn_end)
                turns_data[str(current_turn_id)] = current_turn

        session = UnifiedSession(
            backend=self.BACKEND,
            session_id=session_id,
            created_at=created_at,
            raw_meta={"state": state, "dir": str(sess_dir)},
        )

        for tid, data in sorted(turns_data.items(), key=lambda x: int(x[0])):
            reads, writes = extract_files_from_tools(data.get("tool_calls", []))
            turn = UnifiedTurn(
                backend=self.BACKEND,
                session_id=session_id,
                turn_id=tid,
                timestamp=data.get("timestamp") or datetime.now(timezone.utc),
                user_input=data.get("user_input", ""),
                assistant_response=data.get("assistant_response", ""),
                tool_calls=data.get("tool_calls", []),
                files_read=reads,
                files_written=writes,
                outcome=data.get("outcome", "unknown"),
                outcome_confidence=data.get("outcome_confidence", 0.0),
                pending_approval=state.get("approval", {}).get("yolo", False) is False,
                raw_meta={"has_turn_end": data.get("has_turn_end", False)},
            )
            session.turns.append(turn)

        if session.turns:
            yield session

    @staticmethod
    def _finalize(turn: dict, tool_calls: list[dict], has_turn_end: bool) -> None:
        turn["tool_calls"] = tool_calls[:]
        reads, writes = extract_files_from_tools(tool_calls)
        turn["files_read"] = reads
        turn["files_written"] = writes
        turn["has_turn_end"] = has_turn_end
        if not has_turn_end and turn.get("outcome") == "unknown":
            turn["outcome"] = "exited"
            turn["outcome_confidence"] = 0.6
