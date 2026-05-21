from __future__ import annotations

import glob as glob_mod
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from agentrl.extractors.trajectory import TrajectoryBuilder
from agentrl.models import UnifiedSession, UnifiedTurn
from agentrl.utils import detect_outcome_from_correction, extract_files_from_tools, parse_iso


class HermesParser:
    GLOB = os.path.expanduser("~/.hermes/sessions/*.jsonl")
    BACKEND = "hermes"

    def __init__(self) -> None:
        self.traj_builder = TrajectoryBuilder()

    def iter_sessions(self) -> Iterator[UnifiedSession]:
        for p in sorted(glob_mod.glob(self.GLOB)):
            yield from self._parse_file(Path(p))

    def iter_trajectories(self) -> Iterator[dict[str, Any]]:
        """Yield fine-grained TaskTrajectory dicts (new API)."""
        for p in sorted(glob_mod.glob(self.GLOB)):
            session_id = Path(p).stem
            messages = self._read_jsonl(p)
            if messages:
                traj = self.traj_builder.build(session_id, self.BACKEND, messages)
                yield traj.__dict__  # naive dict export; caller can use dataclass

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        messages: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return messages

    def _parse_file(self, path: Path) -> Iterator[UnifiedSession]:
        messages = self._read_jsonl(path)
        if not messages:
            return

        session_id = path.stem
        created_at = None
        for m in messages:
            if m.get("role") == "session_meta":
                created_at = parse_iso(m.get("timestamp"))
                break
            if m.get("timestamp"):
                created_at = parse_iso(m.get("timestamp"))
                break

        session = UnifiedSession(
            backend=self.BACKEND,
            session_id=session_id,
            created_at=created_at,
            raw_meta={"file": str(path)},
        )

        turns_data: list[dict] = []
        current_turn: dict[str, Any] = {"outcome": "unknown", "outcome_confidence": 0.0}

        for msg in messages:
            role = msg.get("role", "")
            ts = parse_iso(msg.get("timestamp"))

            if role == "user":
                if current_turn.get("user_input"):
                    turns_data.append(current_turn)
                current_turn = {
                    "user_input": msg.get("content", ""),
                    "timestamp": ts,
                    "assistant_response": "",
                    "tool_calls": [],
                    "outcome": "unknown",
                    "outcome_confidence": 0.0,
                }
                corr, conf = detect_outcome_from_correction(msg.get("content", ""))
                if corr:
                    current_turn["is_correction"] = True
                    current_turn["correction_text"] = msg.get("content", "")

            elif role == "assistant":
                current_turn["assistant_response"] = msg.get("content", "")
                for tc in msg.get("tool_calls", []):
                    current_turn["tool_calls"].append(tc)
                finish = msg.get("finish_reason")
                if finish == "stop" and msg.get("content"):
                    current_turn["outcome"] = "approved"
                    current_turn["outcome_confidence"] = 0.7
                elif finish == "length":
                    current_turn["outcome"] = "exited"
                    current_turn["outcome_confidence"] = 0.6

        if current_turn.get("user_input"):
            turns_data.append(current_turn)

        for idx, data in enumerate(turns_data):
            reads, writes = extract_files_from_tools(data.get("tool_calls", []))
            turn = UnifiedTurn(
                backend=self.BACKEND,
                session_id=session_id,
                turn_id=f"turn_{idx}",
                timestamp=data.get("timestamp") or created_at or datetime.now(timezone.utc),
                user_input=data.get("user_input", ""),
                assistant_response=data.get("assistant_response", ""),
                tool_calls=data.get("tool_calls", []),
                files_read=reads,
                files_written=writes,
                outcome=data.get("outcome", "unknown"),
                outcome_confidence=data.get("outcome_confidence", 0.0),
                raw_meta={
                    "is_correction": data.get("is_correction", False),
                    "correction_text": data.get("correction_text", ""),
                },
            )
            session.turns.append(turn)

        if session.turns:
            yield session
