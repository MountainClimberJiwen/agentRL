from __future__ import annotations

import glob as glob_mod
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from agentrl.models import UnifiedSession, UnifiedTurn
from agentrl.utils import detect_outcome_from_correction, parse_iso


class CCConnectParser:
    GLOB = os.path.expanduser("~/.cc-connect/sessions/*.json")
    BACKEND = "cc-connect"

    def iter_sessions(self) -> Iterator[UnifiedSession]:
        for p in sorted(glob_mod.glob(self.GLOB)):
            yield from self._parse_file(Path(p))

    def _parse_file(self, path: Path) -> Iterator[UnifiedSession]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sessions = data.get("sessions", {})
        for sess_key, sess in sessions.items():
            session_id = sess.get("id", sess_key)
            created_at = parse_iso(sess.get("created_at"))
            history = sess.get("history", [])

            session = UnifiedSession(
                backend=self.BACKEND,
                session_id=session_id,
                created_at=created_at,
                raw_meta={"file": str(path), "name": sess.get("name", "")},
            )

            turns_data: list[dict] = []
            current_turn: dict[str, Any] = {"outcome": "unknown", "outcome_confidence": 0.0}

            for msg in history:
                role = msg.get("role", "")
                ts = parse_iso(msg.get("timestamp"))
                content = msg.get("content", "")

                if role == "user":
                    if current_turn.get("user_input"):
                        turns_data.append(current_turn)
                    current_turn = {
                        "user_input": content,
                        "timestamp": ts,
                        "assistant_response": "",
                        "outcome": "unknown",
                        "outcome_confidence": 0.0,
                    }
                    corr, conf = detect_outcome_from_correction(content)
                    if corr:
                        current_turn["is_correction"] = True
                        current_turn["correction_text"] = content

                elif role == "assistant":
                    current_turn["assistant_response"] = content
                    if content:
                        current_turn["outcome"] = "approved"
                        current_turn["outcome_confidence"] = 0.6

            if current_turn.get("user_input"):
                turns_data.append(current_turn)

            for idx, data in enumerate(turns_data):
                turn = UnifiedTurn(
                    backend=self.BACKEND,
                    session_id=session_id,
                    turn_id=f"turn_{idx}",
                    timestamp=data.get("timestamp") or created_at or datetime.now(timezone.utc),
                    user_input=data.get("user_input", ""),
                    assistant_response=data.get("assistant_response", ""),
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
