#!/usr/bin/env python3
"""
agentRL Daemon — Background learner for Hermes sessions.

Scans ~/.hermes/sessions/ for new/modified session files and
automatically feeds them into agentRL for online learning.

Usage:
    python scripts/agentrl_daemon.py
    # or as systemd service
"""

import os
import sys
import time
import json
import glob
import hashlib
from pathlib import Path
from datetime import datetime, timezone

# Ensure agentRL importable
_AGENTRL_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _AGENTRL_SRC not in sys.path:
    sys.path.insert(0, _AGENTRL_SRC)

from agentrl.hermes_plugin import AgentRLHermesHook

SESSIONS_DIR = Path.home() / ".hermes" / "sessions"
DATA_DIR = Path(os.environ.get("AGENTRL_DATA_DIR", "./data"))
PROCESSED_LOG = DATA_DIR / "processed_sessions.json"
SCAN_INTERVAL = 30  # seconds


def get_session_id(path: str) -> str:
    """Stable session ID from file path."""
    return Path(path).stem


def load_processed() -> set[str]:
    """Load set of already-processed session IDs."""
    if not PROCESSED_LOG.exists():
        return set()
    with open(PROCESSED_LOG) as f:
        data = json.load(f)
    return set(data.get("processed", []))


def save_processed(processed: set[str]) -> None:
    """Save processed session IDs."""
    PROCESSED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_LOG, "w") as f:
        json.dump({
            "processed": sorted(processed),
            "last_run": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)


def parse_session_file(path: str) -> list[dict]:
    """Parse a Hermes session file (JSON or JSONL)."""
    if path.endswith(".jsonl"):
        messages = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return messages
    else:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "messages" in data:
            return data["messages"]
        return [data]


def main():
    hook = AgentRLHermesHook()
    processed = load_processed()

    print(f"[agentRL-daemon] Started. Watching {SESSIONS_DIR}")
    print(f"[agentRL-daemon] Already processed: {len(processed)} sessions")

    while True:
        files = sorted(
            glob.glob(str(SESSIONS_DIR / "*.jsonl")) + glob.glob(str(SESSIONS_DIR / "*.json")),
            key=os.path.getmtime,
            reverse=True,
        )

        new_count = 0
        for path in files:
            sid = get_session_id(path)
            if sid in processed:
                continue

            try:
                messages = parse_session_file(path)
                if len(messages) < 3:
                    processed.add(sid)
                    continue

                result = hook.on_session_end(session_id=sid, messages=messages, backend="hermes")
                status = result.get("status", "unknown")
                if status == "learned":
                    new_count += 1
                    print(f"[agentRL-daemon] Learned from {sid}: {result.get('goal', '')[:50]}... "
                          f"(outcome={result.get('final_outcome')}, reward={result.get('trajectory_reward', 0):.2f})")
                else:
                    print(f"[agentRL-daemon] Skipped {sid}: {result.get('reason', status)}")

                processed.add(sid)

            except Exception as e:
                print(f"[agentRL-daemon] ERROR processing {sid}: {e}")
                processed.add(sid)  # Don't retry forever

        if new_count > 0:
            save_processed(processed)
            print(f"[agentRL-daemon] Batch complete. Learned from {new_count} new sessions. "
                  f"Total processed: {len(processed)}")

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
