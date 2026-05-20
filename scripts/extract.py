#!/usr/bin/env python3
"""Extract unified session data and export to JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentrl.extractors import UnifiedExtractor
from agentrl.utils import has_temporal_keywords


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract unified session data")
    parser.add_argument("--out", default="data/unified_sessions.jsonl", help="output JSONL path")
    parser.add_argument("--backend", default="all", help="backend filter")
    parser.add_argument("--stats", action="store_true", help="print stats only")
    parser.add_argument("--temporal-only", action="store_true")
    args = parser.parse_args()

    extractor = UnifiedExtractor()
    stats: dict[str, dict] = {}
    total = 0

    for turn in extractor.iter_turns():
        if args.backend != "all" and turn.backend != args.backend:
            continue
        if args.temporal_only and not has_temporal_keywords(turn.user_input):
            continue

        b = turn.backend
        if b not in stats:
            stats[b] = {"sessions": set(), "turns": 0, "outcomes": {}, "temporal": 0}
        stats[b]["sessions"].add(turn.session_id)
        stats[b]["turns"] += 1
        stats[b]["outcomes"][turn.outcome] = stats[b]["outcomes"].get(turn.outcome, 0) + 1
        if has_temporal_keywords(turn.user_input):
            stats[b]["temporal"] += 1
        total += 1

    if args.stats:
        print("=" * 60)
        for b, s in sorted(stats.items()):
            print(f"\nBackend: {b}")
            print(f"  Sessions: {len(s['sessions'])}")
            print(f"  Turns:    {s['turns']}")
            print(f"  Temporal: {s['temporal']}")
            print("  Outcomes:")
            for o, c in sorted(s["outcomes"].items(), key=lambda x: -x[1]):
                print(f"    {o}: {c}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for turn in extractor.iter_turns():
            if args.backend != "all" and turn.backend != args.backend:
                continue
            if args.temporal_only and not has_temporal_keywords(turn.user_input):
                continue
            f.write(
                json.dumps(
                    {
                        "backend": turn.backend,
                        "session_id": turn.session_id,
                        "turn_id": turn.turn_id,
                        "timestamp": turn.timestamp.isoformat() if turn.timestamp else None,
                        "user_input": turn.user_input,
                        "assistant_response": turn.assistant_response,
                        "tool_calls": turn.tool_calls,
                        "files_read": turn.files_read,
                        "files_written": turn.files_written,
                        "outcome": turn.outcome,
                        "outcome_confidence": turn.outcome_confidence,
                        "pending_approval": turn.pending_approval,
                        "approval_resolved": turn.approval_resolved,
                        "duration_ms": turn.duration_ms,
                        "raw_meta": turn.raw_meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {total} unified turns to {out_path}")
    print("\nStats:")
    for b, s in sorted(stats.items()):
        print(f"  {b}: {len(s['sessions'])} sessions, {s['turns']} turns, {s['temporal']} temporal")


if __name__ == "__main__":
    main()
