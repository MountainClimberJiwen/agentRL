#!/usr/bin/env python3
"""Ingest extracted session turns into agentRL SQLite database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentrl.db import init_db, stats, upsert_turn
from agentrl.extractors import UnifiedExtractor
from agentrl.rewards import compute_all_rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest session data into agentRL db")
    parser.add_argument("--db", help="SQLite db path (default: data/agentrl.db)")
    parser.add_argument("--backend", default="all", help="Filter backend")
    parser.add_argument("--stats-only", action="store_true", help="Only print stats")
    args = parser.parse_args()

    conn = init_db(args.db)

    if args.stats_only:
        stats(conn)
        conn.close()
        return

    extractor = UnifiedExtractor()
    inserted = updated = 0

    for turn in extractor.iter_turns():
        if args.backend != "all" and turn.backend != args.backend:
            continue
        r_acc, r_ground, r_temp = compute_all_rewards(turn)
        if upsert_turn(conn, turn, r_acc, r_ground, r_temp):
            inserted += 1
        else:
            updated += 1

    print(f"Inserted: {inserted}, Updated: {updated}")
    stats(conn)
    conn.close()


if __name__ == "__main__":
    main()
