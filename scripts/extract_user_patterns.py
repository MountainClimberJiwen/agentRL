#!/usr/bin/env python3
"""Extract user behavioral patterns from session history and save as memory.

Usage:
    python scripts/extract_user_patterns.py --db data/agentrl.db
    python scripts/extract_user_patterns.py --db data/agentrl.db --show-prompt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentrl.patterns.miner import PatternMiner
from agentrl.user_memory.store import UserMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine user patterns from session history")
    parser.add_argument("--db", default="data/agentrl.db", help="SQLite DB path")
    parser.add_argument("--show-prompt", action="store_true", help="Show the generated prompt block")
    parser.add_argument("--show-raw", action="store_true", help="Show raw JSON profile")
    parser.add_argument("--save", default="data/user_memory.json", help="Save path for user memory")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}")
        sys.exit(1)

    print("Mining user patterns from session history...")
    print(f"DB: {db_path}")
    print()

    miner = PatternMiner(db_path)
    profile = miner.mine()

    # Show stats
    print(f"Sessions analyzed: {profile.total_sessions}")
    print(f"Turns analyzed: {profile.total_turns}")
    print()

    # Show coding preferences
    if profile.coding_preferences:
        print("=" * 60)
        print("CODING PREFERENCES")
        print("=" * 60)
        for k, v in profile.coding_preferences.items():
            status = "✓" if v else "✗"
            print(f"  {status} {k.replace('_', ' ').title()}")
        print()

    # Show workflows
    if profile.preferred_workflows:
        print("=" * 60)
        print("PREFERRED WORKFLOWS")
        print("=" * 60)
        for wf in profile.preferred_workflows:
            print(f"  • {wf['pattern']}")
            print(f"    Used {wf.get('frequency', 0)}x, {wf.get('success_rate', 0)*100:.0f}% success rate")
        print()

    # Show corrections
    if profile.correction_patterns:
        print("=" * 60)
        print("COMMON CORRECTIONS (AVOID THESE)")
        print("=" * 60)
        for cp in profile.correction_patterns:
            print(f"  ⚠ {cp['pattern']}")
            print(f"    → {cp.get('suggestion', '')}")
        print()

    # Show file clusters
    if profile.file_clusters:
        print("=" * 60)
        print("FREQUENTLY ACCESSED FILE GROUPS")
        print("=" * 60)
        for fc in profile.file_clusters[:5]:
            files = ", ".join(fc.get("files", [])[:4])
            proj = fc.get("project", "")
            print(f"  [{proj}] {files}")
        print()

    # Show project knowledge
    if profile.project_knowledge:
        print("=" * 60)
        print("PROJECT KNOWLEDGE")
        print("=" * 60)
        for pk in profile.project_knowledge[:5]:
            print(f"  📁 {pk['project']} ({pk.get('tech_stack', 'unknown')})")
            print(f"     Sessions: {pk.get('total_sessions', 0)}, Success: {pk.get('success_rate', 0)*100:.0f}%")
            if pk.get("key_files"):
                print(f"     Key files: {', '.join(pk['key_files'][:3])}")
        print()

    # Show intent patterns
    if profile.intent_patterns:
        print("=" * 60)
        print("USER INTENT PATTERNS")
        print("=" * 60)
        for intent, stats in sorted(profile.intent_patterns.items(), key=lambda x: -x[1].get("count", 0)):
            print(f"  {intent:12s}  {stats['count']:3d} sessions  {stats.get('approved_rate', 0)*100:5.0f}% success")
        print()

    # Show failure patterns
    if profile.failure_patterns:
        print("=" * 60)
        print("FAILURE PATTERNS (DEEP ANALYSIS)")
        print("=" * 60)
        for fp in profile.failure_patterns:
            print(f"  ⚠ {fp['pattern']} ({fp['count']}x)")
            print(f"    Insight: {fp.get('insight', '')}")
            print(f"    → {fp.get('suggestion', '')}")
        print()

    # Show success sequences
    if profile.success_sequences:
        print("=" * 60)
        print("SUCCESS SEQUENCES")
        print("=" * 60)
        for ss in profile.success_sequences:
            print(f"  ✓ {ss['sequence']} ({ss['frequency']}x)")
            if ss.get("description"):
                print(f"    {ss['description']}")
        print()

    # Show tool preferences
    if profile.tool_preferences:
        print("=" * 60)
        print("TOOL PREFERENCES")
        print("=" * 60)
        tp = profile.tool_preferences
        if tp.get("most_used_tools"):
            print("  Most used:")
            for tool, count in list(tp["most_used_tools"].items())[:5]:
                print(f"    • {tool}: {count}x")
        if tp.get("tool_success_rates"):
            print("  Best success rates:")
            for tool, rate in list(tp["tool_success_rates"].items())[:5]:
                print(f"    • {tool}: {rate*100:.0f}%")
        print()

    # Save
    store = UserMemoryStore(args.save)
    store.update_from_miner(profile)
    print(f"Saved user memory to: {args.save}")

    if args.show_raw:
        print()
        print("=" * 60)
        print("RAW PROFILE JSON")
        print("=" * 60)
        print(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2))

    if args.show_prompt:
        print()
        print("=" * 60)
        print("GENERATED PROMPT BLOCK (for injection into agent)")
        print("=" * 60)
        print(profile.to_prompt_block())


if __name__ == "__main__":
    main()
