#!/usr/bin/env python3
"""Run offline evaluation with optional LLM Judge for precise scoring.

Usage:
    # Baseline offline metrics only (fast, $0)
    python scripts/eval_with_judge.py --db data/agentrl.db

    # With LLM Judge on 10 samples (~$0.10-0.50)
    python scripts/eval_with_judge.py --db data/agentrl.db --llm-judge --judge-n 10

    # Compare two outcome bias strategies with LLM Judge
    python scripts/eval_with_judge.py --db data/agentrl.db --llm-judge --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentrl.eval.dataset import EvalDataset
from agentrl.eval.offline import OfflineEvaluator
from agentrl.llm_judge import LLMJudge
from agentrl.memory.retrieval import CoarseFilter


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with optional LLM Judge")
    parser.add_argument("--db", default="data/agentrl.db", help="SQLite DB path")
    parser.add_argument("--subset", type=int, default=30, help="Number of val samples to evaluate")
    parser.add_argument("--llm-judge", action="store_true", help="Enable LLM Judge scoring")
    parser.add_argument("--judge-n", type=int, default=10, help="Max samples for LLM Judge")
    parser.add_argument("--compare", action="store_true", help="Compare multiple strategies")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}")
        sys.exit(1)

    print("Loading dataset...")
    ds = EvalDataset(db_path).load()
    stats = ds.stats()
    print(f"Dataset: total={stats['total']}, train={stats['train']}, val={stats['val']}, holdout={stats['holdout']}")

    evaluator = OfflineEvaluator(db_path)
    val_subset = ds.val[: args.subset]

    if args.compare:
        print("\n" + "=" * 60)
        print("Comparing strategies")
        print("=" * 60)

        cf = CoarseFilter(db_path)

        def retrieve_any(s):
            return cf.filter(query=s.query, max_candidates=10, outcome_bias="any", exclude_session_ids={s.session_id})

        def retrieve_approved(s):
            return cf.filter(query=s.query, max_candidates=10, outcome_bias="approved_only", exclude_session_ids={s.session_id})

        def retrieve_avoid_exited(s):
            return cf.filter(query=s.query, max_candidates=10, outcome_bias="avoid_exited", exclude_session_ids={s.session_id})

        strategies = {
            "any": retrieve_any,
            "approved_only": retrieve_approved,
            "avoid_exited": retrieve_avoid_exited,
        }

        results = evaluator.compare_strategies(val_subset, strategies)
        for name, metrics in results.items():
            evaluator.print_report(metrics, title=f"Strategy: {name}")

        if args.llm_judge:
            print("\n" + "=" * 60)
            print("LLM Judge Comparison")
            print("=" * 60)
            judge = LLMJudge()
            for name, fn in strategies.items():
                print(f"\n--- LLM Judge: {name} ---")
                result = evaluator.run_with_llm_judge(
                    val_subset, fn, judge, max_llm_evals=args.judge_n
                )
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"  LLM Overall:     {result['overall']:.3f}")
                    print(f"  LLM Relevance:   {result['relevance']:.3f}")
                    print(f"  LLM Quality:     {result['quality']:.3f}")
                    print(f"  Evaluated:       {result['n_evaluated']} samples")

    else:
        # Single baseline run
        metrics = evaluator.run_baseline(val_subset, max_candidates=10, outcome_bias="any")
        evaluator.print_report(metrics, title="Baseline (any, k=10)")

        if args.llm_judge:
            print("\n" + "=" * 60)
            print("LLM Judge Scoring")
            print("=" * 60)
            judge = LLMJudge()
            result = evaluator.run_with_llm_judge(
                val_subset,
                lambda s: CoarseFilter(db_path).filter(
                    query=s.query, max_candidates=10, outcome_bias="any", exclude_session_ids={s.session_id}
                ),
                judge,
                max_llm_evals=args.judge_n,
            )
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"  LLM Overall:     {result['overall']:.3f}")
                print(f"  LLM Relevance:   {result['relevance']:.3f}")
                print(f"  LLM Quality:     {result['quality']:.3f}")
                print(f"  Evaluated:       {result['n_evaluated']} samples")


if __name__ == "__main__":
    main()
