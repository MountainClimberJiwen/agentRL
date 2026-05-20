"""Offline evaluator: run retrieval on historical data and score with known rewards.

No LLM calls required for baseline metrics. Optional LLM Judge for precise scoring.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable

from agentrl.eval.dataset import EvalDataset, EvalSample
from agentrl.eval.metrics import AggregateMetrics, MetricsCollector, RetrievalMetrics
from agentrl.llm_judge import LLMJudge
from agentrl.memory.retrieval import CoarseFilter


class OfflineEvaluator:
    """Evaluate a retrieval strategy against historical session data.

    Usage:
        ds = EvalDataset("data/agentrl.db").load()
        eval = OfflineEvaluator("data/agentrl.db")
        metrics = eval.run(ds.val, retrieval_fn=my_filter)
        print(metrics.to_dict())
    """

    def __init__(self, db_path: str | Path):
        self.db_path = db_path
        self.metrics = MetricsCollector()
        self._ground_truth = self._build_ground_truth()

    def _build_ground_truth(self) -> dict[str, set[str]]:
        """Pre-compute sets of approved/exited session IDs for fast lookup."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            "SELECT session_id, user_outcome FROM memory_feedback"
        )
        approved = set()
        exited = set()
        for sid, outcome in cur.fetchall():
            if outcome == "approved":
                approved.add(sid)
            elif outcome == "exited":
                exited.add(sid)
        conn.close()
        return {"approved": approved, "exited": exited}

    def run(
        self,
        samples: list[EvalSample],
        retrieval_fn: Callable[[EvalSample], list],
        progress_every: int = 50,
    ) -> AggregateMetrics:
        """Run retrieval on each sample and aggregate metrics.

        Args:
            samples: Evaluation samples (from EvalDataset)
            retrieval_fn: A callable that takes an EvalSample and returns
                a list of CandidateSession objects.
            progress_every: Print progress every N samples.
        """
        approved_set = self._ground_truth["approved"]
        exited_set = self._ground_truth["exited"]

        per_sample: list[RetrievalMetrics] = []

        for i, sample in enumerate(samples):
            candidates = retrieval_fn(sample)
            m = self.metrics.compute_single(
                candidates=candidates,
                ground_truth_approved_ids=approved_set,
                ground_truth_exited_ids=exited_set,
            )
            per_sample.append(m)

            if progress_every and (i + 1) % progress_every == 0:
                print(f"  Evaluated {i + 1}/{len(samples)} samples...")

        agg = self.metrics.aggregate(per_sample)

        # Per-outcome breakdown
        from collections import defaultdict
        outcome_buckets: dict[str, list[RetrievalMetrics]] = defaultdict(list)
        for sample, m in zip(samples, per_sample):
            outcome_buckets[sample.user_outcome].append(m)

        agg.per_outcome = {}
        for outcome, bucket in outcome_buckets.items():
            sub = self.metrics.aggregate(bucket)
            agg.per_outcome[outcome] = {
                "mean_reward_accuracy": sub.mean_reward_accuracy,
                "mean_reward_total": sub.mean_reward_total,
                "coverage": sub.coverage,
                "mean_ndcg": sub.mean_ndcg,
                "mean_mrr": sub.mean_mrr,
                "n": sub.n_samples,
            }

        return agg

    def run_with_llm_judge(
        self,
        samples: list[EvalSample],
        retrieval_fn: Callable[[EvalSample], list],
        judge: LLMJudge,
        max_llm_evals: int = 20,
    ) -> dict[str, Any]:
        """Run retrieval then use LLM Judge for precise scoring.

        This is expensive (~$0.01-0.05 per sample) so cap at max_llm_evals.
        """
        from collections import defaultdict

        by_outcome: dict[str, list[tuple[int, EvalSample]]] = defaultdict(list)
        for i, s in enumerate(samples):
            by_outcome[s.user_outcome].append((i, s))

        # Take up to max_llm_evals evenly across outcomes
        selected = []
        per_outcome = max(1, max_llm_evals // max(len(by_outcome), 1))
        for outcome, items in by_outcome.items():
            selected.extend(items[:per_outcome])
        selected = selected[:max_llm_evals]

        judge_inputs = []
        for idx, sample in selected:
            candidates = retrieval_fn(sample)
            candidate_dicts = [c.to_dict() for c in candidates[:5]]
            judge_inputs.append({
                "query": sample.query,
                "candidates": candidate_dicts,
                "outcome": sample.user_outcome,
            })

        print(f"Running LLM Judge on {len(judge_inputs)} samples...")
        judge_results = judge.batch_evaluate(judge_inputs)

        # Aggregate LLM scores
        scores = [r["score"] for r in judge_results if r["score"] is not None]
        if not scores:
            return {"error": "No successful LLM evaluations", "details": judge_results}

        overall = sum(s.get("overall", 0) for s in scores) / len(scores)
        relevance = sum(s.get("relevance", 0) for s in scores) / len(scores)
        quality = sum(s.get("quality", 0) for s in scores) / len(scores)

        return {
            "n_evaluated": len(scores),
            "overall": round(overall, 4),
            "relevance": round(relevance, 4),
            "quality": round(quality, 4),
            "raw_results": judge_results,
        }

    def run_baseline(
        self,
        samples: list[EvalSample],
        max_candidates: int = 20,
        outcome_bias: str = "any",
    ) -> AggregateMetrics:
        """Run with the default CoarseFilter as baseline strategy."""
        cf = CoarseFilter(self.db_path)

        def _retrieve(sample: EvalSample):
            return cf.filter(
                query=sample.query,
                max_candidates=max_candidates,
                outcome_bias=outcome_bias,
                exclude_session_ids={sample.session_id},
            )

        print(f"Running baseline (max_candidates={max_candidates}, bias={outcome_bias})...")
        return self.run(samples, _retrieve)

    def compare_strategies(
        self,
        samples: list[EvalSample],
        strategies: dict[str, Callable[[EvalSample], list]],
    ) -> dict[str, AggregateMetrics]:
        """Compare multiple retrieval strategies side-by-side."""
        results = {}
        for name, fn in strategies.items():
            print(f"\n=== Strategy: {name} ===")
            results[name] = self.run(samples, fn)
        return results

    def print_report(self, metrics: AggregateMetrics, title: str = "Results") -> None:
        """Pretty-print evaluation results."""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(f"  Samples evaluated: {metrics.n_samples}")
        print(f"  Approved Recall:   {metrics.approved_recall:.3f}")
        print(f"  Approved Precision:{metrics.approved_precision:.3f}")
        print(f"  Exited Contam:     {metrics.exited_contamination:.3f}")
        print(f"  Mean Reward (acc): {metrics.mean_reward_accuracy:.3f}")
        print(f"  Mean Reward (tot): {metrics.mean_reward_total:.3f}")
        print(f"  Coverage:          {metrics.coverage:.3f}")
        print(f"  Mean NDCG:         {metrics.mean_ndcg:.3f}")
        print(f"  Mean MRR:          {metrics.mean_mrr:.3f}")

        if metrics.per_outcome:
            print(f"\n  Per-outcome breakdown:")
            for outcome, vals in sorted(metrics.per_outcome.items()):
                print(
                    f"    {outcome:12s}  n={vals['n']:4d}  "
                    f"reward={vals['mean_reward_accuracy']:+.3f}  "
                    f"cov={vals['coverage']:.3f}  "
                    f"ndcg={vals['mean_ndcg']:.3f}"
                )
        print(f"{'='*60}")
