"""Tests for offline evaluation framework."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentrl.eval.dataset import EvalDataset
from agentrl.eval.metrics import MetricsCollector, RetrievalMetrics
from agentrl.eval.offline import OfflineEvaluator
from agentrl.memory.retrieval import CandidateSession
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent.parent / "data" / "agentrl.db"


def test_dataset_load():
    if not DB_PATH.exists():
        print("SKIP: no DB found")
        return
    ds = EvalDataset(DB_PATH).load()
    stats = ds.stats()
    assert stats["total"] > 0
    assert stats["train"] > 0
    assert stats["val"] > 0
    assert stats["holdout"] > 0
    print(f"Dataset stats: {stats}")


def test_metrics_collector():
    mc = MetricsCollector()

    # Simulate 3 candidates: approved (reward 1.0), completed (0.5), exited (-1.0)
    candidates = [
        CandidateSession(
            session_id="s1", turn_id="t1", backend="codex",
            query="q1", assistant_response="a1",
            user_outcome="approved", created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            computed_reward_accuracy=1.0,
        ),
        CandidateSession(
            session_id="s2", turn_id="t2", backend="codex",
            query="q2", assistant_response="a2",
            user_outcome="completed", created_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
            computed_reward_accuracy=0.5,
        ),
        CandidateSession(
            session_id="s3", turn_id="t3", backend="codex",
            query="q3", assistant_response="a3",
            user_outcome="exited", created_at=datetime(2025, 1, 3, tzinfo=timezone.utc),
            computed_reward_accuracy=-1.0,
        ),
    ]

    m = mc.compute_single(
        candidates=candidates,
        ground_truth_approved_ids={"s1", "s4"},
        ground_truth_exited_ids={"s3"},
    )
    assert m.n_candidates == 3
    assert m.n_approved == 1
    assert m.n_exited == 1
    assert m.approved_recall == 0.5  # s1 retrieved, s4 not
    assert m.exited_contamination == 1 / 3
    assert m.coverage == 1.0
    assert m.mean_reward_accuracy == (1.0 + 0.5 - 1.0) / 3
    print(f"Single metrics: approved_recall={m.approved_recall:.2f}, "
          f"exited_contam={m.exited_contamination:.2f}, ndcg={m.ndcg:.3f}")


def test_offline_evaluator_baseline():
    if not DB_PATH.exists():
        print("SKIP: no DB found")
        return

    ds = EvalDataset(DB_PATH).load()
    evaluator = OfflineEvaluator(DB_PATH)

    # Run on a small subset for speed
    subset = ds.val[:30]
    metrics = evaluator.run_baseline(subset, max_candidates=10, outcome_bias="any")

    evaluator.print_report(metrics, title="Baseline (any, k=10)")
    assert metrics.n_samples == len(subset)
    assert 0.0 <= metrics.approved_recall <= 1.0
    assert 0.0 <= metrics.exited_contamination <= 1.0


def test_offline_evaluator_approved_bias():
    if not DB_PATH.exists():
        print("SKIP: no DB found")
        return

    ds = EvalDataset(DB_PATH).load()
    evaluator = OfflineEvaluator(DB_PATH)
    subset = ds.val[:30]

    metrics_any = evaluator.run_baseline(subset, max_candidates=10, outcome_bias="any")
    metrics_approved = evaluator.run_baseline(subset, max_candidates=10, outcome_bias="approved_only")

    print(f"\nComparison:")
    print(f"  ANY:      recall={metrics_any.approved_recall:.3f}, "
          f"contam={metrics_any.exited_contamination:.3f}")
    print(f"  APPROVED: recall={metrics_approved.approved_recall:.3f}, "
          f"contam={metrics_approved.exited_contamination:.3f}")

    # Approved-only should have higher precision and lower contamination
    # (recall may be lower because search space is smaller)
    assert metrics_approved.approved_precision >= metrics_any.approved_precision
    assert metrics_approved.exited_contamination <= metrics_any.exited_contamination


if __name__ == "__main__":
    test_dataset_load()
    test_metrics_collector()
    test_offline_evaluator_baseline()
    test_offline_evaluator_approved_bias()
    print("\nAll eval tests passed.")
