"""Offline evaluation metrics for memory retrieval policy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from agentrl.memory.retrieval import CandidateSession


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval run."""

    n_candidates: int = 0
    n_approved: int = 0
    n_exited: int = 0
    n_corrected: int = 0
    mean_reward_accuracy: float = 0.0
    mean_reward_total: float = 0.0
    approved_recall: float = 0.0  # of all approved in DB, how many retrieved
    approved_precision: float = 0.0  # of retrieved, how many are approved
    exited_contamination: float = 0.0  # % of retrieved that are exited
    coverage: float = 0.0  # 1 if any approved retrieved, else 0
    ndcg: float = 0.0  # reward-weighted ranking quality
    mrr: float = 0.0  # mean reciprocal rank of first approved


@dataclass
class AggregateMetrics:
    """Aggregated metrics over a dataset."""

    n_samples: int = 0
    approved_recall: float = 0.0
    approved_precision: float = 0.0
    exited_contamination: float = 0.0
    mean_reward_accuracy: float = 0.0
    mean_reward_total: float = 0.0
    coverage: float = 0.0
    mean_ndcg: float = 0.0
    mean_mrr: float = 0.0
    per_outcome: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "approved_recall": round(self.approved_recall, 4),
            "approved_precision": round(self.approved_precision, 4),
            "exited_contamination": round(self.exited_contamination, 4),
            "mean_reward_accuracy": round(self.mean_reward_accuracy, 4),
            "mean_reward_total": round(self.mean_reward_total, 4),
            "coverage": round(self.coverage, 4),
            "mean_ndcg": round(self.mean_ndcg, 4),
            "mean_mrr": round(self.mean_mrr, 4),
            "per_outcome": {
                k: {kk: round(vv, 4) for kk, vv in v.items()}
                for k, v in self.per_outcome.items()
            },
        }


class MetricsCollector:
    """Compute retrieval quality metrics from candidates + ground truth."""

    def compute_single(
        self,
        candidates: list[CandidateSession],
        ground_truth_approved_ids: set[str],
        ground_truth_exited_ids: set[str],
    ) -> RetrievalMetrics:
        """Score one retrieval result."""
        if not candidates:
            return RetrievalMetrics()

        n = len(candidates)
        n_approved = sum(1 for c in candidates if c.user_outcome == "approved")
        n_exited = sum(1 for c in candidates if c.user_outcome == "exited")
        n_corrected = sum(1 for c in candidates if c.user_outcome == "corrected")

        rewards_acc = [c.computed_reward_accuracy or 0.0 for c in candidates]
        rewards_total = [
            (c.computed_reward_accuracy or 0.0)
            for c in candidates
        ]  # simplified; could include grounding/temporal

        mean_r_acc = sum(rewards_acc) / n if n else 0.0
        mean_r_total = sum(rewards_total) / n if n else 0.0

        # Approved recall: how many of the known approved set were retrieved?
        retrieved_ids = {c.session_id for c in candidates}
        approved_recall = (
            len(retrieved_ids & ground_truth_approved_ids) / len(ground_truth_approved_ids)
            if ground_truth_approved_ids
            else 0.0
        )
        approved_precision = n_approved / n if n else 0.0

        exited_contamination = n_exited / n if n else 0.0
        coverage = 1.0 if n_approved > 0 else 0.0

        # NDCG: use reward_accuracy as relevance score
        ndcg = self._ndcg(rewards_acc)

        # MRR: rank of first approved
        mrr = 0.0
        for i, c in enumerate(candidates):
            if c.user_outcome == "approved":
                mrr = 1.0 / (i + 1)
                break

        return RetrievalMetrics(
            n_candidates=n,
            n_approved=n_approved,
            n_exited=n_exited,
            n_corrected=n_corrected,
            mean_reward_accuracy=mean_r_acc,
            mean_reward_total=mean_r_total,
            approved_recall=approved_recall,
            approved_precision=approved_precision,
            exited_contamination=exited_contamination,
            coverage=coverage,
            ndcg=ndcg,
            mrr=mrr,
        )

    def aggregate(self, metrics_list: list[RetrievalMetrics]) -> AggregateMetrics:
        """Average metrics over multiple retrieval runs."""
        if not metrics_list:
            return AggregateMetrics()

        n = len(metrics_list)
        agg = AggregateMetrics(n_samples=n)

        agg.approved_recall = sum(m.approved_recall for m in metrics_list) / n
        agg.approved_precision = sum(m.approved_precision for m in metrics_list) / n
        agg.exited_contamination = sum(m.exited_contamination for m in metrics_list) / n
        agg.mean_reward_accuracy = sum(m.mean_reward_accuracy for m in metrics_list) / n
        agg.mean_reward_total = sum(m.mean_reward_total for m in metrics_list) / n
        agg.coverage = sum(m.coverage for m in metrics_list) / n
        agg.mean_ndcg = sum(m.ndcg for m in metrics_list) / n
        agg.mean_mrr = sum(m.mrr for m in metrics_list) / n

        return agg

    def _ndcg(self, relevance_scores: list[float]) -> float:
        """Compute NDCG for a ranked list."""
        if not relevance_scores:
            return 0.0

        def dcg(scores):
            return sum(
                (2 ** s - 1) / math.log2(i + 2)
                for i, s in enumerate(scores)
            )

        ideal = sorted(relevance_scores, reverse=True)
        actual_dcg = dcg(relevance_scores)
        ideal_dcg = dcg(ideal)
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
