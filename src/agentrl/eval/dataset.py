"""Build train/val/holdout splits from memory_feedback DB."""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EvalSample:
    """A single sample for offline evaluation."""

    session_id: str
    turn_id: str
    backend: str
    query: str
    user_outcome: str
    reward_accuracy: float
    reward_grounding: float
    reward_temporal: float
    created_at: datetime
    assistant_response: str = ""
    files_read: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)

    def total_reward(self) -> float:
        return self.reward_accuracy + self.reward_grounding + self.reward_temporal

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "backend": self.backend,
            "query": self.query,
            "user_outcome": self.user_outcome,
            "reward_accuracy": self.reward_accuracy,
            "reward_grounding": self.reward_grounding,
            "reward_temporal": self.reward_temporal,
            "created_at": self.created_at.isoformat(),
        }


class EvalDataset:
    """Load and split memory_feedback into train/val/holdout."""

    def __init__(
        self,
        db_path: str | Path,
        val_ratio: float = 0.15,
        holdout_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.db_path = db_path
        self.val_ratio = val_ratio
        self.holdout_ratio = holdout_ratio
        self.seed = seed
        self._all: list[EvalSample] = []
        self._train: list[EvalSample] | None = None
        self._val: list[EvalSample] | None = None
        self._holdout: list[EvalSample] | None = None

    def load(self, min_query_length: int = 5) -> "EvalDataset":
        """Load samples from DB."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            """
            SELECT session_id, turn_id, backend, query, user_outcome,
                   computed_reward_accuracy, computed_reward_grounding,
                   computed_reward_temporal, created_at, assistant_response,
                   files_read, files_written, tool_calls
            FROM memory_feedback
            WHERE query IS NOT NULL AND LENGTH(query) >= ?
            ORDER BY created_at
            """,
            (min_query_length,),
        )

        import json

        for row in cur.fetchall():
            ts = row["created_at"]
            try:
                ts = ts.replace("Z", "+00:00")
                created = datetime.fromisoformat(ts)
            except Exception:
                continue

            def _parse_list(val):
                if not val:
                    return []
                try:
                    p = json.loads(val)
                    return p if isinstance(p, list) else []
                except Exception:
                    return []

            self._all.append(
                EvalSample(
                    session_id=row["session_id"],
                    turn_id=row["turn_id"],
                    backend=row["backend"],
                    query=row["query"] or "",
                    user_outcome=row["user_outcome"],
                    reward_accuracy=row["computed_reward_accuracy"] or 0.0,
                    reward_grounding=row["computed_reward_grounding"] or 0.0,
                    reward_temporal=row["computed_reward_temporal"] or 0.0,
                    created_at=created,
                    assistant_response=row["assistant_response"] or "",
                    files_read=_parse_list(row["files_read"]),
                    files_written=_parse_list(row["files_written"]),
                    tool_calls=_parse_list(row["tool_calls"]),
                )
            )

        conn.close()
        self._split()
        return self

    def _split(self) -> None:
        """Stratified split by outcome to keep distribution balanced."""
        rng = random.Random(self.seed)

        # Group by outcome
        by_outcome: dict[str, list[EvalSample]] = {}
        for s in self._all:
            by_outcome.setdefault(s.user_outcome, []).append(s)

        train, val, holdout = [], [], []
        for outcome, samples in by_outcome.items():
            rng.shuffle(samples)
            n = len(samples)
            n_val = max(1, int(n * self.val_ratio))
            n_holdout = max(1, int(n * self.holdout_ratio))
            # Ensure at least one in train if possible
            if n_val + n_holdout >= n:
                n_val = max(1, n // 3)
                n_holdout = max(1, n // 3)

            val.extend(samples[:n_val])
            holdout.extend(samples[n_val : n_val + n_holdout])
            train.extend(samples[n_val + n_holdout :])

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(holdout)

        self._train = train
        self._val = val
        self._holdout = holdout

    @property
    def train(self) -> list[EvalSample]:
        if self._train is None:
            raise RuntimeError("Call load() first")
        return self._train

    @property
    def val(self) -> list[EvalSample]:
        if self._val is None:
            raise RuntimeError("Call load() first")
        return self._val

    @property
    def holdout(self) -> list[EvalSample]:
        if self._holdout is None:
            raise RuntimeError("Call load() first")
        return self._holdout

    def stats(self) -> dict[str, Any]:
        def _count(samples):
            from collections import Counter
            return dict(Counter(s.user_outcome for s in samples))

        return {
            "total": len(self._all),
            "train": len(self.train),
            "val": len(self.val),
            "holdout": len(self.holdout),
            "train_outcomes": _count(self.train),
            "val_outcomes": _count(self.val),
            "holdout_outcomes": _count(self.holdout),
        }
