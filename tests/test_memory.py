"""Tests for memory retrieval (coarse filter, router, selector)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentrl.memory.retrieval import (
    CoarseFilter,
    EvidenceSelector,
    MemoryRouter,
    _extract_keywords,
    _extract_project_from_query,
)
from agentrl.prompts import PromptRegistry

DB_PATH = Path(__file__).parent.parent / "data" / "agentrl.db"


def test_extract_keywords():
    kws = _extract_keywords("how to implement dry-run flag in Python script")
    assert "dry" in kws or "run" in kws or "implement" in kws
    assert len(kws) <= 10


def test_extract_project_from_query():
    q = "# Context\n## Active file: scripts/aster_arbitrage.py\n## My request: fix this"
    proj = _extract_project_from_query(q)
    assert proj == "scripts"

    q2 = "write me a script under folder ./tests"
    proj2 = _extract_project_from_query(q2)
    assert proj2 == "tests"


def test_coarse_filter_runs():
    if not DB_PATH.exists():
        print("SKIP: no DB found")
        return
    cf = CoarseFilter(DB_PATH)
    candidates = cf.filter(
        query="how to implement websocket",
        max_candidates=10,
    )
    assert isinstance(candidates, list)
    assert len(candidates) <= 10
    print(f"CoarseFilter returned {len(candidates)} candidates")


def test_coarse_filter_with_outcome_bias():
    if not DB_PATH.exists():
        print("SKIP: no DB found")
        return
    cf = CoarseFilter(DB_PATH)
    candidates = cf.filter(
        query="python",
        outcome_bias="approved_only",
        max_candidates=5,
    )
    for c in candidates:
        assert c.user_outcome == "approved"
    print(f"Approved-only filter returned {len(candidates)} candidates")


def test_memory_router_render():
    reg = PromptRegistry()
    router = MemoryRouter(reg)
    prompt = router.render(
        query="what did I work on last week?",
        project="scripts",
        recent_sessions=["sess-1", "sess-2"],
    )
    assert "Memory Router" in prompt
    assert "what did I work on last week?" in prompt
    assert "scripts" in prompt
    print("MemoryRouter render OK")


def test_memory_router_parse():
    reg = PromptRegistry()
    router = MemoryRouter(reg)
    fake_response = json.dumps({
        "time_scope": {"start": "2025-01-01T00:00:00+00:00", "end": "2025-01-07T00:00:00+00:00", "description": "last week"},
        "project_filter": "scripts",
        "keywords": ["websocket", "funding"],
        "outcome_bias": "approved_only",
        "max_sessions": 15,
        "reasoning": "user asked about last week"
    })
    strategy = router.parse_strategy(fake_response)
    assert strategy.project_filter == "scripts"
    assert strategy.max_sessions == 15
    assert "websocket" in strategy.keywords
    print(f"Parsed strategy: {strategy}")


def test_evidence_selector_render():
    reg = PromptRegistry()
    selector = EvidenceSelector(reg)

    from agentrl.memory.retrieval import CandidateSession
    from datetime import datetime, timezone

    candidates = [
        CandidateSession(
            session_id="abc-123",
            turn_id="turn-1",
            backend="codex",
            query="fix bug",
            assistant_response="fixed",
            user_outcome="approved",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
    ]
    prompt = selector.render(
        query="how to fix bugs?",
        candidates=candidates,
        project="scripts",
        max_sessions=5,
    )
    assert "Evidence Selector" in prompt
    assert "fix bug" in prompt
    print("EvidenceSelector render OK")


def test_evidence_selector_parse():
    reg = PromptRegistry()
    selector = EvidenceSelector(reg)
    fake_response = json.dumps([
        {"session_id": "abc-123", "turn_id": "turn-1", "reason": "relevant", "relevance_score": 0.95}
    ])
    selected = selector.parse_selection(fake_response)
    assert len(selected) == 1
    assert selected[0].session_id == "abc-123"
    assert selected[0].relevance_score == 0.95
    print("EvidenceSelector parse OK")


if __name__ == "__main__":
    test_extract_keywords()
    test_extract_project_from_query()
    test_coarse_filter_runs()
    test_coarse_filter_with_outcome_bias()
    test_memory_router_render()
    test_memory_router_parse()
    test_evidence_selector_render()
    test_evidence_selector_parse()
    print("\nAll memory tests passed.")
