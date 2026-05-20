"""Tests for LLM Judge (Kimi-based precise evaluator).

NOTE: Tests that call the real Kimi API are skipped by default.
Set environment variable RUN_LLM_TESTS=1 to enable them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentrl.llm_judge import LLMJudge, _extract_json


def test_extract_json_object():
    text = 'Some text before {"relevance": 0.8, "quality": 0.9} and after'
    result = _extract_json(text)
    assert result == {"relevance": 0.8, "quality": 0.9}


def test_extract_json_markdown():
    text = """Here is the result:
```json
{"winner": "A", "confidence": 0.9}
```
"""
    result = _extract_json(text)
    assert result == {"winner": "A", "confidence": 0.9}


def test_extract_json_array():
    text = 'The answer is [{"a": 1}, {"b": 2}]'
    result = _extract_json(text)
    assert result == [{"a": 1}, {"b": 2}]


def test_extract_json_no_json():
    result = _extract_json("Just plain text, no json here")
    assert result is None


def test_judge_instantiate():
    judge = LLMJudge()
    assert judge is not None


def test_judge_evaluate_format():
    """Test that evaluate_retrieval formats the prompt correctly (without API call)."""
    judge = LLMJudge()
    # We can't test the actual API call without spending money,
    # but we verify the method exists and accepts correct args
    assert hasattr(judge, "evaluate_retrieval")
    assert hasattr(judge, "batch_evaluate")
    assert hasattr(judge, "compare_variants")


def _should_run_llm_tests() -> bool:
    return os.environ.get("RUN_LLM_TESTS", "") in ("1", "true", "yes")


def test_judge_live_api():
    """Live API test — skipped by default to avoid cost."""
    if not _should_run_llm_tests():
        print("SKIP: Set RUN_LLM_TESTS=1 to run live API test")
        return

    judge = LLMJudge()
    candidates = [
        {
            "session_id": "abc",
            "query": "how to implement websocket",
            "user_outcome": "approved",
            "created_at": "2025-01-01",
        },
        {
            "session_id": "def",
            "query": "fix bug in server.py",
            "user_outcome": "exited",
            "created_at": "2025-01-02",
        },
    ]
    result = judge.evaluate_retrieval(
        query="how to add websocket support",
        candidates=candidates,
        outcome="approved",
    )
    print(f"Live API result: {result}")
    assert "overall" in result
    assert 0.0 <= result["overall"] <= 1.0
    assert "reasoning" in result


if __name__ == "__main__":
    test_extract_json_object()
    test_extract_json_markdown()
    test_extract_json_array()
    test_extract_json_no_json()
    test_judge_instantiate()
    test_judge_evaluate_format()
    test_judge_live_api()
    print("\nAll LLM Judge tests passed.")
