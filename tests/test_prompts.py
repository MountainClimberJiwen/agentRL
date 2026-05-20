"""Tests for prompt registry and assembly."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentrl.prompts import PromptAssembler, PromptRegistry


def test_registry_load_default():
    reg = PromptRegistry()
    text = reg.load("router")
    assert "Memory Router" in text
    assert "{query}" in text  # template has placeholders


def test_registry_load_system():
    reg = PromptRegistry()
    text = reg.load("system")
    assert "AI coding assistant" in text


def test_registry_list_versions():
    reg = PromptRegistry()
    versions = reg.list_versions("router")
    assert "default" in versions


def test_registry_register_in_memory():
    reg = PromptRegistry()
    reg.register("router", "EVOLVED TEMPLATE", version="v2")
    assert reg.load("router", "v2") == "EVOLVED TEMPLATE"


def test_assembler_basic():
    asm = PromptAssembler()
    msgs = asm.assemble(
        user_query="Hello",
        system_prompt="You are helpful.",
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "Hello"


def test_assembler_with_memory():
    asm = PromptAssembler()
    msgs = asm.assemble(
        user_query="Hello",
        system_prompt="You are helpful.",
        memory_context="Past: you fixed a bug.",
    )
    assert len(msgs) == 3
    assert "Past: you fixed a bug." in msgs[1]["content"]


def test_assembler_build_memory_context():
    asm = PromptAssembler()
    from agentrl.prompts import SelectedEvidence

    selected = [
        SelectedEvidence(
            session_id="abc-123",
            turn_id="turn-1",
            reason="same project",
            relevance_score=0.9,
        )
    ]
    lookup = {
        "abc-123": {
            "created_at": "2025-01-01T00:00:00+00:00",
            "user_outcome": "approved",
            "query": "fix the bug",
            "assistant_response": "done",
        }
    }
    ctx = asm.build_memory_context(selected, lookup)
    assert "Retrieved Memory Context" in ctx
    assert "fix the bug" in ctx
    assert "0.90" in ctx


if __name__ == "__main__":
    test_registry_load_default()
    test_registry_load_system()
    test_registry_list_versions()
    test_registry_register_in_memory()
    test_assembler_basic()
    test_assembler_with_memory()
    test_assembler_build_memory_context()
    print("All prompt tests passed.")
