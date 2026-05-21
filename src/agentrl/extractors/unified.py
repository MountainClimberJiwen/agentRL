from __future__ import annotations

from typing import Any, Iterator

from agentrl.extractors.claude_code import ClaudeCodeParser
from agentrl.extractors.ccconnect import CCConnectParser
from agentrl.extractors.codex import CodexParser
from agentrl.extractors.hermes import HermesParser
from agentrl.extractors.kimi import KimiParser
from agentrl.models import UnifiedSession, UnifiedTurn


class UnifiedExtractor:
    PARSERS = [
        ClaudeCodeParser(),
        CodexParser(),
        KimiParser(),
        HermesParser(),
        CCConnectParser(),
    ]

    def iter_sessions(self) -> Iterator[UnifiedSession]:
        for parser in self.PARSERS:
            try:
                yield from parser.iter_sessions()
            except Exception as e:
                print(f"[{parser.BACKEND}] parser error: {e}")

    def iter_turns(self) -> Iterator[UnifiedTurn]:
        for sess in self.iter_sessions():
            yield from sess.turns

    def iter_trajectories(self) -> Iterator[dict[str, Any]]:
        """Yield fine-grained TaskTrajectory dicts with step-level corrections."""
        for parser in self.PARSERS:
            if not hasattr(parser, "iter_trajectories"):
                continue
            try:
                yield from parser.iter_trajectories()
            except Exception as e:
                print(f"[{parser.BACKEND}] trajectory error: {e}")
