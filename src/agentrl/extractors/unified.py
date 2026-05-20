from __future__ import annotations

from typing import Iterator

from agentrl.extractors.codex import CodexParser
from agentrl.extractors.kimi import KimiParser
from agentrl.extractors.hermes import HermesParser
from agentrl.extractors.ccconnect import CCConnectParser
from agentrl.models import UnifiedSession, UnifiedTurn


class UnifiedExtractor:
    PARSERS = [
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
