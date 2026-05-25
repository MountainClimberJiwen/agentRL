"""Mem0-oss memory client (HTTP API)."""

from __future__ import annotations

import json
import os
from typing import Any

from agentrl.memory.unified_client import UnifiedMemoryClient
from agentrl.sync.mem0_sync import Mem0PolicySync


class Mem0MemoryClient(UnifiedMemoryClient):
    """Mem0 HTTP API backend."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self._sync = Mem0PolicySync(base_url=base_url, api_key=api_key)

    def is_available(self) -> bool:
        return bool(self._sync.api_key and self._sync.base_url)

    def push_policy(self, policy_data: dict[str, Any]) -> bool:
        return self._sync.push_policy(
            first_action_stats=policy_data.get("first_action_stats", {}),
            transition_counts=policy_data.get("transition_counts", {}),
            correction_fixes=policy_data.get("correction_fixes", []),
            agent_steps_total=policy_data.get("agent_steps_total", 0),
        )

    def pull_policy(self) -> dict[str, Any] | None:
        return self._sync.pull_policy()

    def write_memory(self, content: str, metadata: dict[str, Any] | None = None) -> bool:
        # Mem0 doesn't have a generic write endpoint in the same way;
        # we use the push_policy format for structured data.
        return self.push_policy(metadata or {"content": content})

    def search_memory(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        try:
            payload = {
                "query": query,
                "user_id": self._sync.user_id,
                "agent_id": self._sync.agent_id,
                "limit": limit,
            }
            result = self._sync._api_post("/v1/memories/search/", payload)
            memories = result.get("results", [])
            return [{"id": m.get("id", ""), "content": m.get("memory", ""), "score": m.get("score", 0)} for m in memories]
        except Exception:
            return []
