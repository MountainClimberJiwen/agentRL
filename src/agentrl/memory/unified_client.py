"""
UnifiedMemoryClient — Abstract interface for multiple memory backends.

Supports:
  - mem0-oss (HTTP API)
  - memos-local-hermes-plugin (SQLite)

Auto-detects which backend is available via environment variables.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any


class UnifiedMemoryClient(ABC):
    """Abstract memory client for policy sync and memory operations."""

    @abstractmethod
    def push_policy(self, policy_data: dict[str, Any]) -> bool:
        """Push policy snapshot to shared memory."""
        ...

    @abstractmethod
    def pull_policy(self) -> dict[str, Any] | None:
        """Pull latest policy snapshot from shared memory."""
        ...

    @abstractmethod
    def write_memory(self, content: str, metadata: dict[str, Any] | None = None) -> bool:
        """Write a memory entry."""
        ...

    @abstractmethod
    def search_memory(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search memory entries."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is configured and reachable."""
        ...


def get_memory_client() -> UnifiedMemoryClient:
    """
    Factory: auto-detect and return the best available memory client.

    Priority:
      1. mem0-oss (if MEM0_API_KEY or MEM0_BASE_URL is set)
      2. memos-local (if ~/.hermes/memos/memos.db exists or MEMOS_EMBEDDING_MODEL is set)
      3. No-op fallback
    """
    from .mem0_client import Mem0MemoryClient
    from .memos_client import MemosLocalMemoryClient

    # Try mem0 first
    mem0 = Mem0MemoryClient()
    if mem0.is_available():
        return mem0

    # Fallback to memos-local
    memos = MemosLocalMemoryClient()
    if memos.is_available():
        return memos

    # No-op fallback
    return _NoOpMemoryClient()


class _NoOpMemoryClient(UnifiedMemoryClient):
    """Fallback when no memory backend is available."""

    def push_policy(self, policy_data: dict[str, Any]) -> bool:
        return False

    def pull_policy(self) -> dict[str, Any] | None:
        return None

    def write_memory(self, content: str, metadata: dict[str, Any] | None = None) -> bool:
        return False

    def search_memory(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return []

    def is_available(self) -> bool:
        return True
