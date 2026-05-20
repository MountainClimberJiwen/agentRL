"""PromptRegistry: load, version, and switch prompt templates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).parent


@dataclass
class RetrievalStrategy:
    """Output of Memory Router: how to retrieve candidate sessions."""

    time_scope: dict[str, Any] = field(default_factory=dict)
    project_filter: str = "any"
    keywords: list[str] = field(default_factory=list)
    outcome_bias: str = "any"  # approved_only | avoid_exited | any
    max_sessions: int = 20
    reasoning: str = ""


@dataclass
class SelectedEvidence:
    """Output of Evidence Selector: which sessions to inject."""

    session_id: str
    turn_id: str
    reason: str = ""
    relevance_score: float = 0.0


class PromptRegistry:
    """Load, version, and switch prompt templates from the prompts/ directory.

    Templates are plain text files. Versions are suffixes:
      router.txt        -> default version
      router_v2.txt     -> version "v2"
      router_v3_ab.txt  -> version "v3_ab"
    """

    def __init__(self, prompts_dir: Path | None = None):
        self._dir = prompts_dir or PROMPTS_DIR
        self._cache: dict[str, str] = {}

    def load(self, name: str, version: str | None = None) -> str:
        """Load a prompt template by name and optional version."""
        key = f"{name}:{version or 'default'}"
        if key in self._cache:
            return self._cache[key]

        filename = self._resolve_filename(name, version)
        path = self._dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")

        text = path.read_text(encoding="utf-8")
        self._cache[key] = text
        return text

    def register(self, name: str, template: str, version: str | None = None) -> None:
        """Register a template in-memory (useful for evolved variants)."""
        key = f"{name}:{version or 'default'}"
        self._cache[key] = template

    def list_versions(self, name: str) -> list[str]:
        """List available versions for a prompt name."""
        versions = []
        for path in self._dir.glob(f"{name}*.txt"):
            stem = path.stem
            if stem == name:
                versions.append("default")
            elif stem.startswith(f"{name}_"):
                versions.append(stem[len(name) + 1 :])
        return sorted(set(versions))

    def _resolve_filename(self, name: str, version: str | None) -> str:
        if not version or version == "default":
            return f"{name}.txt"
        candidate = f"{name}_{version}.txt"
        if (self._dir / candidate).exists():
            return candidate
        # Fallback: try with hyphen
        candidate2 = f"{name}-{version}.txt"
        if (self._dir / candidate2).exists():
            return candidate2
        raise FileNotFoundError(
            f"No prompt '{name}' with version '{version}' in {self._dir}"
        )


