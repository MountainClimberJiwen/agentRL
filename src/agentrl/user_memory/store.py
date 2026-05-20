"""Store and load user preference memory as structured data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_STORE_PATH = Path(__file__).parent.parent.parent.parent / "data" / "user_memory.json"


class UserMemoryStore:
    """Persist user profile and preferences to disk.

    The memory is stored as JSON — readable, editable, version-controllable.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_STORE_PATH
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_profile(self) -> dict[str, Any] | None:
        return self._data.get("profile")

    def set_profile(self, profile: dict[str, Any]) -> None:
        self._data["profile"] = profile
        self.save()

    def get_prompt_block(self) -> str:
        """Return the user preference block formatted for prompt injection."""
        profile = self.get_profile()
        if not profile:
            return ""
        from agentrl.patterns.miner import UserProfile
        p = UserProfile(**profile)
        return p.to_prompt_block()

    def update_from_miner(self, profile: "UserProfile") -> None:
        """Replace the stored profile with freshly mined data."""
        self.set_profile(profile.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()
