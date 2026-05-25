"""
SoftmaxPolicy — Lightweight tabular policy network.

Stores action preferences per state as a JSON-serializable table.
Uses softmax exploration with temperature annealing.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StateActionPrefs:
    """Action preferences for a single state."""
    prefs: dict[str, float] = field(default_factory=dict)
    visit_count: int = 0


class SoftmaxPolicy:
    """
    Tabular policy with softmax action selection.
    State = f"{intent}:{current_action}:{step_idx}" (see OnlinePolicyUpdater._encode_state)
    Action = tool name or "llm_response"
    """

    def __init__(self, temperature: float = 1.0, min_temp: float = 0.2, decay: float = 0.995) -> None:
        self.temperature = temperature
        self.min_temp = min_temp
        self.decay = decay
        self.table: dict[str, StateActionPrefs] = {}  # state -> prefs
        self.global_step = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select(self, state: str, available_actions: list[str]) -> tuple[str, dict[str, float]]:
        """Sample action from softmax distribution."""
        prefs = self._get_prefs(state)
        probs = self._softmax(prefs, available_actions)
        chosen = random.choices(available_actions, weights=[probs[a] for a in available_actions], k=1)[0]
        return chosen, probs

    def get_probs(self, state: str, available_actions: list[str]) -> dict[str, float]:
        """Return probability distribution over actions."""
        prefs = self._get_prefs(state)
        return self._softmax(prefs, available_actions)

    def get_best(self, state: str, available_actions: list[str]) -> str:
        """Greedy best action."""
        prefs = self._get_prefs(state)
        return max(available_actions, key=lambda a: prefs.prefs.get(a, 0.0))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, state: str, action: str, reward: float, lr: float = 0.1) -> None:
        """Single-step policy gradient update (REINFORCE-style)."""
        prefs = self._get_prefs(state)
        prefs.visit_count += 1

        # Preference gradient: increase chosen action, decrease others
        current = prefs.prefs.get(action, 0.0)
        prefs.prefs[action] = current + lr * reward

        # Small negative drift for unchosen actions (soft competitive update)
        for other_action in list(prefs.prefs.keys()):
            if other_action != action:
                prefs.prefs[other_action] -= lr * reward * 0.1

        self.global_step += 1
        self.temperature = max(self.min_temp, self.temperature * self.decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "SoftmaxPolicy":
        """Load policy from JSON file."""
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p) as f:
            data = json.load(f)
        policy = cls(
            temperature=data.get("temperature", 1.0),
            min_temp=data.get("min_temp", 0.2),
            decay=data.get("decay", 0.995),
        )
        policy.global_step = data.get("global_step", 0)
        for state, payload in data.get("table", {}).items():
            policy.table[state] = StateActionPrefs(
                prefs=payload.get("prefs", {}),
                visit_count=payload.get("visit_count", 0),
            )
        return policy

    def save(self, path: str) -> None:
        """Save policy to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "temperature": self.temperature,
            "min_temp": self.min_temp,
            "decay": self.decay,
            "global_step": self.global_step,
            "table": {
                state: {
                    "prefs": sap.prefs,
                    "visit_count": sap.visit_count,
                }
                for state, sap in self.table.items()
            },
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_prefs(self, state: str) -> StateActionPrefs:
        if state not in self.table:
            self.table[state] = StateActionPrefs()
        return self.table[state]

    def _softmax(self, prefs: StateActionPrefs, actions: list[str]) -> dict[str, float]:
        vals = [prefs.prefs.get(a, 0.0) for a in actions]
        max_v = max(vals) if vals else 0.0
        exp_vals = [math.exp((v - max_v) / max(self.temperature, 1e-6)) for v in vals]
        sum_exp = sum(exp_vals)
        return {a: ev / sum_exp for a, ev in zip(actions, exp_vals)}
