"""
KimiClient — Lightweight OpenAI-compatible client for Moonshot API.

No external dependencies (uses urllib).
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any


class KimiClient:
    """Call Kimi (Moonshot) API for lightweight completions."""

    def __init__(self, api_key: str | None = None, model: str = "moonshot-v1-8k") -> None:
        self.api_key = api_key or os.environ.get("KIMI_API_KEY", "")
        self.model = model
        self.base_url = "https://api.moonshot.cn/v1"

    def chat(
        self,
        user: str,
        system: str = "You are a helpful assistant.",
        temperature: float = 0.3,
        max_tokens: int = 256,
    ) -> str:
        """Single-turn chat completion."""
        if not self.api_key:
            return ""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8")
            raise RuntimeError(f"Kimi API HTTP {e.code}: {err}")
        except Exception as e:
            raise RuntimeError(f"Kimi API failed: {e}")

    def infer_target(
        self,
        goal: str,
        current_action: str,
        recommended_action: str,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Ask Kimi to suggest a concrete target (file path, command, URL)
        for the recommended action.
        """
        history_str = ""
        if history:
            history_str = "\nRecent actions:\n" + "\n".join(
                f"- {h.get('action', '?')}: {h.get('target', '')}" for h in history[-5:]
            )

        prompt = (
            f"Task goal: {goal}\n"
            f"Current action just completed: {current_action}\n"
            f"Next recommended action type: {recommended_action}\n"
            f"{history_str}\n\n"
            f"Based on the task goal and current progress, what is the most logical "
            f"concrete target for '{recommended_action}'?\n"
            f"- If read_file/write_file/patch: give a likely file path\n"
            f"- If terminal: give a likely command or check\n"
            f"- If browser: give a likely URL or search query\n"
            f"- If search: give a likely search query\n"
            f"- If llm_response: say '(no target needed)'\n\n"
            f"Respond with ONLY the target string, no explanation."
        )

        try:
            result = self.chat(
                user=prompt,
                system="You are an expert devops assistant. Output only the target string.",
                temperature=0.2,
                max_tokens=128,
            )
            # Clean up
            result = result.strip().strip('"').strip("'")
            if result.lower() in ("(no target needed)", "none", "n/a"):
                return ""
            return result
        except Exception:
            return ""

    def judge_outcome(
        self,
        goal: str,
        messages: list[dict[str, Any]],
    ) -> tuple[str, float]:
        """
        Use Kimi to judge whether a session succeeded, was corrected, or failed.
        Returns (outcome, confidence).
        """
        # Build a concise transcript
        transcript = []
        for m in messages[-20:]:  # last 20 messages for context limit
            role = m.get("role", "")
            content = m.get("content", "")[:200]
            if role == "user":
                transcript.append(f"User: {content}")
            elif role == "assistant":
                transcript.append(f"Agent: {content[:150]}")

        prompt = (
            f"Task goal: {goal}\n\n"
            f"Session transcript (last turns):\n"
            + "\n".join(transcript)
            + "\n\n"
            "Judge the final outcome:\n"
            "- approved: task completed successfully, user satisfied\n"
            "- corrected: user had to correct or redirect the agent\n"
            "- failed: task abandoned or user explicitly rejected\n"
            "- unknown: cannot determine\n\n"
            "Respond with ONLY the outcome label, e.g. 'approved'."
        )

        try:
            result = self.chat(
                user=prompt,
                system="You are a strict session outcome judge. Output only the label.",
                temperature=0.0,
                max_tokens=16,
            )
            outcome = result.strip().lower().split()[0]
            if outcome not in ("approved", "corrected", "failed", "unknown"):
                outcome = "unknown"
            return outcome, 0.8
        except Exception:
            return "unknown", 0.0
