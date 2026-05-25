"""
Mem0PolicySync — Sync agentRL policy data to Mem0 server for multi-agent sharing.

Pushes policy summaries as shared memories so other agents can learn
from this agent's experiences.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from collections import defaultdict
from typing import Any


class Mem0PolicySync:
    """Sync policy snapshots to/from Mem0 server."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, agent_id: str = "hermes-agent", user_id: str = "default") -> None:
        self.api_key = api_key or os.environ.get("MEM0_API_KEY", "")
        self.base_url = base_url or self._infer_base_url()
        self.agent_id = agent_id
        self.user_id = user_id

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    def push_policy(
        self,
        first_action_stats: dict[str, Any],
        transition_counts: dict[str, dict[str, int]],
        correction_fixes: list[dict[str, Any]],
        agent_steps_total: int = 0,
    ) -> bool:
        """Push a policy snapshot to Mem0 as shared memory."""
        if not self.api_key or not self.base_url:
            return False

        # Build human-readable policy summary
        lines = ["agentRL shared policy snapshot:"]
        lines.append(f"Agent: {self.agent_id} | Total agent steps: {agent_steps_total}")
        lines.append("")

        # First actions
        lines.append("## First Actions by Intent")
        for intent, stats in sorted(first_action_stats.items()):
            if stats.get("total", 0) > 0:
                sr = stats.get("successes", 0) / stats["total"]
                lines.append(f"- {intent}: {stats.get('action', '?')} ({stats['total']} tasks, {sr:.0%} success)")

        # Top transitions
        lines.append("")
        lines.append("## Top Transitions")
        trans = []
        for cur, nexts in transition_counts.items():
            for nxt, cnt in nexts.items():
                trans.append((cur, nxt, cnt))
        trans.sort(key=lambda x: -x[2])
        for cur, nxt, cnt in trans[:15]:
            lines.append(f"- {cur} -> {nxt} ({cnt}x)")

        # Corrections
        if correction_fixes:
            lines.append("")
            lines.append("## Recent Corrections")
            for fix in correction_fixes[-10:]:
                lines.append(f"- {fix.get('wrong_action', '?')} corrected({fix.get('correction_type', '?')}) in '{fix.get('context', '')[:60]}'")

        content = "\n".join(lines)

        payload = {
            "messages": [{"role": "user", "content": content}],
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "metadata": {
                "type": "agentrl_policy",
                "sync_version": "1",
            },
            "infer": False,
        }

        try:
            self._api_post("/v1/memories/", payload)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Pull
    # ------------------------------------------------------------------

    def pull_policy(self) -> dict[str, Any] | None:
        """Pull shared policy data from Mem0. Returns parsed dict or None."""
        if not self.api_key or not self.base_url:
            return None

        payload = {
            "query": "agentRL shared policy snapshot",
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "limit": 20,
        }

        try:
            result = self._api_post("/v1/memories/search/", payload)
            memories = result.get("results", [])
            if not memories:
                return None

            return self._parse_policy_memories(memories)
        except Exception:
            return None

    def _parse_policy_memories(self, memories: list[dict]) -> dict[str, Any]:
        """Parse policy snapshots from Mem0 search results."""
        merged = {
            "first_action_stats": defaultdict(lambda: {"action": "", "successes": 0, "total": 0}),
            "transition_counts": defaultdict(lambda: defaultdict(int)),
            "correction_fixes": [],
        }

        for mem in memories:
            text = mem.get("memory", "")
            if not text:
                continue

            # Parse first actions: "- intent: action (N tasks, X% success)"
            import re
            for m in re.finditer(r"-\s+(\w+):\s+(\S+)\s+\((\d+)\s+tasks,\s+(\d+)%\s+success\)", text):
                intent, action, total_str, sr_str = m.groups()
                total = int(total_str)
                successes = int(total * int(sr_str) / 100)
                stats = merged["first_action_stats"][intent]
                if stats["total"] == 0:
                    stats["action"] = action
                stats["total"] += total
                stats["successes"] += successes

            # Parse transitions: "- action -> action (Nx)"
            for m in re.finditer(r"-\s+(\S+)\s+->\s+(\S+)\s+\((\d+)x\)", text):
                cur, nxt, cnt = m.groups()
                merged["transition_counts"][cur][nxt] += int(cnt)

            # Parse corrections: "- action corrected(type) in 'context'"
            for m in re.finditer(r"-\s+(\S+)\s+corrected\(([^)]+)\)\s+in\s+'([^']+)'", text):
                wrong, ctype, ctx = m.groups()
                merged["correction_fixes"].append({
                    "wrong_action": wrong,
                    "correction_type": ctype,
                    "context": ctx,
                })

        return {
            "first_action_stats": {k: dict(v) for k, v in merged["first_action_stats"].items()},
            "transition_counts": {k: dict(v) for k, v in merged["transition_counts"].items()},
            "correction_fixes": merged["correction_fixes"],
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _infer_base_url(self) -> str:
        base = os.environ.get("MEM0_BASE_URL", "")
        if base:
            return base
        host = os.environ.get("MEM0_HOST", "127.0.0.1")
        port = os.environ.get("MEM0_PORT", "8888")
        if port == "8443":
            return f"https://{host}:{port}"
        return f"http://{host}:{port}"

    def _api_post(self, path: str, data: dict) -> dict:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
