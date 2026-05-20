"""LLM Judge for precise evaluation of prompt/memory policy variants.

Uses Kimi API (curl subprocess, zero pip dependencies) to score retrieval
quality. Adapted from text-to-cad/freecad-assembler/kimi_http.py.

All evaluation uses the FROZEN LLM — we only send prompts to it and read
scores back. No model weights are touched.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from typing import Any

# Kimi For Coding endpoint (from text-to-cad project)
KIMI_API_KEY = os.environ.get("KIMI_API_KEY")
KIMI_BASE_URL = "https://api.kimi.com/coding"
KIMI_MODEL = "k2p5"          # k2p5 maps to kimi-for-coding on this endpoint
KIMI_CLAW_ID = os.environ.get("KIMI_CLAW_ID")


def _extract_json(text: str) -> Any | None:
    """Extract the largest valid JSON object or array from text."""
    best = None
    best_size = 0

    # Try markdown code blocks first
    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
        matches = re.findall(pattern, text, re.DOTALL)
        for m in matches:
            m = m.strip()
            if m.startswith("json"):
                m = m[3:].strip()
            try:
                candidate = json.loads(m)
                size = len(json.dumps(candidate))
                if size > best_size:
                    best = candidate
                    best_size = size
            except json.JSONDecodeError:
                continue

    # Fallback: find balanced braces
    for start in [m.start() for m in re.finditer(r"[\{\[]", text)]:
        brace = text[start]
        close = "}" if brace == "{" else "]"
        count = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == brace:
                count += 1
            elif text[i] == close:
                count -= 1
                if count == 0:
                    end = i + 1
                    break
        if end > start:
            try:
                candidate = json.loads(text[start:end])
                size = len(json.dumps(candidate))
                if size > best_size:
                    best = candidate
                    best_size = size
            except json.JSONDecodeError:
                pass

    return best


def call_kimi(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> dict:
    """Call Kimi For Coding chat completions API via curl.

    Returns the parsed JSON extracted from the response.
    """
    url = f"{KIMI_BASE_URL}/v1/chat/completions"
    headers = [
        "Content-Type: application/json",
        f"Authorization: Bearer {KIMI_API_KEY}",
        "User-Agent: Desktop Kimi Claw Plugin",
        f"X-Kimi-Claw-ID: {KIMI_CLAW_ID}",
    ]
    payload = {
        "model": KIMI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(payload, f, ensure_ascii=False)
        payload_path = f.name

    cmd = ["curl", "-s", "-w", r"\nHTTP_CODE:%{http_code}\n", "--max-time", "180"]
    for h in headers:
        cmd.extend(["-H", h])
    cmd.extend(["-d", f"@{payload_path}", url])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=200)
    finally:
        os.unlink(payload_path)

    stdout = result.stdout
    lines = stdout.strip().splitlines()
    if not lines:
        raise RuntimeError("Empty response from curl")

    http_code_line = lines[-1]
    if http_code_line.startswith("HTTP_CODE:"):
        http_code = int(http_code_line.split(":", 1)[1])
        body = "\n".join(lines[:-1])
    else:
        http_code = 0
        body = stdout

    if http_code != 200:
        raise RuntimeError(f"Kimi API error {http_code}: {body[:1000]}")

    data = json.loads(body)
    msg = data["choices"][0]["message"]

    # k2p5 reasoning model: content may be empty, analysis in reasoning_content
    for src in [msg.get("content", ""), msg.get("reasoning_content", "")]:
        if not src:
            continue
        parsed = _extract_json(src)
        if parsed is not None:
            return parsed

    raise RuntimeError("Could not extract valid JSON from API response.")


_JUDGE_PROMPT = """You are an expert evaluator of AI agent memory retrieval systems.

Your task: given a user query and a list of retrieved candidate sessions from the agent's memory, evaluate the retrieval quality.

## Evaluation Criteria

1. **Relevance** (0.0-1.0): Are the retrieved sessions topically relevant to the user query? Would including these sessions in the agent's context help answer the query?
2. **Quality** (0.0-1.0): Do the retrieved sessions have positive outcomes (approved/completed) rather than negative ones (exited/rejected/corrected)? High-quality memories are more trustworthy.
3. **Diversity** (0.0-1.0): Are the retrieved sessions diverse enough to provide broad context, or are they redundant repeats of the same information?
4. **Temporal Accuracy** (0.0-1.0): If the query contains temporal references (e.g., "last week", "yesterday"), are the retrieved sessions from the correct time period? If no temporal reference, score 0.5.

## Input

User Query: {query}

Retrieved Sessions:
{candidates_json}

Ground Truth Outcome for this query: {outcome}

## Output

Return ONLY a JSON object (no markdown fences):
{{
  "relevance": 0.0-1.0,
  "quality": 0.0-1.0,
  "diversity": 0.0-1.0,
  "temporal_accuracy": 0.0-1.0,
  "overall": 0.0-1.0,
  "reasoning": "one-sentence explanation"
}}
"""


class LLMJudge:
    """Use a frozen LLM (Kimi) to score retrieval quality.

    This is expensive (~$0.01-0.05 per call) so use sparingly:
    - Full eval: sample 20-50 examples from val set
    - A/B test: compare 2 prompt variants on 10 examples each
    """

    def __init__(self, api_key: str | None = None):
        global KIMI_API_KEY
        if api_key:
            KIMI_API_KEY = api_key

    def evaluate_retrieval(
        self,
        query: str,
        candidates: list[dict],
        outcome: str,
    ) -> dict[str, Any]:
        """Score one retrieval result. Returns dict with relevance/quality/etc."""
        prompt = _JUDGE_PROMPT.format(
            query=query,
            candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2),
            outcome=outcome,
        )
        messages = [{"role": "user", "content": prompt}]
        result = call_kimi(messages, temperature=0.2, max_tokens=1024)

        # Validate scores are in [0, 1]
        for key in ["relevance", "quality", "diversity", "temporal_accuracy", "overall"]:
            if key in result and isinstance(result[key], (int, float)):
                result[key] = max(0.0, min(1.0, float(result[key])))

        return result

    def batch_evaluate(
        self,
        items: list[dict],
        progress_every: int = 5,
    ) -> list[dict]:
        """Evaluate multiple retrieval results.

        Each item must have keys: query, candidates, outcome.
        """
        results = []
        for i, item in enumerate(items):
            try:
                score = self.evaluate_retrieval(
                    query=item["query"],
                    candidates=item["candidates"],
                    outcome=item["outcome"],
                )
                results.append({"index": i, "score": score, "error": None})
            except Exception as e:
                results.append({"index": i, "score": None, "error": str(e)})

            if progress_every and (i + 1) % progress_every == 0:
                print(f"  LLM Judge: evaluated {i + 1}/{len(items)} items...")

        return results

    def compare_variants(
        self,
        query: str,
        outcome: str,
        variant_a_candidates: list[dict],
        variant_b_candidates: list[dict],
        variant_a_name: str = "A",
        variant_b_name: str = "B",
    ) -> dict[str, Any]:
        """Direct A/B comparison of two retrieval strategies on the same query.

        Returns which variant is better and why.
        """
        prompt = f"""You are an expert evaluator comparing two memory retrieval strategies.

User Query: {query}
Ground Truth Outcome: {outcome}

## Variant {variant_a_name}
{json.dumps(variant_a_candidates, ensure_ascii=False, indent=2)}

## Variant {variant_b_name}
{json.dumps(variant_b_candidates, ensure_ascii=False, indent=2)}

Compare the two variants on:
1. Which retrieves more relevant sessions?
2. Which retrieves higher-quality sessions (better outcomes)?
3. Which would be more helpful to the agent?

Output ONLY JSON:
{{
  "winner": "A" or "B",
  "confidence": 0.0-1.0,
  "reasoning": "explanation",
  "variant_a_score": 0.0-1.0,
  "variant_b_score": 0.0-1.0
}}
"""
        messages = [{"role": "user", "content": prompt}]
        return call_kimi(messages, temperature=0.2, max_tokens=1024)
