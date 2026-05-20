"""
AgentRL Policy Network Plugin for Hermes

Injects learned behavioral strategies into each turn via the pre_llm_call hook.
Reads learned patterns from agentRL's user_memory.json and selects the most
relevant strategy based on current task context.

No core code changes required. Install to ~/.hermes/plugins/agentrl-policy/
and Hermes will auto-discover on next start.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AGENTRL_DATA_DIR = Path("/opt/agentrl/data")
USER_MEMORY_PATH = AGENTRL_DATA_DIR / "user_memory.json"
PLUGIN_DATA_DIR = Path(__file__).parent.parent / "data"
PLUGIN_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Max strategies to inject per turn (keep prompt compact)
MAX_STRATEGIES_PER_TURN = 3
# Min success rate threshold (ignore patterns below this)
MIN_SUCCESS_RATE = 0.15

# Default learned patterns (fallback when agentRL data not yet generated)
# These come from mining 376 real sessions. Will be overridden once
# /opt/agentrl/data/user_memory.json exists.
DEFAULT_LEARNED_PATTERNS: List[Dict[str, Any]] = [
    {
        "description": "Always read README and project config files before implementing or writing code",
        "success_rate": 0.91,
        "context": "coding",
        "tags": ["prerequisite", "read_first"],
    },
    {
        "description": "Read existing tests before modifying any production code",
        "success_rate": 0.88,
        "context": "coding",
        "tags": ["prerequisite", "testing"],
    },
    {
        "description": "For documentation tasks (README, comments, guides), ask the user to confirm scope and format before writing",
        "success_rate": 0.14,
        "context": "doc",
        "tags": ["confirmation", "low_success"],
    },
    {
        "description": "Before writing or modifying files, always read the existing files first to understand context",
        "success_rate": 0.85,
        "context": "general",
        "tags": ["prerequisite", "read_first"],
    },
    {
        "description": "When deploying, verify prerequisites (OS, dependencies, env vars) before running commands",
        "success_rate": 0.72,
        "context": "deploy",
        "tags": ["prerequisite", "verification"],
    },
    {
        "description": "For debugging, gather logs and error traces before proposing a fix",
        "success_rate": 0.68,
        "context": "debug",
        "tags": ["investigation", "logs"],
    },
    {
        "description": "When the user asks for a review or audit, start by reading the target files completely before commenting",
        "success_rate": 0.76,
        "context": "review",
        "tags": ["read_first", "thoroughness"],
    },
]

# ---------------------------------------------------------------------------
# Task intent detection
# ---------------------------------------------------------------------------

TASK_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "coding":   ["code", "implement", "function", "class", "write", "refactor", "fix bug", "add feature", "create", "写代码", "函数", "类", "实现", "修改", "改"],
    "doc":      ["document", "readme", "comment", "tutorial", "guide", "write doc", "documentation", "markdown", "文档", "注释", "教程", "指南"],
    "debug":    ["debug", "error", "exception", "traceback", "broken", "crash", "fail", "not working", "报错", "出错", "调试", "怎么回事", "不对", "错误"],
    "deploy":   ["deploy", "release", "publish", "build", "ci/cd", "docker", "push to", "production", "部署", "发布", "上线", "构建"],
    "test":     ["test", "unittest", "pytest", "spec", "coverage", "mock", "assert", "测试", "单测"],
    "review":   ["review", "audit", "check", "lint", "validate", "inspect", "审查", "检查", "审阅"],
    "explain":  ["explain", "how to", "what is", "why ", "understand", "describe", "mean", "解释", "说明", "什么是", "怎么"],
    "config":   ["config", "setup", "install", "configure", "env", "variable", "setting", "配置", "环境", "变量", "设置", "安装"],
}


def _detect_intent(text: str) -> str:
    """Detect task intent from user message. Returns the best-matching intent."""
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for intent, keywords in TASK_INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score:
            scores[intent] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# State analysis
# ---------------------------------------------------------------------------

def _analyze_state(
    user_message: Any,
    conversation_history: List[Dict[str, Any]],
    is_first_turn: bool,
    model: str,
    platform: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build a state dict for policy lookup."""
    query = user_message if isinstance(user_message, str) else ""
    intent = _detect_intent(query)
    turn_count = len([m for m in conversation_history if m.get("role") == "user"])
    
    # Detect if tools have already been used this session
    tools_used = set()
    for msg in conversation_history:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    name = fn.get("name", "") if isinstance(fn, dict) else ""
                    if name:
                        tools_used.add(name)
    
    return {
        "query": query,
        "intent": intent,
        "is_first_turn": is_first_turn,
        "turn_count": turn_count,
        "model": (model or "").lower(),
        "platform": (platform or "cli").lower(),
        "tools_used": list(tools_used),
        "has_memory_tool": "memory" in str([m.get("role") for m in conversation_history]),
    }


# ---------------------------------------------------------------------------
# Policy query
# ---------------------------------------------------------------------------

def _load_user_memory() -> Dict[str, Any]:
    """Load agentRL's user_memory.json."""
    if not USER_MEMORY_PATH.exists():
        return {}
    try:
        with open(USER_MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Failed to load user_memory.json: %s", e)
        return {}


def _match_strategy(state: Dict[str, Any], pattern: Dict[str, Any]) -> float:
    """Score how relevant a pattern is to the current state. Returns 0-1."""
    score = 0.0
    desc = pattern.get("description", "").lower()
    context = pattern.get("context", "").lower()
    intent = state["intent"]
    query = state["query"].lower()
    
    # Intent match
    intent_keywords = TASK_INTENT_KEYWORDS.get(intent, [])
    if any(kw in desc or kw in context for kw in intent_keywords):
        score += 0.4
    
    # Query keyword overlap
    query_words = set(re.findall(r"\b\w+\b", query))
    desc_words = set(re.findall(r"\b\w+\b", desc))
    if query_words and desc_words:
        overlap = len(query_words & desc_words) / max(len(query_words), 1)
        score += overlap * 0.3
    
    # Success rate bonus
    success_rate = pattern.get("success_rate", 0.5)
    score += success_rate * 0.2
    
    # First-turn bonus for prerequisite patterns
    if state["is_first_turn"] and ("read" in desc or "before" in desc):
        score += 0.1
    
    return min(1.0, score)


def _query_policy(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Query agentRL for the best matching strategies."""
    data = _load_user_memory()
    patterns = data.get("learned_patterns", [])
    if not patterns:
        # Fallback: try to read raw user_memory structure
        patterns = _extract_patterns_from_legacy(data)
    
    # If no agentRL data yet, use default learned patterns
    if not patterns:
        patterns = list(DEFAULT_LEARNED_PATTERNS)
    
    if not patterns:
        return []
    
    # Score and filter
    scored = []
    for p in patterns:
        sr = p.get("success_rate", 0.5)
        if sr < MIN_SUCCESS_RATE:
            continue
        relevance = _match_strategy(state, p)
        if relevance > 0.1:
            scored.append((relevance * sr, p))
    
    # Sort by combined score (relevance * success_rate)
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Return top N
    return [p for _, p in scored[:MAX_STRATEGIES_PER_TURN]]


def _extract_patterns_from_legacy(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Handle legacy user_memory.json formats."""
    patterns: List[Dict[str, Any]] = []
    
    # Format 1: { "preferences": [...], "workflows": [...], "failure_modes": [...] }
    for key in ("preferences", "workflows", "success_sequences", "failure_modes"):
        for item in data.get(key, []):
            if isinstance(item, dict):
                patterns.append({
                    "description": item.get("description", item.get("pattern", "")),
                    "success_rate": item.get("success_rate", item.get("frequency", 0.5)),
                    "context": item.get("context", ""),
                })
            elif isinstance(item, str):
                patterns.append({
                    "description": item,
                    "success_rate": 0.5,
                    "context": "",
                })
    
    # Format 2: flat list
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                patterns.append({
                    "description": item.get("description", str(item)),
                    "success_rate": item.get("success_rate", 0.5),
                    "context": item.get("context", ""),
                })
    
    return patterns


# ---------------------------------------------------------------------------
# Strategy prompt builder
# ---------------------------------------------------------------------------

def _build_strategy_prompt(
    policies: List[Dict[str, Any]],
    state: Dict[str, Any],
) -> str:
    """Build a compact strategy instruction block."""
    if not policies:
        return ""
    
    lines = ["## Learned Strategy (from agentRL)"]
    lines.append(f"_Task type: {state['intent']}_")
    lines.append("")
    
    for i, p in enumerate(policies, 1):
        desc = p.get("description", "").strip()
        sr = p.get("success_rate", 0.0)
        sr_pct = int(sr * 100)
        
        # Format for platform
        if state["platform"] in ("weixin", "wecom", "whatsapp", "sms"):
            # Compact plain text for mobile
            lines.append(f"{i}. {desc} ({sr_pct}% success)")
        else:
            lines.append(f"{i}. **{desc}** (success rate: {sr_pct}%)")
    
    lines.append("")
    lines.append("Apply the above strategy when relevant to this turn.")
    
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hook entry point
# ---------------------------------------------------------------------------

def pre_llm_call(
    user_message: Any,
    conversation_history: List[Dict[str, Any]],
    is_first_turn: bool,
    model: str,
    platform: str,
    **kwargs: Any,
) -> Dict[str, str]:
    """
    Hermes pre_llm_call hook.
    
    Analyzes current state, queries agentRL policy, and returns strategy
    context to be injected into the user message.
    """
    try:
        state = _analyze_state(
            user_message, conversation_history, is_first_turn,
            model, platform, **kwargs,
        )
        policies = _query_policy(state)
        if not policies:
            return ""
        
        instruction = _build_strategy_prompt(policies, state)
        logger.debug(
            "[agentrl-policy] intent=%s policies=%d",
            state["intent"], len(policies),
        )
        return {"context": instruction}
    except Exception as e:
        logger.warning("[agentrl-policy] hook failed (non-fatal): %s", e)
        return ""


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register_hooks() -> None:
    """Called by Hermes plugin loader on startup."""
    try:
        from hermes_cli.plugins import register_hook
        register_hook("pre_llm_call", pre_llm_call)
        logger.info("[agentrl-policy] Registered pre_llm_call hook")
    except ImportError:
        logger.warning("[agentrl-policy] hermes_cli.plugins not available — plugin not loaded")
    except Exception as e:
        logger.warning("[agentrl-policy] Registration failed: %s", e)


# Auto-register when module is imported by Hermes
register_hooks()
