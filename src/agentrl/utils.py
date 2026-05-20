from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Optional


def parse_iso(ts: str | float | None) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    ts = str(ts)
    try:
        ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except Exception:
            return None


def extract_files_from_tools(tool_calls: list[dict]) -> tuple[list[str], list[str]]:
    reads, writes = [], []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        path = args.get("path") or args.get("file_path") or args.get("filename", "")
        if name in (
            "read_file", "browser_navigate", "browser_snapshot",
            "search_files", "skill_view", "fetch_url",
        ):
            if path and path not in reads:
                reads.append(path)
        elif name in ("write_file", "patch", "StrReplaceFile"):
            if path and path not in writes:
                writes.append(path)
    return reads, writes


def detect_outcome_from_correction(text: str | None) -> tuple[str, float]:
    if not text:
        return "", 0.0
    text = text.lower()
    markers = [
        "不对", "错了", "不是", "不要", "不应该", "重新", "改一下",
        "不对，", "错了，", "不是这个", "换", "换成", "用另一个",
        "不对,", "错了,", "not this", "wrong", "incorrect",
        "should be", "instead of", "don't use", "use ", "改为",
    ]
    if any(m in text for m in markers) and len(text) > 5:
        return "corrected", 0.7
    return "", 0.0


def has_temporal_keywords(text: str | None) -> bool:
    if not text:
        return False
    text = str(text)
    patterns = [
        r"\b(昨天|前天|今天|明天|上周|下周|这周|本周|上个月|下个月|这个月)\b",
        r"\b(\d{1,2}月\d{1,2}日?|\d{4}-\d{2}-\d{2})\b",
        r"\b(三天前|两天前|一周前|一个月前|几天前)\b",
        r"\b(last week|yesterday|today|tomorrow|last month|next week)\b",
        r"\b(\d+ days? ago|\d+ weeks? ago)\b",
        r"\b(之前|以前|最近|前段时间|早些时候)\b",
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False
