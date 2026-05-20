"""Mine behavioral patterns from session history.

Extracts user preferences, workflow patterns, and project context
from the memory_feedback database.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class UserProfile:
    """Structured representation of user behavioral patterns."""

    # When generated
    generated_at: str = ""
    total_sessions: int = 0
    total_turns: int = 0

    # 1. Preferred workflow patterns
    preferred_workflows: list[dict] = field(default_factory=list)
    # e.g., [{"pattern": "read tests first, then implement", "frequency": 12, "success_rate": 0.9}]

    # 2. Frequently accessed file clusters
    file_clusters: list[dict] = field(default_factory=list)
    # e.g., [{"files": ["README.md", "pyproject.toml"], "project": "agentRL", "frequency": 15}]

    # 3. Correction patterns (what the user frequently corrects)
    correction_patterns: list[dict] = field(default_factory=list)
    # e.g., [{"pattern": "forgets to add dry-run flag", "count": 5, "suggestion": "always suggest dry-run first"}]

    # 4. Coding preferences
    coding_preferences: dict[str, Any] = field(default_factory=dict)
    # e.g., {"prefers_dry_run": true, "likes_type_hints": true, "test_first": false}

    # 5. Tool usage preferences
    tool_preferences: dict[str, Any] = field(default_factory=dict)
    # e.g., {"preferred_search": "search_files over grep", "likes_read_file_over_cat": true}

    # 6. Project-specific knowledge
    project_knowledge: list[dict] = field(default_factory=list)
    # e.g., [{"project": "agentRL", "key_files": ["src/..."], "tech_stack": "Python/SQLite"}]

    # 7. Intent patterns (what user is trying to do)
    intent_patterns: dict[str, Any] = field(default_factory=dict)
    # e.g., {"debug": {"count": 15, "approved_rate": 0.8}, "feature": {...}}

    # 8. Deep failure analysis
    failure_patterns: list[dict] = field(default_factory=list)
    # e.g., [{"pattern": "writes files without reading", "count": 5, "suggestion": "..."}]

    # 9. Success sequences
    success_sequences: list[dict] = field(default_factory=list)
    # e.g., [{"sequence": "Read → Write", "frequency": 20}]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt_block(self) -> str:
        """Format as a prompt-ready memory block for the agent."""
        lines = ["## User Preference Profile\n"]

        if self.coding_preferences:
            lines.append("### Coding Preferences")
            for k, v in self.coding_preferences.items():
                lines.append(f"- {k.replace('_', ' ').title()}: {v}")
            lines.append("")

        if self.preferred_workflows:
            lines.append("### Preferred Workflows")
            for wf in sorted(self.preferred_workflows, key=lambda x: -x.get("frequency", 0))[:5]:
                lines.append(
                    f"- {wf['pattern']} "
                    f"(used {wf.get('frequency', 0)}x, "
                    f"{wf.get('success_rate', 0)*100:.0f}% success)"
                )
            lines.append("")

        if self.correction_patterns:
            lines.append("### Common Corrections (avoid these mistakes)")
            for cp in sorted(self.correction_patterns, key=lambda x: -x.get("count", 0))[:5]:
                lines.append(f"- **{cp['pattern']}** — {cp.get('suggestion', '')}")
            lines.append("")

        if self.file_clusters:
            lines.append("### Frequently Accessed File Groups")
            for fc in sorted(self.file_clusters, key=lambda x: -x.get("frequency", 0))[:5]:
                files = ", ".join(fc.get("files", [])[:4])
                proj = fc.get("project", "")
                lines.append(f"- [{proj}] {files} ({fc.get('frequency', 0)}x)")
            lines.append("")

        if self.project_knowledge:
            lines.append("### Project Context")
            for pk in self.project_knowledge[:5]:
                proj = pk.get("project", "")
                stack = pk.get("tech_stack", "")
                files = ", ".join(pk.get("key_files", [])[:5])
                lines.append(f"- **{proj}** ({stack})")
                if files:
                    lines.append(f"  Key files: {files}")
            lines.append("")

        return "\n".join(lines)


class PatternMiner:
    """Mine patterns from memory_feedback database."""

    def __init__(self, db_path: str | Path):
        self.db_path = db_path

    def mine(self) -> UserProfile:
        """Run all pattern mining and return a UserProfile."""
        profile = UserProfile()
        profile.generated_at = datetime.now().isoformat()
        profile.total_sessions, profile.total_turns = self._count_stats()

        profile.coding_preferences = self._mine_coding_preferences()
        profile.preferred_workflows = self._mine_workflows()
        profile.correction_patterns = self._mine_corrections()
        profile.file_clusters = self._mine_file_clusters()
        profile.project_knowledge = self._mine_project_knowledge()
        profile.intent_patterns = self._mine_intent_patterns()
        profile.failure_patterns = self._mine_failure_patterns()
        profile.success_sequences = self._mine_success_sequences()
        profile.tool_preferences = self._mine_tool_preferences()

        return profile

    def _query(self, sql: str, params: tuple = ()) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()
        return rows

    def _count_stats(self) -> tuple[int, int]:
        rows = self._query("SELECT COUNT(DISTINCT session_id), COUNT(*) FROM memory_feedback")
        return rows[0][0], rows[0][1]

    def _mine_coding_preferences(self) -> dict[str, Any]:
        """Extract coding style preferences from queries and corrections."""
        prefs = {}

        # Check for dry-run mentions
        rows = self._query(
            "SELECT COUNT(*) FROM memory_feedback WHERE query LIKE '%dry-run%' OR query LIKE '%dry run%'"
        )
        prefs["prefers_dry_run"] = rows[0][0] > 5

        # Check for test-related requests
        rows = self._query(
            "SELECT COUNT(*) FROM memory_feedback WHERE query LIKE '%test%'"
        )
        prefs["frequently_asks_for_tests"] = rows[0][0] > 10

        # Check for type hints / typing
        rows = self._query(
            "SELECT COUNT(*) FROM memory_feedback WHERE query LIKE '%type hint%' OR query LIKE '%typing%'"
        )
        prefs["cares_about_type_hints"] = rows[0][0] > 3

        # Check for documentation requests
        rows = self._query(
            "SELECT COUNT(*) FROM memory_feedback WHERE query LIKE '%doc%' OR query LIKE '%README%'"
        )
        prefs["frequently_updates_docs"] = rows[0][0] > 5

        # Check for refactor requests
        rows = self._query(
            "SELECT COUNT(*) FROM memory_feedback WHERE query LIKE '%refactor%'"
        )
        prefs["frequently_refactors"] = rows[0][0] > 3

        # Preferred approach: incremental vs rewrite
        rows = self._query(
            """SELECT user_outcome, COUNT(*) FROM memory_feedback
               WHERE query LIKE '%improve%' OR query LIKE '%改进%' GROUP BY user_outcome"""
        )
        # Just a signal, not a strict preference

        return prefs

    def _mine_workflows(self) -> list[dict]:
        """Mine successful workflow patterns from approved sessions."""
        workflows = []

        # Pattern 1: "reads tests before modifying code"
        rows = self._query(
            """SELECT COUNT(*) FROM memory_feedback
               WHERE user_outcome = 'approved'
               AND (files_read LIKE '%test%' OR files_read LIKE '%Test%')
               AND (files_written LIKE '%.py%' OR assistant_response LIKE '%def %')"""
        )
        count = rows[0][0]
        if count > 2:
            workflows.append({
                "pattern": "Read existing tests before writing/modifying code",
                "frequency": count,
                "success_rate": self._success_rate_of_pattern("files_read LIKE '%test%'"),
            })

        # Pattern 2: "checks README/docs before implementing"
        rows = self._query(
            """SELECT COUNT(*) FROM memory_feedback
               WHERE user_outcome = 'approved'
               AND (files_read LIKE '%README%' OR files_read LIKE '%.md%')"""
        )
        count = rows[0][0]
        if count > 2:
            workflows.append({
                "pattern": "Review documentation (README, .md) before implementation",
                "frequency": count,
                "success_rate": self._success_rate_of_pattern("files_read LIKE '%README%'"),
            })

        # Pattern 3: "reads config files early"
        rows = self._query(
            """SELECT COUNT(*) FROM memory_feedback
               WHERE user_outcome = 'approved'
               AND (files_read LIKE '%pyproject.toml%' OR files_read LIKE '%package.json%'
                    OR files_read LIKE '%config%')"""
        )
        count = rows[0][0]
        if count > 2:
            workflows.append({
                "pattern": "Read project config files early in the session",
                "frequency": count,
                "success_rate": self._success_rate_of_pattern(
                    "files_read LIKE '%pyproject.toml%' OR files_read LIKE '%config%'"
                ),
            })

        return workflows

    def _success_rate_of_pattern(self, where_clause: str) -> float:
        """Compute success rate (approved / total) for sessions matching a pattern."""
        rows = self._query(
            f"""SELECT user_outcome, COUNT(*) FROM memory_feedback WHERE {where_clause} GROUP BY user_outcome"""
        )
        total = sum(r[1] for r in rows)
        approved = sum(r[1] for r in rows if r[0] == "approved")
        return approved / total if total > 0 else 0.0

    def _mine_corrections(self) -> list[dict]:
        """Extract common correction patterns from user feedback."""
        # Get all queries from corrected/exited sessions
        rows = self._query(
            """SELECT query, user_outcome, correction_text FROM memory_feedback
               WHERE user_outcome IN ('corrected', 'exited', 'rejected')
               AND (query IS NOT NULL OR correction_text IS NOT NULL)"""
        )

        patterns = []

        # Pattern: user mentions specific files that were missed
        missed_file = sum(
            1 for r in rows
            if r["correction_text"] and any(k in (r["correction_text"] or "").lower() for k in ["文件", "path", "file", "应该看"])
        )
        if missed_file > 2:
            patterns.append({
                "pattern": "Agent misses relevant files that user expects it to read",
                "count": missed_file,
                "suggestion": "Proactively identify and read all related files before making changes",
            })

        # Pattern: user corrects implementation details
        impl_corr = sum(
            1 for r in rows
            if r["query"] and any(k in (r["query"] or "").lower() for k in ["不对", "错了", "不是", "不要", "改"])
        )
        if impl_corr > 2:
            patterns.append({
                "pattern": "User frequently corrects implementation details",
                "count": impl_corr,
                "suggestion": "Ask for confirmation before making non-trivial changes",
            })

        # Pattern: user mentions temporal issues
        temporal_corr = sum(
            1 for r in rows
            if r["query"] and any(k in (r["query"] or "").lower() for k in ["昨天", "时间", "日期"])
        )
        if temporal_corr > 2:
            patterns.append({
                "pattern": "Agent struggles with temporal/time-based queries",
                "count": temporal_corr,
                "suggestion": "Pay extra attention to time references in user queries",
            })

        return patterns

    def _mine_file_clusters(self) -> list[dict]:
        """Find files that are frequently accessed together."""
        rows = self._query(
            "SELECT session_id, files_read FROM memory_feedback WHERE files_read IS NOT NULL AND files_read != '[]'"
        )

        # Build co-occurrence within sessions
        session_files: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            try:
                files = json.loads(row["files_read"])
                if isinstance(files, list) and len(files) > 1:
                    session_files[row["session_id"]].extend(files)
            except Exception:
                continue

        # Count pairwise co-occurrence
        pair_counts: Counter = Counter()
        for files in session_files.values():
            unique = sorted(set(files))
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    pair_counts[(unique[i], unique[j])] += 1

        # Extract clusters (files that appear together >= 3 times)
        cluster_map: dict[str, set[str]] = {}
        for (f1, f2), count in pair_counts.most_common(30):
            if count < 2:
                continue
            if f1 not in cluster_map:
                cluster_map[f1] = {f1}
            if f2 not in cluster_map:
                cluster_map[f2] = {f2}
            cluster_map[f1].add(f2)
            cluster_map[f2].add(f1)

        # Convert to list
        seen = set()
        clusters = []
        for f, members in cluster_map.items():
            if f in seen:
                continue
            cluster = sorted(members)
            seen.update(cluster)
            # Try to identify project from file paths
            project = self._infer_project(cluster[0]) if cluster else ""
            clusters.append({
                "files": cluster[:6],  # cap size
                "project": project,
                "frequency": pair_counts.get((cluster[0], cluster[1] if len(cluster) > 1 else cluster[0]), 1),
            })

        return clusters[:10]

    def _infer_project(self, filepath: str) -> str:
        """Extract project name from a file path."""
        path = Path(filepath.replace("~", str(Path.home())))
        parts = path.parts
        # Look for known project root indicators
        for i, part in enumerate(parts):
            if part in ("PycharmProjects", "Projects", "workspace", "src"):
                if i + 1 < len(parts):
                    return parts[i + 1]
        # Fallback: use directory name
        if len(parts) >= 2:
            return parts[-2]
        return "unknown"

    def _mine_project_knowledge(self) -> list[dict]:
        """Extract per-project context."""
        rows = self._query(
            """SELECT query, files_read, files_written, user_outcome, created_at
               FROM memory_feedback WHERE query IS NOT NULL"""
        )

        project_data: dict[str, dict] = defaultdict(lambda: {
            "queries": [], "files_read": Counter(), "files_written": Counter(),
            "outcomes": Counter(), "tech_keywords": Counter(),
        })

        for row in rows:
            # Infer project from query or files
            project = self._infer_project_from_row(row)
            if not project or project == "unknown":
                continue

            pd = project_data[project]
            pd["queries"].append(row["query"])
            pd["outcomes"][row["user_outcome"]] += 1

            try:
                for f in json.loads(row["files_read"] or "[]"):
                    pd["files_read"][f] += 1
            except Exception:
                pass
            try:
                for f in json.loads(row["files_written"] or "[]"):
                    pd["files_written"][f] += 1
            except Exception:
                pass

            # Extract tech stack from queries
            query = (row["query"] or "").lower()
            for tech in ["python", "javascript", "typescript", "react", "fastapi", "django",
                         "sqlite", "postgres", "docker", "kubernetes", "rust", "go"]:
                if tech in query:
                    pd["tech_keywords"][tech] += 1

        results = []
        for project, data in sorted(project_data.items(), key=lambda x: -sum(x[1]["outcomes"].values()))[:10]:
            if sum(data["outcomes"].values()) < 3:
                continue

            top_files = [f for f, c in data["files_read"].most_common(5)]
            top_tech = [t for t, c in data["tech_keywords"].most_common(3)]
            success_rate = data["outcomes"].get("approved", 0) / sum(data["outcomes"].values())

            results.append({
                "project": project,
                "key_files": top_files,
                "tech_stack": ", ".join(top_tech) if top_tech else "unknown",
                "total_sessions": sum(data["outcomes"].values()),
                "success_rate": round(success_rate, 2),
            })

        return results

    def _infer_project_from_row(self, row: sqlite3.Row) -> str:
        """Infer project name from a DB row."""
        # Try files_read first
        try:
            files = json.loads(row["files_read"] or "[]")
            for f in files:
                proj = self._infer_project(f)
                if proj and proj != "unknown":
                    return proj
        except Exception:
            pass

        # Try query text for project references
        query = row["query"] or ""
        m = re.search(r"PycharmProjects/([^/\s]+)", query)
        if m:
            return m.group(1)
        m = re.search(r"folder\s+\./([^/\s]+)", query, re.IGNORECASE)
        if m:
            return m.group(1)

        return "unknown"

    def _mine_intent_patterns(self) -> dict[str, Any]:
        """Extract query intent patterns (what the user is trying to do)."""
        rows = self._query(
            "SELECT query, user_outcome FROM memory_feedback WHERE query IS NOT NULL"
        )

        intent_patterns = {
            "debug": ["bug", "fix", "error", "crash", "broken", "失败", "报错", "不对"],
            "feature": ["add", "implement", "create", "new", "增加", "实现", "添加"],
            "refactor": ["refactor", "reorganize", "clean up", "重构", "整理"],
            "review": ["review", "check", "look at", "看看", "检查"],
            "explain": ["explain", "how does", "what is", "为什么", "怎么回事"],
            "test": ["test", "pytest", "unittest", "测试"],
            "doc": ["doc", "readme", "comment", "文档", "注释"],
            "deploy": ["deploy", "release", "build", "docker", "部署", "发布"],
        }

        intent_stats: dict[str, dict] = {}
        for intent, keywords in intent_patterns.items():
            matches = []
            for row in rows:
                q = (row["query"] or "").lower()
                if any(k in q for k in keywords):
                    matches.append(row["user_outcome"])

            if len(matches) >= 2:
                total = len(matches)
                approved = matches.count("approved")
                exited = matches.count("exited")
                corrected = matches.count("corrected")
                intent_stats[intent] = {
                    "count": total,
                    "approved_rate": round(approved / total, 2),
                    "exited_rate": round(exited / total, 2),
                    "corrected_rate": round(corrected / total, 2),
                }

        return intent_stats

    def _mine_failure_patterns(self) -> list[dict]:
        """Extract deep failure patterns from exited/corrected sessions."""
        rows = self._query(
            """SELECT query, assistant_response, files_read, files_written, user_outcome
               FROM memory_feedback WHERE user_outcome IN ('exited', 'corrected')"""
        )

        patterns = []

        # Pattern: Agent wrote files but user exited (possibly wrong files)
        wrote_then_exited = sum(
            1 for r in rows
            if r["user_outcome"] == "exited" and r["files_written"] and r["files_written"] != "[]"
        )
        if wrote_then_exited > 2:
            patterns.append({
                "pattern": "Agent writes files but user abandons session",
                "count": wrote_then_exited,
                "insight": "Agent may be modifying wrong files or making unwanted changes",
                "suggestion": "Always confirm file selection and preview changes before writing",
            })

        # Pattern: Long response but bad outcome (rambling or off-target)
        long_bad = sum(
            1 for r in rows
            if r["user_outcome"] in ("exited", "corrected")
            and r["assistant_response"]
            and len(r["assistant_response"]) > 2000
        )
        total_bad = sum(1 for r in rows if r["user_outcome"] in ("exited", "corrected"))
        if total_bad > 0 and long_bad / total_bad > 0.3:
            patterns.append({
                "pattern": "Long responses correlate with bad outcomes",
                "count": long_bad,
                "insight": f"{long_bad/total_bad*100:.0f}% of bad sessions have very long responses",
                "suggestion": "Be concise. If response needs to be long, break it into steps and confirm each",
            })

        # Pattern: No files read but user asked for code changes (flying blind)
        blind_changes = sum(
            1 for r in rows
            if r["user_outcome"] in ("exited", "corrected")
            and (not r["files_read"] or r["files_read"] == "[]")
            and any(k in (r["query"] or "").lower() for k in ["fix", "add", "change", "改", "修"])
        )
        if blind_changes > 2:
            patterns.append({
                "pattern": "Agent changes code without reading relevant files first",
                "count": blind_changes,
                "insight": "Making code changes without reading context leads to errors",
                "suggestion": "Always read existing code before proposing modifications",
            })

        return patterns

    def _mine_success_sequences(self) -> list[dict]:
        """Extract action sequences that lead to approved outcomes."""
        rows = self._query(
            """SELECT session_id, query, files_read, files_written, user_outcome, created_at
               FROM memory_feedback WHERE user_outcome = 'approved' ORDER BY session_id, created_at"""
        )

        # Group by session to see sequences
        sessions: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            sessions[row["session_id"]].append({
                "query": row["query"],
                "read": bool(row["files_read"] and row["files_read"] != "[]"),
                "wrote": bool(row["files_written"] and row["files_written"] != "[]"),
            })

        sequences = []

        # Pattern: Read → Write → Approved (the classic)
        read_then_write = sum(
            1 for turns in sessions.values()
            if len(turns) >= 2
            and any(t["read"] for t in turns[:-1])
            and any(t["wrote"] for t in turns)
        )
        if read_then_write > 2:
            sequences.append({
                "sequence": "Read files → Understand → Write files",
                "frequency": read_then_write,
                "description": "The most reliable path: read before writing",
            })

        # Pattern: Question → Explanation → Approved (no file changes)
        explain_only = sum(
            1 for turns in sessions.values()
            if len(turns) == 1
            and not turns[0]["wrote"]
            and not turns[0]["read"]
        )
        if explain_only > 2:
            sequences.append({
                "sequence": "Answer question directly (no file changes)",
                "frequency": explain_only,
                "description": "Simple explanations without touching files often succeed",
            })

        return sequences

    def _mine_tool_preferences(self) -> dict[str, Any]:
        """Extract tool usage patterns."""
        rows = self._query(
            "SELECT tool_calls, user_outcome FROM memory_feedback WHERE tool_calls IS NOT NULL AND tool_calls != '[]'"
        )

        tool_counts: Counter = Counter()
        tool_outcomes: dict[str, Counter] = defaultdict(Counter)

        for row in rows:
            try:
                tcs = json.loads(row["tool_calls"])
                if not isinstance(tcs, list):
                    continue
                for tc in tcs:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    if name:
                        tool_counts[name] += 1
                        tool_outcomes[name][row["user_outcome"]] += 1
            except Exception:
                continue

        if not tool_counts:
            return {}

        # Find tools with best success rates
        tool_success = {}
        for tool, outcomes in tool_outcomes.items():
            total = sum(outcomes.values())
            if total >= 3:
                tool_success[tool] = round(outcomes.get("approved", 0) / total, 2)

        return {
            "most_used_tools": dict(tool_counts.most_common(10)),
            "tool_success_rates": dict(sorted(tool_success.items(), key=lambda x: -x[1])[:5]),
        }
