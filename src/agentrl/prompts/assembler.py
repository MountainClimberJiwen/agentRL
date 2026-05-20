"""PromptAssembler: build the final prompt messages sent to the frozen LLM."""

from __future__ import annotations

from agentrl.prompts.registry import SelectedEvidence


class PromptAssembler:
    """Assemble the final prompt messages sent to the frozen LLM."""

    def __init__(self, inject_user_memory: bool = True):
        self.inject_user_memory = inject_user_memory

    def assemble(
        self,
        user_query: str,
        system_prompt: str | None = None,
        memory_context: str = "",
        extra_context: str = "",
        user_memory_block: str = "",
    ) -> list[dict[str, str]]:
        """Build OpenAI-style messages list.

        Order: system -> user memory -> retrieved memory context -> user query
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Inject user preference memory (learned patterns)
        if self.inject_user_memory and user_memory_block:
            messages.append({"role": "system", "content": user_memory_block})

        if memory_context:
            messages.append({"role": "system", "content": memory_context})

        if extra_context:
            user_query = f"{extra_context}\n\n{user_query}"

        messages.append({"role": "user", "content": user_query})
        return messages

    def build_memory_context(
        self,
        selected: list[SelectedEvidence],
        session_lookup: dict[str, dict],
    ) -> str:
        """Format selected evidence into a memory block for the prompt."""
        blocks = ["## Retrieved Memory Context\n"]
        for ev in selected:
            sess = session_lookup.get(ev.session_id, {})
            if not sess:
                continue
            ts = sess.get("created_at", "unknown")
            outcome = sess.get("user_outcome", "unknown")
            query = sess.get("query", "")
            response = sess.get("assistant_response", "")
            blocks.append(
                f"### Session {ev.session_id[:8]}... (turn {ev.turn_id[:8]}...)\n"
                f"- Time: {ts}\n"
                f"- Outcome: {outcome}\n"
                f"- Relevance: {ev.relevance_score:.2f} ({ev.reason})\n"
                f"- Query: {query[:200]}{'...' if len(query or '') > 200 else ''}\n"
                f"- Response: {response[:300]}{'...' if len(response or '') > 300 else ''}\n"
            )
        return "\n".join(blocks)
