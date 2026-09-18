from __future__ import annotations

from app.llm.prompts import SYSTEM_PROMPT
from app.memory.schemas import RetrievedMemory


def approximate_tokens(text: str) -> int:
    return max(1, len(text.split()))


class ContextBuilder:
    def build(self, user_message: str, retrieved: list[RetrievedMemory], history: list[tuple[str, str]] | None = None) -> tuple[str, dict]:
        memory_lines = [f"{idx}. {item.memory.text}" for idx, item in enumerate(retrieved, start=1)]
        history_lines = [f"{role.title()}: {text}" for role, text in (history or [])]
        prompt = "\n\n".join(
            [
                f"SYSTEM INSTRUCTIONS:\n{SYSTEM_PROMPT}",
                "RELEVANT LONG-TERM MEMORIES:\n" + ("\n".join(memory_lines) if memory_lines else "None"),
                "CURRENT CONVERSATION:\n" + "\n".join(history_lines + [f"User: {user_message}"]),
                "Assistant:",
            ]
        )
        return prompt, {"approx_context_tokens": approximate_tokens(prompt), "retrieved_memory_count": len(retrieved)}
