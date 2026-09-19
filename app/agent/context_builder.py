from __future__ import annotations

import math
import re
from typing import Any

from app.llm.prompts import SYSTEM_PROMPT
from app.memory.schemas import RetrievedMemory


def estimate_tokens(text: str) -> int:
    """
    Provider-agnostic heuristic token estimator based on English word and subword patterns.
    Estimates roughly 1.3 tokens per whitespace-separated word (or ~4 chars per token).
    Clearly labeled as an estimate; never claimed as an exact token count.
    """
    if not text:
        return 0
    words = len(text.split())
    # Subword and punctuation adjustment
    punctuation_count = len(re.findall(r"[^\w\s]", text))
    estimated = int(math.ceil(words * 1.25 + punctuation_count * 0.5))
    return max(1, estimated)


class ContextBuilder:
    """
    Constructs structured LLM prompt context separating:
    1. System Instructions
    2. Relevant Retrieved Long-Term Memories (with similarity scores)
    3. Conversation History
    4. Current User Turn

    Provides provider-aware token counting (exact if Gemini tokenizer is available,
    otherwise clearly labeled as an estimated heuristic).
    """

    def build(
        self,
        user_message: str,
        retrieved: list[RetrievedMemory],
        history: list[tuple[str, str]] | None = None,
        llm_model: Any = None,
    ) -> tuple[str, dict[str, Any]]:
        memory_lines = [
            f"{idx}. {item.memory.text} (similarity: {item.score:.2f})"
            for idx, item in enumerate(retrieved, start=1)
        ]
        history_lines = [f"{role.title()}: {text}" for role, text in (history or [])]

        prompt_sections = [
            f"SYSTEM INSTRUCTIONS:\n{SYSTEM_PROMPT}",
            "RELEVANT LONG-TERM MEMORIES:\n" + ("\n".join(memory_lines) if memory_lines else "None"),
            "CURRENT CONVERSATION:\n" + "\n".join(history_lines + [f"User: {user_message}"]),
            "Assistant:",
        ]
        prompt = "\n\n".join(prompt_sections)

        # Provider-aware token measurement
        token_count = 0
        token_method = "estimated_heuristic"

        if llm_model is not None and hasattr(llm_model, "count_tokens"):
            try:
                res = llm_model.count_tokens(prompt)
                token_count = int(getattr(res, "total_tokens", 0))
                token_method = "exact_gemini_tokenizer"
            except Exception:
                token_count = estimate_tokens(prompt)
                token_method = "estimated_heuristic"
        else:
            token_count = estimate_tokens(prompt)
            token_method = "estimated_heuristic"

        stats = {
            "context_tokens": token_count,
            "token_count_method": token_method,
            "is_token_count_estimated": token_method != "exact_gemini_tokenizer",
            "retrieved_memory_count": len(retrieved),
            "context_char_length": len(prompt),
        }
        return prompt, stats
