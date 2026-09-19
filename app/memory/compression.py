from __future__ import annotations

import re


class MemoryCompressor:
    """
    Compression component that reduces verbose candidate text while preserving core facts.
    Also provides semantic merging logic for combining redundant memories.
    """

    def __init__(self, max_length: int = 180):
        self.max_length = max_length

    def compress(self, text: str) -> str:
        clean = " ".join(text.split())
        if len(clean) <= self.max_length:
            return clean

        # If verbose, extract core clauses or truncate cleanly at sentence/clause boundary
        # Try cutting at sentence punctuation
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        accumulated = []
        curr_len = 0
        for s in sentences:
            if curr_len + len(s) + 1 <= self.max_length:
                accumulated.append(s)
                curr_len += len(s) + 1
            else:
                break

        if accumulated:
            return " ".join(accumulated)

        # Fallback to clean word boundary cut
        cut = clean[: self.max_length].rsplit(" ", 1)[0]
        return f"{cut}..."

    def merge(self, texts: list[str]) -> str:
        """
        Consolidates redundant facts into a clean merged representation.
        Removes repetitive clauses across sentences.
        """
        seen_clauses: list[str] = []
        lowered_clauses: set[str] = set()

        for text in texts:
            # Normalize whitespace
            norm = " ".join(text.strip().split())
            # Split into simple clauses / sentences
            parts = [p.strip() for p in re.split(r"[;.]", norm) if p.strip()]
            for part in parts:
                key = part.lower()
                # Check for high overlap with already seen clauses
                is_dup = any(key in existing or existing in key for existing in lowered_clauses)
                if not is_dup:
                    seen_clauses.append(part)
                    lowered_clauses.add(key)

        if not seen_clauses:
            return texts[0] if texts else ""

        merged = ". ".join(seen_clauses)
        if not merged.endswith("."):
            merged += "."
        return self.compress(merged)
