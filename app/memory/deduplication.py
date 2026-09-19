from __future__ import annotations

import re
from app.memory.schemas import RetrievedMemory


def most_redundant(results: list[RetrievedMemory]) -> RetrievedMemory | None:
    if not results:
        return None
    return max(results, key=lambda item: item.score)


def lexical_similarity(left: str, right: str) -> float:
    """Jaccard similarity on content tokens excluding common English stopwords."""
    stopwords = {
        "the", "a", "an", "is", "am", "are", "was", "were", "for", "my",
        "user", "users", "using", "uses", "has", "have", "had", "with",
        "and", "or", "in", "on", "at", "to", "of", "that", "this"
    }
    left_tokens = {re.sub(r"[^\w]", "", token).lower() for token in left.split()} - stopwords - {""}
    right_tokens = {re.sub(r"[^\w]", "", token).lower() for token in right.split()} - stopwords - {""}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def looks_like_update(new_text: str, old_text: str) -> bool:
    """
    Evidence-based UPDATE detection.
    Requires:
    1. Explicit temporal transition / supersession / contradiction indicator.
    2. Significant topical / substantive token overlap between the old and new facts.
    Semantic similarity alone is NOT sufficient for an update.
    """
    new_lower = new_text.lower()
    old_lower = old_text.lower()

    # Explicit supersession / temporal change signals
    update_signals = [
        r"\bnow\b",
        r"\bstarted using\b",
        r"\bswitched to\b",
        r"\bchanged to\b",
        r"\binstead\b",
        r"\bcurrently\b",
        r"\bno longer\b",
        r"\bmoved to\b",
        r"\breplaced\b",
        r"\bupgraded to\b",
        r"\bmigrated to\b",
        r"\bcurrent backend project\b",
        r"\blatest\b",
    ]
    has_update_signal = any(re.search(pat, new_lower) for pat in update_signals)
    if not has_update_signal:
        return False

    # Extract substantive nouns and keywords
    stopwords = {
        "the", "a", "an", "is", "am", "are", "was", "for", "my", "user",
        "users", "using", "uses", "has", "have", "had", "with", "and", "or",
        "in", "on", "at", "to", "of", "that", "this", "now", "started",
        "switched", "changed", "instead", "currently"
    }
    new_words = {re.sub(r"[^\w]", "", w).lower() for w in new_lower.split()} - stopwords - {""}
    old_words = {re.sub(r"[^\w]", "", w).lower() for w in old_lower.split()} - stopwords - {""}

    shared_substantive = new_words & old_words
    # Must share at least one substantive contextual domain term (e.g. "backend", "project", "database", etc.)
    return len(shared_substantive) >= 1
