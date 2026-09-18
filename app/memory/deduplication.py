from __future__ import annotations

from app.memory.schemas import RetrievedMemory


def most_redundant(results: list[RetrievedMemory]) -> RetrievedMemory | None:
    if not results:
        return None
    return max(results, key=lambda item: item.score)


def lexical_similarity(left: str, right: str) -> float:
    stopwords = {"the", "a", "an", "is", "am", "are", "for", "my", "user", "users", "using", "uses", "has"}
    left_tokens = {token.strip(".,'").lower() for token in left.split()} - stopwords
    right_tokens = {token.strip(".,'").lower() for token in right.split()} - stopwords
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def looks_like_update(new_text: str, old_text: str) -> bool:
    new_lower = new_text.lower()
    old_lower = old_text.lower()
    update_terms = {"currently", "started", "now", "instead", "changed", "switched", "current"}
    topic_terms = {"backend", "project", "language", "python", "java", "programming"}
    has_update_signal = any(term in new_lower for term in update_terms)
    shared_topic = bool(set(new_lower.split()) & set(old_lower.split()) & topic_terms)
    return has_update_signal and shared_topic
