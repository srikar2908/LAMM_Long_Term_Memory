from __future__ import annotations

from typing import Any
from app.memory.manager import MemoryManager


class UnmanagedMemoryBaseline:
    """
    UNMANAGED_MEMORY baseline:
    Extracts candidate memories and unconditionally stores every fact.
    Performs no adaptive lifecycle management, deduplication, compression,
    archival, or forgetting.
    Demonstrates unbounded memory growth and accumulation of obsolete/redundant facts.
    """

    def __init__(self, manager: MemoryManager):
        self.manager = manager

    def ingest(self, user_message: str, conversation_id: str = "baseline_unmanaged") -> list[dict[str, Any]]:
        candidates = self.manager.extract(user_message)
        output = []
        for candidate in candidates:
            decision, record = self.manager.process_candidate_unmanaged(candidate, conversation_id)
            output.append({
                "decision": decision.model_dump(mode="json"),
                "memory_id": record.id,
                "text": record.text,
            })
        return output


class RecencyOnlyBaseline:
    """
    RECENCY_ONLY (LRU) baseline:
    Stores candidate memories under a fixed memory budget N.
    When active memory count exceeds N, it evicts the least-recently-accessed (LRU)
    memory purely based on timestamp, without considering relevance, confidence,
    redundancy, or utility signals.
    """

    def __init__(self, manager: MemoryManager, budget: int = 5):
        self.manager = manager
        self.budget = budget

    def ingest(self, user_message: str, conversation_id: str = "baseline_recency") -> list[dict[str, Any]]:
        candidates = self.manager.extract(user_message)
        output = []
        for candidate in candidates:
            decision, record, evicted = self.manager.process_candidate_recency_only(
                candidate=candidate,
                budget=self.budget,
                conversation_id=conversation_id,
            )
            output.append({
                "decision": decision.model_dump(mode="json"),
                "memory_id": record.id,
                "text": record.text,
                "evicted_id": evicted.id if evicted else None,
            })
        return output
