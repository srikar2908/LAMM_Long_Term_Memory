from __future__ import annotations

from app.memory.manager import MemoryManager


class UnmanagedMemoryBaseline:
    def __init__(self, manager: MemoryManager):
        self.manager = manager

    def ingest(self, user_message: str, conversation_id: str = "baseline") -> list[dict]:
        candidates = self.manager.extract(user_message)
        output = []
        for candidate in candidates:
            decision, record = self.manager.process_candidate_unmanaged(candidate, conversation_id)
            output.append({"decision": decision.model_dump(mode="json"), "memory_id": record.id})
        return output
