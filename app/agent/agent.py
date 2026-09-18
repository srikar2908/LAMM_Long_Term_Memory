from __future__ import annotations

import time

from app.agent.context_builder import ContextBuilder
from app.agent.conversation import ChatResponse
from app.core.config import Settings, get_settings
from app.llm.gemini import GeminiLLMClient, MockLLMClient
from app.memory.manager import MemoryManager


class LammAgent:
    def __init__(self, settings: Settings | None = None, manager: MemoryManager | None = None):
        self.settings = settings or get_settings()
        self.manager = manager or MemoryManager(self.settings)
        self.context_builder = ContextBuilder()
        self.llm = GeminiLLMClient(self.settings.gemini_api_key, self.settings.gemini_model) if self.settings.gemini_available else MockLLMClient()

    def chat(self, message: str, conversation_id: str = "demo") -> ChatResponse:
        start = time.perf_counter()
        retrieved = self.manager.retrieve(message)
        prompt, context_stats = self.context_builder.build(message, retrieved)
        response = self.llm.generate(prompt)
        candidates = self.manager.extract(message, response)
        decisions = []
        for candidate in candidates:
            decision, _ = self.manager.process_candidate(candidate, conversation_id)
            decisions.append(decision)
        elapsed = time.perf_counter() - start
        return ChatResponse(
            response=response,
            retrieved_memories=[{"id": item.memory.id, "text": item.memory.text, "score": item.score} for item in retrieved],
            extracted_memories=[candidate.model_dump() for candidate in candidates],
            lifecycle_decisions=[decision.model_dump(mode="json") for decision in decisions],
            stats={**context_stats, "latency_seconds": elapsed, **self.manager.stats()},
        )
