from __future__ import annotations

import time
from typing import Any

from app.agent.context_builder import ContextBuilder
from app.agent.conversation import ChatResponse
from app.core.config import Settings, get_settings
from app.llm.gemini import GeminiLLMClient, MockLLMClient
from app.memory.manager import MemoryManager


class LammAgent:
    """
    End-to-End LLM Agent with external LAMM memory lifecycle layer.
    Workflow:
        1. Receive user message
        2. Semantic retrieval of top-k active memories via FAISS
        3. Structured context construction with provider-aware token stats
        4. Response generation via LLM (or deterministic mock)
        5. Extraction of candidate memories
        6. LAMM scoring and lifecycle decision (Update, Merge, Compress, Forget, Archive, Retain)
        7. Synchronization of SQLite records and FAISS vector index
        8. Return response, retrieved memories, lifecycle decisions, and latency metrics
    """

    def __init__(self, settings: Settings | None = None, manager: MemoryManager | None = None):
        self.settings = settings or get_settings()
        self.manager = manager or MemoryManager(self.settings)
        self.context_builder = ContextBuilder()
        self.llm = (
            GeminiLLMClient(self.settings.gemini_api_key, self.settings.gemini_model)
            if self.settings.gemini_available
            else MockLLMClient()
        )

    def chat(
        self,
        message: str,
        conversation_id: str = "demo",
        history: list[tuple[str, str]] | None = None,
    ) -> ChatResponse:
        total_start = time.perf_counter()

        # 1. Retrieval
        retrieval_start = time.perf_counter()
        retrieved = self.manager.retrieve(message)
        retrieval_latency = time.perf_counter() - retrieval_start

        # 2. Context Construction & Token Estimation
        model_obj = getattr(self.llm, "model", None)
        prompt, context_stats = self.context_builder.build(
            user_message=message,
            retrieved=retrieved,
            history=history,
            llm_model=model_obj,
        )

        # 3. LLM Response Generation
        gen_start = time.perf_counter()
        response = self.llm.generate(prompt)
        generation_latency = time.perf_counter() - gen_start

        # 4. Memory Extraction
        extract_start = time.perf_counter()
        candidates = self.manager.extract(message, response)
        extraction_latency = time.perf_counter() - extract_start

        # 5. Lifecycle Management (LAMM decisions & persistence)
        mgmt_start = time.perf_counter()
        decisions = []
        for candidate in candidates:
            decision, _ = self.manager.process_candidate(candidate, conversation_id)
            decisions.append(decision)
        mgmt_latency = time.perf_counter() - mgmt_start

        total_latency = time.perf_counter() - total_start

        latency_breakdown = {
            "retrieval_latency_seconds": retrieval_latency,
            "generation_latency_seconds": generation_latency,
            "extraction_latency_seconds": extraction_latency,
            "memory_management_latency_seconds": mgmt_latency,
            "total_latency_seconds": total_latency,
        }

        return ChatResponse(
            response=response,
            retrieved_memories=[
                {"id": item.memory.id, "text": item.memory.text, "score": item.score}
                for item in retrieved
            ],
            extracted_memories=[candidate.model_dump() for candidate in candidates],
            lifecycle_decisions=[decision.model_dump(mode="json") for decision in decisions],
            stats={
                **context_stats,
                **latency_breakdown,
                **self.manager.stats(),
            },
        )
