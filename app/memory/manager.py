from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.core.config import Settings, get_settings
from app.embeddings.sentence_transformer import build_embedding_provider
from app.memory.compression import MemoryCompressor
from app.memory.extraction import DeterministicMemoryExtractor, GeminiMemoryExtractor, MemoryExtractor
from app.memory.lifecycle import LammLifecycleController, unmanaged_decision
from app.memory.schemas import CandidateMemory, LifecycleDecision, LifecycleOperation, LifecycleStatus, MemoryRecord, RetrievedMemory
from app.memory.scoring import recency_score
from app.retrieval.faiss_store import FaissVectorStore
from app.storage.database import connect
from app.storage.repositories import LifecycleEventRepository, MemoryRepository


class MemoryManager:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.conn = connect(self.settings.sqlite_path)
        self.embedder = build_embedding_provider(self.settings.embedding_model, self.settings.embedding_dimension)
        self.memories = MemoryRepository(self.conn, self.embedder.dimension)
        self.events = LifecycleEventRepository(self.conn)
        self.vector_store = FaissVectorStore(self.settings.faiss_index_path, self.embedder.dimension)
        self.controller = LammLifecycleController(self.settings)
        self.compressor = MemoryCompressor(self.settings.compress_length)
        self.extractor: MemoryExtractor = (
            GeminiMemoryExtractor(self.settings.gemini_api_key, self.settings.gemini_model)
            if self.settings.gemini_available
            else DeterministicMemoryExtractor()
        )
        self.rebuild_index()

    def extract(self, user_message: str, assistant_message: str | None = None) -> list[CandidateMemory]:
        return self.extractor.extract(user_message, assistant_message)

    def retrieve(self, query: str, top_k: int | None = None, include_archived: bool = False) -> list[RetrievedMemory]:
        query_embedding = self.embedder.embed_one(query)
        raw = self.vector_store.search(query_embedding, top_k or self.settings.top_k)
        results: list[RetrievedMemory] = []
        for memory_id, raw_score in raw:
            memory = self.memories.get(memory_id)
            if not memory:
                continue
            if memory.lifecycle_status != LifecycleStatus.ACTIVE and not include_archived:
                continue
            normalized_score = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))
            memory.recency_score = recency_score(memory.updated_at, self.settings.recency_decay)
            self.memories.mark_accessed(memory_id, normalized_score)
            results.append(RetrievedMemory(memory=memory, score=normalized_score))
        return results

    def process_candidate(self, candidate: CandidateMemory, conversation_id: str | None = None) -> tuple[LifecycleDecision, MemoryRecord | None]:
        embedding = self.embedder.embed_one(candidate.text)
        similar = self.retrieve(candidate.text, top_k=self.settings.top_k)
        active_count = len(self.memories.list(include_inactive=False))
        decision = self.controller.decide(candidate, similar, active_count)
        record = self.apply_decision(candidate, decision, embedding, conversation_id)
        self.events.record(record.id if record else None, decision)
        return decision, record

    def process_candidate_unmanaged(self, candidate: CandidateMemory, conversation_id: str | None = None) -> tuple[LifecycleDecision, MemoryRecord]:
        decision = unmanaged_decision(candidate)
        embedding = self.embedder.embed_one(candidate.text)
        record = self._create_memory(candidate.text, candidate.confidence_score, conversation_id, candidate.source, embedding)
        self.events.record(record.id, decision)
        return decision, record

    def apply_decision(self, candidate: CandidateMemory, decision: LifecycleDecision, embedding, conversation_id: str | None) -> MemoryRecord | None:
        op = decision.operation
        if op == LifecycleOperation.FORGET:
            return self._create_memory(
                candidate.text,
                candidate.confidence_score,
                conversation_id,
                candidate.source,
                embedding,
                LifecycleStatus.FORGOTTEN,
                False,
            )
        if op == LifecycleOperation.UPDATE and decision.affected_memory_ids:
            existing = self.memories.get(decision.affected_memory_ids[0])
            if existing:
                existing.text = candidate.text
                existing.updated_at = datetime.now(timezone.utc)
                existing.confidence_score = max(existing.confidence_score, candidate.confidence_score)
                existing.lifecycle_status = LifecycleStatus.ACTIVE
                existing.metadata["update_reason"] = decision.reason
                self.memories.update(existing, embedding)
                self.rebuild_index()
                return existing
        if op == LifecycleOperation.MERGE and decision.affected_memory_ids:
            existing = self.memories.get(decision.affected_memory_ids[0])
            if existing:
                merged_text = self.compressor.merge([existing.text, candidate.text])
                existing.text = merged_text
                existing.updated_at = datetime.now(timezone.utc)
                existing.lifecycle_status = LifecycleStatus.ACTIVE
                existing.metadata["merge_reason"] = decision.reason
                new_embedding = self.embedder.embed_one(merged_text)
                self.memories.update(existing, new_embedding)
                self.rebuild_index()
                return existing

        text = candidate.text
        status = LifecycleStatus.ACTIVE
        archived = False
        compressed = None
        if op == LifecycleOperation.COMPRESS:
            compressed = self.compressor.compress(text)
            text = compressed
            status = LifecycleStatus.COMPRESSED
        elif op == LifecycleOperation.ARCHIVE:
            archived = True
            status = LifecycleStatus.ARCHIVED
        record = self._create_memory(text, candidate.confidence_score, conversation_id, candidate.source, embedding, status, archived, compressed)
        if record.lifecycle_status == LifecycleStatus.COMPRESSED:
            self.vector_store.add(record.id, embedding)
        return record

    def _create_memory(
        self,
        text: str,
        confidence: float,
        conversation_id: str | None,
        source: str,
        embedding,
        status: LifecycleStatus = LifecycleStatus.ACTIVE,
        archived: bool = False,
        compressed_text: str | None = None,
    ) -> MemoryRecord:
        now = datetime.now(timezone.utc)
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            text=text,
            created_at=now,
            updated_at=now,
            confidence_score=confidence,
            lifecycle_status=status,
            source=source,
            conversation_id=conversation_id,
            archived=archived,
            compressed_text=compressed_text,
        )
        self.memories.create(record, embedding)
        if status == LifecycleStatus.ACTIVE:
            self.vector_store.add(record.id, embedding)
        return record

    def forget(self, memory_id: str) -> bool:
        memory = self.memories.get(memory_id)
        if not memory:
            return False
        memory.lifecycle_status = LifecycleStatus.FORGOTTEN
        memory.archived = False
        memory.updated_at = datetime.now(timezone.utc)
        self.memories.update(memory)
        self.rebuild_index()
        return True

    def rebuild_index(self) -> None:
        self.vector_store.rebuild(self.memories.active_embeddings())

    def stats(self) -> dict:
        all_memories = self.memories.list()
        return {
            "total": len(all_memories),
            "active": sum(m.lifecycle_status == LifecycleStatus.ACTIVE for m in all_memories),
            "archived": sum(m.lifecycle_status == LifecycleStatus.ARCHIVED for m in all_memories),
            "forgotten": sum(m.lifecycle_status == LifecycleStatus.FORGOTTEN for m in all_memories),
            "compressed": sum(m.lifecycle_status == LifecycleStatus.COMPRESSED for m in all_memories),
            "merged": sum("merge_reason" in m.metadata for m in all_memories),
        }
