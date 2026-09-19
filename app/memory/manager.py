from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.config import Settings, get_settings
from app.embeddings.sentence_transformer import build_embedding_provider
from app.memory.compression import MemoryCompressor
from app.memory.extraction import (
    DeterministicMemoryExtractor,
    GeminiMemoryExtractor,
    MemoryExtractor,
)
from app.memory.lifecycle import LammLifecycleController, unmanaged_decision
from app.memory.schemas import (
    CandidateMemory,
    LifecycleDecision,
    LifecycleOperation,
    LifecycleStatus,
    MemoryRecord,
    RetrievedMemory,
)
from app.memory.scoring import recency_score
from app.retrieval.faiss_store import FaissVectorStore
from app.storage.database import connect
from app.storage.repositories import LifecycleEventRepository, MemoryRepository

logger = logging.getLogger("lamm.manager")


class MemoryManager:
    """
    Core Memory Management Layer.
    Orchestrates extraction, scoring, lifecycle decisions, FAISS retrieval,
    and SQLite persistence.

    Architectural Invariant:
    SQLite is the canonical source of truth for all memory records and metadata.
    FAISS is a derived retrieval index containing exclusively ACTIVE memories.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.conn = connect(self.settings.sqlite_path)
        self.embedder = build_embedding_provider(
            self.settings.embedding_provider,
            self.settings.embedding_model,
            self.settings.embedding_dimension,
        )
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
        # Rebuild / synchronize index on initialization
        self.rebuild_index()

    def extract(self, user_message: str, assistant_message: str | None = None) -> list[CandidateMemory]:
        return self.extractor.extract(user_message, assistant_message)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        include_archived: bool = False,
    ) -> list[RetrievedMemory]:
        k = top_k or self.settings.top_k
        query_embedding = self.embedder.embed_one(query)
        raw_results = self.vector_store.search(query_embedding, k)

        results: list[RetrievedMemory] = []
        for memory_id, raw_score in raw_results:
            memory = self.memories.get(memory_id)
            if not memory:
                continue
            if memory.lifecycle_status != LifecycleStatus.ACTIVE and not include_archived:
                continue

            # Normalized cosine similarity to [0, 1]
            normalized_score = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))
            memory.recency_score = recency_score(memory.updated_at, self.settings.recency_decay)

            if not self.settings.dry_run:
                self.memories.mark_accessed(memory_id, normalized_score)

            results.append(RetrievedMemory(memory=memory, score=normalized_score))
        return results

    def process_candidate(
        self,
        candidate: CandidateMemory,
        conversation_id: str | None = None,
        dry_run: bool | None = None,
    ) -> tuple[LifecycleDecision, MemoryRecord | None]:
        is_dry_run = self.settings.dry_run if dry_run is None else dry_run
        embedding = self.embedder.embed_one(candidate.text)
        similar = self.retrieve(candidate.text, top_k=self.settings.top_k)
        active_count = len(self.memories.list(include_inactive=False))

        decision = self.controller.decide(candidate, similar, active_count)

        if is_dry_run:
            simulated = MemoryRecord(
                id=f"dry_run_{uuid.uuid4().hex[:8]}",
                text=candidate.text,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                confidence_score=candidate.confidence_score,
                lifecycle_status=LifecycleStatus.ACTIVE,
                metadata={"dry_run": True},
            )
            return decision, simulated

        record = self.apply_decision(candidate, decision, embedding, conversation_id)
        thresholds = {
            "redundancy_threshold": self.settings.redundancy_threshold,
            "archive_threshold": self.settings.archive_threshold,
            "forget_threshold": self.settings.forget_threshold,
        }
        self.events.record(
            memory_id=record.id if record else None,
            decision=decision,
            before_state=decision.metadata.get("before_state"),
            after_state=record.model_dump(mode="json") if record else None,
            thresholds=thresholds,
        )
        return decision, record

    def process_candidate_unmanaged(
        self,
        candidate: CandidateMemory,
        conversation_id: str | None = None,
    ) -> tuple[LifecycleDecision, MemoryRecord]:
        """Unmanaged baseline: store every candidate without eviction."""
        decision = unmanaged_decision(candidate)
        embedding = self.embedder.embed_one(candidate.text)
        record = self._create_memory(
            text=candidate.text,
            confidence=candidate.confidence_score,
            conversation_id=conversation_id,
            source=candidate.source,
            embedding=embedding,
            status=LifecycleStatus.ACTIVE,
        )
        self.events.record(record.id, decision, after_state=record.model_dump(mode="json"))
        return decision, record

    def process_candidate_recency_only(
        self,
        candidate: CandidateMemory,
        budget: int = 5,
        conversation_id: str | None = None,
    ) -> tuple[LifecycleDecision, MemoryRecord, MemoryRecord | None]:
        """
        Recency-Only (LRU) baseline:
        Store candidate memory. If active memory count exceeds budget,
        evict the least recently accessed (LRU) memory.
        """
        embedding = self.embedder.embed_one(candidate.text)
        evicted_record: MemoryRecord | None = None

        # Check active count before adding
        active_memories = self.memories.list(include_inactive=False)
        if len(active_memories) >= budget:
            lru_victim = self.memories.get_least_recently_used()
            if lru_victim:
                lru_victim.lifecycle_status = LifecycleStatus.FORGOTTEN
                lru_victim.updated_at = datetime.now(timezone.utc)
                lru_victim.metadata["evicted_by_policy"] = "RECENCY_ONLY_LRU"
                self.memories.update(lru_victim)
                self.vector_store.remove(lru_victim.id)
                evicted_record = lru_victim

        record = self._create_memory(
            text=candidate.text,
            confidence=candidate.confidence_score,
            conversation_id=conversation_id,
            source=candidate.source,
            embedding=embedding,
            status=LifecycleStatus.ACTIVE,
        )
        decision = LifecycleDecision(
            operation=LifecycleOperation.RETAIN,
            reason="Stored under RECENCY_ONLY (LRU) policy." + (f" Evicted LRU memory '{evicted_record.id}'." if evicted_record else ""),
            scores=unmanaged_decision(candidate).scores,
            affected_memory_ids=[evicted_record.id] if evicted_record else [],
            metadata={"policy": "RECENCY_ONLY_LRU", "evicted_id": evicted_record.id if evicted_record else None},
        )
        self.events.record(record.id, decision, after_state=record.model_dump(mode="json"))
        return decision, record, evicted_record

    def apply_decision(
        self,
        candidate: CandidateMemory,
        decision: LifecycleDecision,
        embedding,
        conversation_id: str | None,
    ) -> MemoryRecord | None:
        op = decision.operation
        now = datetime.now(timezone.utc)

        # 1. UPDATE: supersedes or modifies existing active memory
        if op == LifecycleOperation.UPDATE and decision.affected_memory_ids:
            existing = self.memories.get(decision.affected_memory_ids[0])
            if existing:
                decision.metadata["before_state"] = existing.model_dump(mode="json")
                existing.text = candidate.text
                existing.updated_at = now
                existing.confidence_score = max(existing.confidence_score, candidate.confidence_score)
                existing.lifecycle_status = LifecycleStatus.ACTIVE
                existing.metadata["update_reason"] = decision.reason
                existing.metadata["prior_text"] = decision.metadata.get("prior_text")
                new_embedding = self.embedder.embed_one(candidate.text)
                self.memories.update(existing, new_embedding)
                self.vector_store.add(existing.id, new_embedding)
                return existing

        # 2. MERGE: consolidate redundant facts into single active memory
        if op == LifecycleOperation.MERGE and decision.affected_memory_ids:
            existing = self.memories.get(decision.affected_memory_ids[0])
            if existing:
                decision.metadata["before_state"] = existing.model_dump(mode="json")
                merged_text = self.compressor.merge([existing.text, candidate.text])
                existing.text = merged_text
                existing.updated_at = now
                existing.lifecycle_status = LifecycleStatus.ACTIVE
                existing.metadata["merge_reason"] = decision.reason
                new_embedding = self.embedder.embed_one(merged_text)
                self.memories.update(existing, new_embedding)
                self.vector_store.add(existing.id, new_embedding)
                return existing

        # 3. COMPRESS: replace verbose text with concise representation (no duplicates)
        if op == LifecycleOperation.COMPRESS:
            compressed_text = self.compressor.compress(candidate.text)
            comp_embedding = self.embedder.embed_one(compressed_text)
            record = self._create_memory(
                text=compressed_text,
                confidence=candidate.confidence_score,
                conversation_id=conversation_id,
                source=candidate.source,
                embedding=comp_embedding,
                status=LifecycleStatus.ACTIVE,
                archived=False,
                compressed_text=candidate.text,
            )
            record.metadata["original_verbose_text"] = candidate.text
            self.memories.update(record, comp_embedding)
            return record

        # 4. FORGET: store audit record in SQLite but exclude from active FAISS index
        if op == LifecycleOperation.FORGET:
            return self._create_memory(
                text=candidate.text,
                confidence=candidate.confidence_score,
                conversation_id=conversation_id,
                source=candidate.source,
                embedding=embedding,
                status=LifecycleStatus.FORGOTTEN,
                archived=False,
            )

        # 5. ARCHIVE: store in SQLite, mark archived, omit from active FAISS index
        if op == LifecycleOperation.ARCHIVE:
            return self._create_memory(
                text=candidate.text,
                confidence=candidate.confidence_score,
                conversation_id=conversation_id,
                source=candidate.source,
                embedding=embedding,
                status=LifecycleStatus.ARCHIVED,
                archived=True,
            )

        # 6. RETAIN: standard active memory
        return self._create_memory(
            text=candidate.text,
            confidence=candidate.confidence_score,
            conversation_id=conversation_id,
            source=candidate.source,
            embedding=embedding,
            status=LifecycleStatus.ACTIVE,
            archived=False,
        )

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
        # Add to vector index only if status is ACTIVE
        if status == LifecycleStatus.ACTIVE:
            self.vector_store.add(record.id, embedding)
        return record

    def forget(self, memory_id: str) -> bool:
        """
        User-requested explicit forgetting (privacy deletion).
        Marks record as FORGOTTEN in SQLite, removes vector from active FAISS index,
        and logs the audit event.
        """
        memory = self.memories.get(memory_id)
        if not memory:
            return False

        before_state = memory.model_dump(mode="json")
        memory.lifecycle_status = LifecycleStatus.FORGOTTEN
        memory.archived = False
        memory.updated_at = datetime.now(timezone.utc)
        memory.metadata["deletion_reason"] = "user_requested_deletion"
        self.memories.update(memory)

        # Remove from active FAISS index
        self.vector_store.remove(memory_id)

        # Record audit event
        decision = LifecycleDecision(
            operation=LifecycleOperation.FORGET,
            reason="User explicitly requested deletion of this memory record.",
            scores=self.controller.scorer.score(0.0, memory.confidence_score, 0.0, memory.utility_score, 0.0),
            affected_memory_ids=[memory_id],
            confidence=1.0,
            metadata={"user_action": "delete"},
        )
        self.events.record(
            memory_id=memory_id,
            decision=decision,
            before_state=before_state,
            after_state=memory.model_dump(mode="json"),
        )
        return True

    def rebuild_index(self) -> None:
        """Rebuild FAISS index from the canonical active memories in SQLite."""
        self.vector_store.rebuild(self.memories.active_embeddings())

    def stats(self) -> dict[str, Any]:
        all_memories = self.memories.list(include_inactive=True)
        return {
            "total": len(all_memories),
            "active": sum(m.lifecycle_status == LifecycleStatus.ACTIVE for m in all_memories),
            "archived": sum(m.lifecycle_status == LifecycleStatus.ARCHIVED for m in all_memories),
            "forgotten": sum(m.lifecycle_status == LifecycleStatus.FORGOTTEN for m in all_memories),
            "compressed": sum(m.compressed_text is not None for m in all_memories),
            "merged": sum("merge_reason" in m.metadata for m in all_memories),
            "updated": sum("update_reason" in m.metadata for m in all_memories),
            "faiss_indexed_count": len(self.vector_store.ids),
        }
