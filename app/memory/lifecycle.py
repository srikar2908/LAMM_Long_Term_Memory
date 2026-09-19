from __future__ import annotations

from app.core.config import Settings
from app.memory.deduplication import lexical_similarity, looks_like_update, most_redundant
from app.memory.schemas import (
    CandidateMemory,
    LifecycleDecision,
    LifecycleOperation,
    RetrievedMemory,
    ScoreBundle,
)
from app.memory.scoring import WeightedScorer


class LammLifecycleController:
    """
    Central LAMM External Lifecycle Controller.

    Decision Priority:
        1. UPDATE   -> Candidate supersedes, contradicts, or modifies an existing memory.
        2. MERGE    -> Candidate has high semantic redundancy with an existing memory.
        3. COMPRESS -> Candidate is excessively verbose but contains durable knowledge.
        4. FORGET   -> Candidate is ephemeral/filler or overall score is below forget threshold.
        5. ARCHIVE  -> Candidate has low confidence, exceeds resource capacity, or score is below archive threshold.
        6. RETAIN   -> Candidate is informative, non-redundant, and meets active retention criteria.

    Architectural Invariant:
    Semantic relationships (UPDATE, MERGE) are evaluated before score-based eviction,
    preventing a lower overall score from discarding an incoming modification or deduplication.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.scorer = WeightedScorer(settings)

    def decide(
        self,
        candidate: CandidateMemory,
        similar_memories: list[RetrievedMemory],
        active_memory_count: int,
        context_relevance: float = 0.8,
    ) -> LifecycleDecision:
        closest = most_redundant(similar_memories)
        vector_redundancy = max([item.score for item in similar_memories], default=0.0)
        lexical_redundancy = max(
            [lexical_similarity(candidate.text, item.memory.text) for item in similar_memories],
            default=0.0,
        )
        redundancy = max(vector_redundancy, lexical_redundancy)
        utility = closest.memory.utility_score if closest else 0.0
        recency = closest.memory.recency_score if closest else 1.0

        scores = self.scorer.score(
            relevance=context_relevance,
            memory_confidence=candidate.confidence_score,
            redundancy=redundancy,
            utility=utility,
            recency=recency,
        )
        affected = [closest.memory.id] if closest else []
        candidate_lower = candidate.text.lower()

        # -------------------------------------------------------------
        # 1. UPDATE: Evidence of modification, contradiction, or supersession
        # -------------------------------------------------------------
        if closest and looks_like_update(candidate.text, closest.memory.text):
            return LifecycleDecision(
                operation=LifecycleOperation.UPDATE,
                reason=(
                    f"New candidate modifies or supersedes existing memory '{closest.memory.id}' "
                    f"with temporal or topical update signal."
                ),
                scores=scores,
                affected_memory_ids=affected,
                confidence=0.88,
                metadata={"supersedes_id": closest.memory.id, "prior_text": closest.memory.text},
            )

        # -------------------------------------------------------------
        # 2. MERGE: High semantic or lexical redundancy with an active memory
        # -------------------------------------------------------------
        if closest and (redundancy >= self.settings.redundancy_threshold or lexical_redundancy >= 0.40):
            return LifecycleDecision(
                operation=LifecycleOperation.MERGE,
                reason=(
                    f"Candidate is semantically redundant (score={redundancy:.2f} >= threshold "
                    f"{self.settings.redundancy_threshold:.2f}) with existing memory '{closest.memory.id}'."
                ),
                scores=scores,
                affected_memory_ids=affected,
                confidence=0.86,
                metadata={"merged_with_id": closest.memory.id, "redundancy_score": redundancy},
            )

        # -------------------------------------------------------------
        # 3. COMPRESS: Excessively verbose candidate
        # -------------------------------------------------------------
        if len(candidate.text) > self.settings.compress_length:
            return LifecycleDecision(
                operation=LifecycleOperation.COMPRESS,
                reason=(
                    f"Candidate length ({len(candidate.text)} chars) exceeds compression threshold "
                    f"({self.settings.compress_length} chars). Compressing before active indexing."
                ),
                scores=scores,
                affected_memory_ids=affected,
                confidence=0.80,
                metadata={"original_length": len(candidate.text)},
            )

        # -------------------------------------------------------------
        # 4. FORGET: Ephemeral filler, temporary info, or score <= forget_threshold
        # -------------------------------------------------------------
        is_ephemeral = (
            "temporary" in candidate_lower
            or "not very useful" in candidate_lower
            or "filler" in candidate_lower
            or "disregard" in candidate_lower
        )
        if is_ephemeral or scores.overall <= self.settings.forget_threshold:
            reason = (
                "Candidate contains explicit ephemeral/filler phrasing."
                if is_ephemeral
                else f"Composite score ({scores.overall:.2f}) is at or below forget threshold ({self.settings.forget_threshold:.2f})."
            )
            return LifecycleDecision(
                operation=LifecycleOperation.FORGET,
                reason=reason,
                scores=scores,
                affected_memory_ids=affected,
                confidence=0.85 if is_ephemeral else 0.75,
                metadata={"is_ephemeral": is_ephemeral},
            )

        # -------------------------------------------------------------
        # 5. ARCHIVE: Low confidence, capacity limit, or score <= archive_threshold
        # -------------------------------------------------------------
        is_low_conf = candidate.confidence_score < 0.50
        is_over_capacity = active_memory_count >= self.settings.max_active_memories
        is_below_archive = scores.overall <= self.settings.archive_threshold

        if is_low_conf or is_over_capacity or is_below_archive:
            if is_low_conf:
                reason = f"Candidate confidence ({candidate.confidence_score:.2f}) is below active threshold (0.50); archived."
            elif is_over_capacity:
                reason = f"Active memory capacity reached ({active_memory_count}/{self.settings.max_active_memories}); candidate archived."
            else:
                reason = f"Composite score ({scores.overall:.2f}) is below archive threshold ({self.settings.archive_threshold:.2f}); candidate archived."

            return LifecycleDecision(
                operation=LifecycleOperation.ARCHIVE,
                reason=reason,
                scores=scores,
                affected_memory_ids=affected,
                confidence=0.75,
                metadata={"low_confidence": is_low_conf, "capacity_limit": is_over_capacity},
            )

        # -------------------------------------------------------------
        # 6. RETAIN: Informative, durable, and non-redundant
        # -------------------------------------------------------------
        return LifecycleDecision(
            operation=LifecycleOperation.RETAIN,
            reason=(
                f"Candidate represents useful, high-confidence ({candidate.confidence_score:.2f}) knowledge "
                f"with acceptable redundancy ({redundancy:.2f}) and score ({scores.overall:.2f})."
            ),
            scores=scores,
            affected_memory_ids=affected,
            confidence=0.85,
            metadata={},
        )


def unmanaged_decision(candidate: CandidateMemory) -> LifecycleDecision:
    """Unmanaged memory baseline policy: unconditionally store all candidate memories."""
    scores = ScoreBundle(
        relevance=0.8,
        recency=1.0,
        confidence=candidate.confidence_score,
        redundancy=0.0,
        utility=0.0,
        overall=candidate.confidence_score,
    )
    return LifecycleDecision(
        operation=LifecycleOperation.RETAIN,
        reason="UNMANAGED_MEMORY baseline unconditionally stores every extracted candidate without lifecycle management.",
        scores=scores,
        confidence=candidate.confidence_score,
        metadata={"baseline": "UNMANAGED_MEMORY"},
    )
