from __future__ import annotations

from app.core.config import Settings
from app.memory.deduplication import lexical_similarity, looks_like_update, most_redundant
from app.memory.schemas import CandidateMemory, LifecycleDecision, LifecycleOperation, RetrievedMemory, ScoreBundle
from app.memory.scoring import WeightedScorer


class LammLifecycleController:
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
        lexical_redundancy = max([lexical_similarity(candidate.text, item.memory.text) for item in similar_memories], default=0.0)
        redundancy = max(max([item.score for item in similar_memories], default=0.0), lexical_redundancy)
        utility = closest.memory.utility_score if closest else 0.0
        recency = closest.memory.recency_score if closest else 1.0
        scores = self.scorer.score(context_relevance, candidate.confidence_score, redundancy, utility, recency)
        affected = [closest.memory.id] if closest else []

        candidate_lower = candidate.text.lower()
        if "temporary" in candidate_lower or "not very useful" in candidate_lower:
            return LifecycleDecision(
                operation=LifecycleOperation.FORGET,
                reason="Candidate is explicitly temporary or not useful for long-term memory.",
                scores=scores,
                affected_memory_ids=affected,
                confidence=0.80,
            )
        if closest and looks_like_update(candidate.text, closest.memory.text):
            return LifecycleDecision(
                operation=LifecycleOperation.UPDATE,
                reason="New candidate appears to update an existing memory on the same topic.",
                scores=scores,
                affected_memory_ids=affected,
                confidence=0.82,
            )
        if (redundancy >= self.settings.redundancy_threshold or lexical_redundancy >= 0.25) and closest:
            return LifecycleDecision(
                operation=LifecycleOperation.MERGE,
                reason=f"High semantic similarity with existing memory ({redundancy:.2f}).",
                scores=scores,
                affected_memory_ids=affected,
                confidence=0.86,
            )
        if len(candidate.text) > self.settings.compress_length:
            return LifecycleDecision(
                operation=LifecycleOperation.COMPRESS,
                reason="Candidate is verbose and can be compressed before storage.",
                scores=scores,
                confidence=0.75,
            )
        if candidate.confidence_score < 0.50:
            return LifecycleDecision(
                operation=LifecycleOperation.ARCHIVE,
                reason="Candidate is low confidence, so it is archived instead of active retrieval.",
                scores=scores,
                affected_memory_ids=affected,
                confidence=0.72,
            )
        if scores.overall <= self.settings.forget_threshold:
            return LifecycleDecision(
                operation=LifecycleOperation.FORGET,
                reason="Overall score is below forget threshold.",
                scores=scores,
                affected_memory_ids=affected,
                confidence=0.70,
            )
        if active_memory_count >= self.settings.max_active_memories or scores.overall <= self.settings.archive_threshold:
            return LifecycleDecision(
                operation=LifecycleOperation.ARCHIVE,
                reason="Memory has low active priority under configured resource constraints.",
                scores=scores,
                affected_memory_ids=affected,
                confidence=0.72,
            )
        return LifecycleDecision(
            operation=LifecycleOperation.RETAIN,
            reason="Useful candidate with acceptable redundancy and sufficient score.",
            scores=scores,
            affected_memory_ids=affected,
            confidence=0.80,
        )


def unmanaged_decision(candidate: CandidateMemory) -> LifecycleDecision:
    scores = ScoreBundle(confidence=candidate.confidence_score, relevance=0.8, overall=candidate.confidence_score)
    return LifecycleDecision(
        operation=LifecycleOperation.RETAIN,
        reason="UNMANAGED_MEMORY baseline stores every extracted candidate.",
        scores=scores,
        confidence=candidate.confidence_score,
        metadata={"baseline": "UNMANAGED_MEMORY"},
    )
