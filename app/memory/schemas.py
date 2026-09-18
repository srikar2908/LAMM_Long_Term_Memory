from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LifecycleStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    FORGOTTEN = "FORGOTTEN"
    MERGED = "MERGED"
    COMPRESSED = "COMPRESSED"


class LifecycleOperation(str, Enum):
    RETAIN = "RETAIN"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    COMPRESS = "COMPRESS"
    ARCHIVE = "ARCHIVE"
    FORGET = "FORGET"


class CandidateMemory(BaseModel):
    text: str
    confidence_score: float = Field(default=0.75, ge=0.0, le=1.0)
    source: str = "deterministic"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    id: str
    text: str
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None = None
    access_count: int = 0
    relevance_score: float = 0.0
    recency_score: float = 1.0
    confidence_score: float = 0.75
    redundancy_score: float = 0.0
    utility_score: float = 0.0
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE
    source: str | None = None
    conversation_id: str | None = None
    archived: bool = False
    compressed_text: str | None = None
    parent_memory_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoreBundle(BaseModel):
    relevance: float = 0.0
    recency: float = 1.0
    confidence: float = 0.75
    redundancy: float = 0.0
    utility: float = 0.0
    overall: float = 0.0


class LifecycleDecision(BaseModel):
    operation: LifecycleOperation
    reason: str
    scores: ScoreBundle
    affected_memory_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedMemory(BaseModel):
    memory: MemoryRecord
    score: float
