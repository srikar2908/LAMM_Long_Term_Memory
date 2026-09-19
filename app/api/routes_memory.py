from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.memory.manager import MemoryManager
from app.memory.schemas import CandidateMemory

router = APIRouter(tags=["Memory Management"])
manager = MemoryManager()


class MemoryCreateRequest(BaseModel):
    text: str = Field(..., description="The candidate memory text to evaluate and store")
    confidence_score: float = Field(default=0.75, ge=0.0, le=1.0, description="Extraction confidence score")
    conversation_id: str = Field(default="api", description="Conversation context identifier")
    dry_run: bool = Field(default=False, description="Simulate decision without modifying SQLite or FAISS")


@router.post("/memory", summary="Ingest candidate memory through LAMM lifecycle")
def create_memory(request: MemoryCreateRequest) -> dict[str, Any]:
    """
    Submits a candidate memory to the LAMM lifecycle controller.
    Applies scoring across relevance, recency, confidence, redundancy, and utility signals,
    making an explainable decision (UPDATE, MERGE, COMPRESS, FORGET, ARCHIVE, RETAIN).
    """
    candidate = CandidateMemory(
        text=request.text,
        confidence_score=request.confidence_score,
        source="api",
    )
    decision, record = manager.process_candidate(
        candidate=candidate,
        conversation_id=request.conversation_id,
        dry_run=request.dry_run,
    )
    return {
        "decision": decision.model_dump(mode="json"),
        "memory": record.model_dump(mode="json") if record else None,
    }


@router.get("/memories", summary="List all memories")
def list_memories(include_inactive: bool = Query(True, description="Include archived and forgotten memories")):
    """Retrieves all memory records stored in the canonical SQLite database."""
    return [m.model_dump(mode="json") for m in manager.memories.list(include_inactive=include_inactive)]


@router.get("/memory/search", summary="Semantic vector search over active memories")
def search_memories(
    q: str = Query(..., description="Query text to search semantically"),
    top_k: int = Query(5, ge=1, le=50, description="Maximum number of memories to retrieve"),
    include_archived: bool = Query(False, description="Whether to include archived memories in retrieval pool"),
):
    """
    Searches the FAISS vector index for active memories most semantically relevant
    to the query. Returns cosine similarity scores and access count metadata.
    """
    retrieved = manager.retrieve(query=q, top_k=top_k, include_archived=include_archived)
    return [
        {
            "id": item.memory.id,
            "text": item.memory.text,
            "relevance_score": round(item.score, 4),
            "access_count": item.memory.access_count,
            "lifecycle_status": item.memory.lifecycle_status.value,
            "created_at": item.memory.created_at.isoformat(),
            "updated_at": item.memory.updated_at.isoformat(),
        }
        for item in retrieved
    ]


@router.get("/memory/stats", summary="Memory system statistics")
def get_memory_stats() -> dict[str, Any]:
    """Returns counts of total, active, archived, forgotten, compressed, and merged memories."""
    return manager.stats()


@router.get("/memory/{memory_id}", summary="Get a specific memory by ID")
def get_memory(memory_id: str):
    """Fetches a specific memory record by its UUID."""
    memory = manager.memories.get(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail=f"Memory with ID '{memory_id}' not found.")
    return memory.model_dump(mode="json")


@router.delete("/memory/{memory_id}", summary="Explicit privacy deletion (Forget memory)")
def delete_memory(memory_id: str):
    """
    Explicit user privacy deletion:
    Marks the record as FORGOTTEN in SQLite, removes its vector from the active FAISS index,
    and logs the audit event.
    """
    success = manager.forget(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Memory with ID '{memory_id}' not found.")
    return {"status": "forgotten", "memory_id": memory_id, "detail": "Removed from active FAISS index and marked FORGOTTEN."}


@router.post("/memory/rebuild-index", summary="Rebuild FAISS index from SQLite active memories")
def rebuild_index():
    """Synchronizes and rebuilds the FAISS vector index from canonical SQLite active memories."""
    manager.rebuild_index()
    return {"status": "rebuilt", "indexed_active_count": len(manager.vector_store.ids)}


@router.get("/lifecycle/events", summary="Audit trail of lifecycle decisions")
def list_lifecycle_events():
    """Returns the immutable log of all lifecycle decisions, reasons, scores, and state transitions."""
    return manager.events.list()
