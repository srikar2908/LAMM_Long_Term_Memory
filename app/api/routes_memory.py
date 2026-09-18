from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.memory.manager import MemoryManager
from app.memory.schemas import CandidateMemory

router = APIRouter()
manager = MemoryManager()


class MemoryCreateRequest(BaseModel):
    text: str
    confidence_score: float = 0.75
    conversation_id: str = "api"


@router.post("/memory")
def create_memory(request: MemoryCreateRequest):
    decision, record = manager.process_candidate(
        CandidateMemory(text=request.text, confidence_score=request.confidence_score, source="api"),
        request.conversation_id,
    )
    return {"decision": decision.model_dump(mode="json"), "memory": record.model_dump(mode="json") if record else None}


@router.get("/memory/{memory_id}")
def get_memory(memory_id: str):
    memory = manager.memories.get(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory.model_dump(mode="json")


@router.get("/memories")
def list_memories():
    return [memory.model_dump(mode="json") for memory in manager.memories.list()]


@router.delete("/memory/{memory_id}")
def delete_memory(memory_id: str):
    if not manager.forget(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "forgotten", "memory_id": memory_id}


@router.post("/memory/rebuild-index")
def rebuild_index():
    manager.rebuild_index()
    return {"status": "rebuilt"}


@router.get("/memory/stats")
def stats():
    return manager.stats()


@router.get("/lifecycle/events")
def events():
    return manager.events.list()
