from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "demo"


class ChatResponse(BaseModel):
    response: str
    retrieved_memories: list[dict] = Field(default_factory=list)
    extracted_memories: list[dict] = Field(default_factory=list)
    lifecycle_decisions: list[dict] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
