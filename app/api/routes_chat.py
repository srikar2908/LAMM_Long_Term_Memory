from __future__ import annotations

from fastapi import APIRouter

from app.agent.agent import LammAgent
from app.agent.conversation import ChatRequest, ChatResponse
from app.api.routes_memory import manager

router = APIRouter(tags=["Conversational Agent"])
agent = LammAgent(manager=manager)


@router.post("/chat", response_model=ChatResponse, summary="Chat with agent and apply memory lifecycle")
def chat(request: ChatRequest) -> ChatResponse:
    """
    Submits a conversational message to the agent.
    Retrieves active memories, generates a response, extracts facts,
    and governs memory lifecycle through LAMM.
    """
    return agent.chat(request.message, request.conversation_id)
