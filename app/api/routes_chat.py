from fastapi import APIRouter

from app.agent.agent import LammAgent
from app.agent.conversation import ChatRequest, ChatResponse

router = APIRouter()
agent = LammAgent()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return agent.chat(request.message, request.conversation_id)
