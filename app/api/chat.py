from fastapi import APIRouter

from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.persistence_service import save_chat_exchange
from app.services.rag_service import answer_question


router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = answer_question(
        question=request.question,
        top_k=request.top_k,
        min_score=request.min_score,
        mode=request.mode,
        rerank=request.rerank,
        candidate_k=request.candidate_k,
    )
    session_id = save_chat_exchange(
        question=request.question,
        answer=result["answer"],
        sources=result.get("sources", []),
        session_id=request.session_id,
    )
    return ChatResponse(**result, session_id=session_id)
