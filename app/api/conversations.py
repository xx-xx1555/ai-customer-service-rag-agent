from typing import List

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.schemas.conversation_schema import ConversationDetail, ConversationSummary
from app.services.persistence_service import (
    delete_chat_session,
    get_chat_session,
    list_chat_sessions,
)


router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.get("", response_model=List[ConversationSummary])
def conversations(limit: int = Query(default=50, ge=1, le=200)):
    return list_chat_sessions(limit=limit)


@router.get("/{session_id}", response_model=ConversationDetail)
def conversation_detail(session_id: str):
    result = get_chat_session(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return result


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_conversation(session_id: str):
    if not delete_chat_session(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
