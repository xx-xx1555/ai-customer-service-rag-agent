from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class ConversationSummary(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationMessage(BaseModel):
    message_id: int
    role: str
    content: str
    sources: List[str] = Field(default_factory=list)
    created_at: datetime


class ConversationDetail(ConversationSummary):
    messages: List[ConversationMessage] = Field(default_factory=list)
