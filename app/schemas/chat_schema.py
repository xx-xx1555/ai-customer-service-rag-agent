from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.document_schema import SearchHit


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(default=None, max_length=64)
    top_k: int = Field(default=4, ge=1, le=10)
    min_score: float = Field(default=0.02, ge=0, le=1)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    rerank: bool = True
    candidate_k: int | None = Field(default=None, ge=1, le=100)


class ChatResponse(BaseModel):
    question: str
    answer: str
    session_id: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    contexts: List[SearchHit] = Field(default_factory=list)
