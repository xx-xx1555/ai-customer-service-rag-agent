from typing import Optional
from pydantic import BaseModel


class MessageResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    detail: str
    suggestion: Optional[str] = None
