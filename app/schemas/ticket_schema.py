from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TicketBase(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    issue_type: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=2, max_length=5000)
    status: str = Field(default="待处理", min_length=1, max_length=32)
    created_at: datetime = Field(default_factory=datetime.now)
    resolved_hours: float = Field(default=0.0, ge=0)
    satisfaction: int = Field(default=3, ge=1, le=5)


class TicketCreate(TicketBase):
    ticket_id: Optional[str] = Field(default=None, min_length=1, max_length=64)


class TicketUpdate(BaseModel):
    user_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    issue_type: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, min_length=2, max_length=5000)
    status: Optional[str] = Field(default=None, min_length=1, max_length=32)
    created_at: Optional[datetime] = None
    resolved_hours: Optional[float] = Field(default=None, ge=0)
    satisfaction: Optional[int] = Field(default=None, ge=1, le=5)

    @field_validator("user_id", "issue_type", "description", "status")
    @classmethod
    def reject_blank_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("字段不能为空白字符串")
        return value


class TicketResponse(TicketBase):
    model_config = ConfigDict(from_attributes=True)

    ticket_id: str
    updated_at: datetime


class TicketListResponse(BaseModel):
    items: List[TicketResponse] = Field(default_factory=list)
    total: int
    page: int
    page_size: int
    pages: int


class TicketSearchRequest(BaseModel):
    keyword: str = Field(..., min_length=1)


class TicketSummaryResponse(BaseModel):
    total_tickets: int
    issue_type_counts: Dict[str, int]
    status_counts: Dict[str, int]
    avg_satisfaction: float
    avg_resolved_hours: float
    unresolved_rate: float = 0.0
    low_satisfaction_rate: float = 0.0
    risk_tickets: List[Dict[str, Any]] = Field(default_factory=list)


class TicketTrendPoint(BaseModel):
    date: str
    count: int


class TicketDashboardResponse(BaseModel):
    summary: TicketSummaryResponse
    trends: List[TicketTrendPoint] = Field(default_factory=list)
    issue_types: List[str] = Field(default_factory=list)
    statuses: List[str] = Field(default_factory=list)
    period_comparison: Dict[str, Any] = Field(default_factory=dict)
