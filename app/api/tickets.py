from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.schemas.ticket_schema import (
    TicketCreate,
    TicketDashboardResponse,
    TicketListResponse,
    TicketResponse,
    TicketSearchRequest,
    TicketSummaryResponse,
    TicketUpdate,
)
from app.services.ticket_ai_service import generate_ticket_ai_report
from app.services.ticket_service import (
    create_ticket,
    delete_ticket,
    get_low_satisfaction_tickets,
    get_ticket,
    get_ticket_dashboard,
    get_ticket_filter_options,
    get_ticket_summary,
    get_ticket_trends,
    get_top_issue_types,
    get_unresolved_tickets,
    list_tickets,
    search_tickets_by_keyword,
    update_ticket,
)


router = APIRouter(prefix="/tickets", tags=["Tickets"])


@router.get("/dashboard", response_model=TicketDashboardResponse)
def ticket_dashboard(days: int = Query(default=30, ge=1, le=365)):
    return get_ticket_dashboard(days=days)


@router.get("/summary", response_model=TicketSummaryResponse)
def ticket_summary():
    return get_ticket_summary()


@router.get("/trends")
def ticket_trends(days: int = Query(default=30, ge=1, le=365)):
    return {"days": days, "trends": get_ticket_trends(days=days)}


@router.get("/filter-options")
def ticket_filter_options():
    return get_ticket_filter_options()


@router.get("/unresolved")
def unresolved_tickets():
    return {"tickets": get_unresolved_tickets()}


@router.get("/top-issues")
def top_issues(top_k: int = Query(default=3, ge=1, le=20)):
    return get_top_issue_types(top_k=top_k)


@router.get("/low-satisfaction")
def low_satisfaction(threshold: int = Query(default=2, ge=1, le=5)):
    return {"tickets": get_low_satisfaction_tickets(threshold=threshold)}


@router.post("/search")
def search_tickets(request: TicketSearchRequest):
    return {
        "keyword": request.keyword,
        "tickets": search_tickets_by_keyword(request.keyword),
    }


@router.get("/ai-report")
def ticket_ai_report():
    return generate_ticket_ai_report()


@router.post("", response_model=TicketResponse, status_code=status.HTTP_201_CREATED)
def create_ticket_endpoint(payload: TicketCreate):
    try:
        return create_ticket(payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("", response_model=TicketListResponse)
def list_tickets_endpoint(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status_value: Optional[str] = Query(default=None, alias="status"),
    issue_type: Optional[str] = None,
    satisfaction_lte: Optional[int] = Query(default=None, ge=1, le=5),
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    keyword: Optional[str] = None,
):
    return list_tickets(
        page=page,
        page_size=page_size,
        status=status_value,
        issue_type=issue_type,
        satisfaction_lte=satisfaction_lte,
        date_from=date_from,
        date_to=date_to,
        keyword=keyword,
    )


@router.get("/{ticket_id}", response_model=TicketResponse)
def get_ticket_endpoint(ticket_id: str):
    ticket = get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="工单不存在")
    return ticket


@router.patch("/{ticket_id}", response_model=TicketResponse)
def update_ticket_endpoint(ticket_id: str, payload: TicketUpdate):
    ticket = update_ticket(ticket_id, payload)
    if not ticket:
        raise HTTPException(status_code=404, detail="工单不存在")
    return ticket


@router.delete("/{ticket_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_ticket_endpoint(ticket_id: str):
    if not delete_ticket(ticket_id):
        raise HTTPException(status_code=404, detail="工单不存在")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
