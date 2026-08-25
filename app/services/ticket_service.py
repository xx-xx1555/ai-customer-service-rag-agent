from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy.exc import IntegrityError

from app.db.init_db import initialize_database
from app.db.models import Ticket
from app.db.session import session_scope
from app.repositories.ticket_repository import TicketRepository
from app.schemas.ticket_schema import TicketCreate, TicketUpdate


RESOLVED_STATUSES = {"已解决", "已关闭"}


def _ensure_database() -> None:
    initialize_database()


def _ticket_to_dict(ticket: Ticket) -> Dict[str, Any]:
    return {
        "ticket_id": ticket.ticket_id,
        "user_id": ticket.user_id,
        "issue_type": ticket.issue_type,
        "description": ticket.description,
        "status": ticket.status,
        "created_at": ticket.created_at,
        "resolved_hours": float(ticket.resolved_hours or 0),
        "satisfaction": int(ticket.satisfaction or 0),
        "updated_at": ticket.updated_at,
    }


def _generate_ticket_id() -> str:
    return f"T{datetime.now(timezone.utc):%Y%m%d}{uuid4().hex[:6].upper()}"


def create_ticket(payload: TicketCreate) -> Dict[str, Any]:
    _ensure_database()
    ticket = Ticket(
        ticket_id=payload.ticket_id or _generate_ticket_id(),
        user_id=payload.user_id.strip(),
        issue_type=payload.issue_type.strip(),
        description=payload.description.strip(),
        status=payload.status.strip(),
        created_at=payload.created_at,
        resolved_hours=payload.resolved_hours,
        satisfaction=payload.satisfaction,
    )
    try:
        with session_scope() as session:
            created = TicketRepository(session).create(ticket)
            return _ticket_to_dict(created)
    except IntegrityError as exc:
        raise ValueError(f"工单编号已存在：{ticket.ticket_id}") from exc


def get_ticket(ticket_id: str) -> Optional[Dict[str, Any]]:
    _ensure_database()
    with session_scope() as session:
        ticket = TicketRepository(session).get(ticket_id)
        return _ticket_to_dict(ticket) if ticket else None


def list_tickets(
    *,
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    issue_type: Optional[str] = None,
    satisfaction_lte: Optional[int] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    keyword: Optional[str] = None,
) -> Dict[str, Any]:
    _ensure_database()
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 100)
    with session_scope() as session:
        items, total = TicketRepository(session).list(
            page=page,
            page_size=page_size,
            status=status,
            issue_type=issue_type,
            satisfaction_lte=satisfaction_lte,
            date_from=date_from,
            date_to=date_to,
            keyword=keyword,
        )
        return {
            "items": [_ticket_to_dict(item) for item in items],
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": ceil(total / page_size) if total else 0,
        }


def update_ticket(ticket_id: str, payload: TicketUpdate) -> Optional[Dict[str, Any]]:
    _ensure_database()
    values = payload.model_dump(exclude_unset=True)
    for field in ("user_id", "issue_type", "description", "status"):
        if field in values and isinstance(values[field], str):
            values[field] = values[field].strip()

    with session_scope() as session:
        repository = TicketRepository(session)
        ticket = repository.get(ticket_id)
        if not ticket:
            return None
        updated = repository.update(ticket, values)
        return _ticket_to_dict(updated)


def delete_ticket(ticket_id: str) -> bool:
    _ensure_database()
    with session_scope() as session:
        repository = TicketRepository(session)
        ticket = repository.get(ticket_id)
        if not ticket:
            return False
        repository.delete(ticket)
        return True


def _all_ticket_records() -> List[Dict[str, Any]]:
    _ensure_database()
    with session_scope() as session:
        return [_ticket_to_dict(item) for item in TicketRepository(session).list_all()]


def get_ticket_summary() -> Dict[str, Any]:
    _ensure_database()
    with session_scope() as session:
        repository = TicketRepository(session)
        tickets = repository.list_all()
        total = len(tickets)
        if not tickets:
            return {
                "total_tickets": 0,
                "issue_type_counts": {},
                "status_counts": {},
                "avg_satisfaction": 0.0,
                "avg_resolved_hours": 0.0,
                "unresolved_rate": 0.0,
                "low_satisfaction_rate": 0.0,
                "risk_tickets": [],
            }

        resolved_hours = [float(item.resolved_hours) for item in tickets if item.resolved_hours > 0]
        unresolved = [item for item in tickets if item.status not in RESOLVED_STATUSES]
        low_satisfaction = [item for item in tickets if item.satisfaction <= 2]
        risk_map = {
            item.ticket_id: item
            for item in [*unresolved, *low_satisfaction]
        }

        return {
            "total_tickets": total,
            "issue_type_counts": repository.issue_type_counts(),
            "status_counts": repository.status_counts(),
            "avg_satisfaction": round(sum(item.satisfaction for item in tickets) / total, 2),
            "avg_resolved_hours": round(sum(resolved_hours) / len(resolved_hours), 2)
            if resolved_hours
            else 0.0,
            "unresolved_rate": round(len(unresolved) / total, 4),
            "low_satisfaction_rate": round(len(low_satisfaction) / total, 4),
            "risk_tickets": [_ticket_to_dict(item) for item in risk_map.values()],
        }


def get_unresolved_tickets() -> List[Dict[str, Any]]:
    return [
        item
        for item in _all_ticket_records()
        if item["status"] not in RESOLVED_STATUSES
    ]


def get_top_issue_types(top_k: int = 3) -> Dict[str, Any]:
    _ensure_database()
    top_k = max(int(top_k), 1)
    with session_scope() as session:
        counts = TicketRepository(session).issue_type_counts()
    return {
        "top_k": top_k,
        "top_issue_types": dict(list(counts.items())[:top_k]),
    }


def search_tickets_by_keyword(keyword: str) -> List[Dict[str, Any]]:
    _ensure_database()
    with session_scope() as session:
        tickets = TicketRepository(session).list_all(keyword=keyword)
        return [_ticket_to_dict(item) for item in tickets]


def get_low_satisfaction_tickets(threshold: int = 2) -> List[Dict[str, Any]]:
    _ensure_database()
    with session_scope() as session:
        tickets = TicketRepository(session).list_all(satisfaction_lte=threshold)
        return [_ticket_to_dict(item) for item in tickets]


def get_ticket_trends(days: int = 30) -> List[Dict[str, Any]]:
    _ensure_database()
    days = max(int(days), 1)
    date_from = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as session:
        trends = TicketRepository(session).daily_counts(date_from=date_from)
        if trends:
            return trends
        # 演示数据可能早于当前时间，空结果时按全部数据返回，前端不会变成一块寂静的白板。
        return TicketRepository(session).daily_counts()


def get_ticket_filter_options() -> Dict[str, List[str]]:
    _ensure_database()
    with session_scope() as session:
        repository = TicketRepository(session)
        return {
            "issue_types": repository.distinct_issue_types(),
            "statuses": repository.distinct_statuses(),
        }


def build_ticket_analysis_context() -> Dict[str, Any]:
    return {
        "summary": get_ticket_summary(),
        "unresolved_tickets": get_unresolved_tickets(),
        "top_issues": get_top_issue_types(top_k=3),
        "low_satisfaction_tickets": get_low_satisfaction_tickets(threshold=2),
        "period_comparison": compare_ticket_periods(days=30),
    }


def _period_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "ticket_count": 0,
            "avg_satisfaction": 0.0,
            "avg_resolved_hours": 0.0,
            "unresolved_count": 0,
            "top_issue_types": {},
        }

    resolved = [item["resolved_hours"] for item in records if item["resolved_hours"] > 0]
    issue_counts: Dict[str, int] = {}
    for item in records:
        issue_counts[item["issue_type"]] = issue_counts.get(item["issue_type"], 0) + 1
    top_issues = dict(sorted(issue_counts.items(), key=lambda pair: pair[1], reverse=True)[:3])
    return {
        "ticket_count": len(records),
        "avg_satisfaction": round(sum(item["satisfaction"] for item in records) / len(records), 2),
        "avg_resolved_hours": round(sum(resolved) / len(resolved), 2) if resolved else 0.0,
        "unresolved_count": sum(
            item["status"] not in RESOLVED_STATUSES for item in records
        ),
        "top_issue_types": top_issues,
    }


def compare_ticket_periods(days: int = 30) -> Dict[str, Any]:
    records = _all_ticket_records()
    if not records:
        return {"days": days, "current_period": {}, "previous_period": {}, "changes": {}}

    valid_records = [item for item in records if isinstance(item["created_at"], datetime)]
    if not valid_records:
        return {
            "days": days,
            "current_period": {},
            "previous_period": {},
            "changes": {},
            "warning": "created_at 无法解析",
        }

    days = max(int(days), 1)
    anchor_date = max(item["created_at"] for item in valid_records)
    if anchor_date.tzinfo is None:
        anchor_date = anchor_date.replace(tzinfo=timezone.utc)
    anchor = anchor_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    current_start = anchor - timedelta(days=days)
    previous_start = current_start - timedelta(days=days)

    def normalized_date(item: Dict[str, Any]) -> datetime:
        value = item["created_at"]
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value

    current = [item for item in valid_records if current_start <= normalized_date(item) < anchor]
    previous = [item for item in valid_records if previous_start <= normalized_date(item) < current_start]

    current_metrics = _period_metrics(current)
    previous_metrics = _period_metrics(previous)
    return {
        "days": days,
        "anchor_date": str((anchor - timedelta(days=1)).date()),
        "current_period": {
            "start": str(current_start.date()),
            "end": str((anchor - timedelta(days=1)).date()),
            **current_metrics,
        },
        "previous_period": {
            "start": str(previous_start.date()),
            "end": str((current_start - timedelta(days=1)).date()),
            **previous_metrics,
        },
        "changes": {
            "ticket_count": current_metrics["ticket_count"] - previous_metrics["ticket_count"],
            "avg_satisfaction": round(
                current_metrics["avg_satisfaction"] - previous_metrics["avg_satisfaction"], 2
            ),
            "avg_resolved_hours": round(
                current_metrics["avg_resolved_hours"] - previous_metrics["avg_resolved_hours"], 2
            ),
            "unresolved_count": current_metrics["unresolved_count"]
            - previous_metrics["unresolved_count"],
        },
    }


def create_faq_candidates(top_k: int = 5) -> Dict[str, Any]:
    records = _all_ticket_records()
    if not records:
        return {"top_k": top_k, "candidates": []}

    top_k = max(int(top_k), 1)
    issue_groups: Dict[str, List[str]] = {}
    for item in records:
        issue_groups.setdefault(item["issue_type"], []).append(item["description"])

    sorted_groups = sorted(issue_groups.items(), key=lambda pair: len(pair[1]), reverse=True)[:top_k]
    candidates = []
    for issue_type, descriptions in sorted_groups:
        samples = list(dict.fromkeys(descriptions))[:3]
        candidates.append(
            {
                "issue_type": issue_type,
                "suggested_question": f"遇到{issue_type}问题时应该如何处理？",
                "evidence_count": len(descriptions),
                "sample_descriptions": samples,
                "status": "待知识库管理员补充标准答案",
            }
        )
    return {"top_k": top_k, "candidates": candidates}


def get_ticket_dashboard(days: int = 30) -> Dict[str, Any]:
    options = get_ticket_filter_options()
    return {
        "summary": get_ticket_summary(),
        "trends": get_ticket_trends(days=days),
        "issue_types": options["issue_types"],
        "statuses": options["statuses"],
        "period_comparison": compare_ticket_periods(days=days),
    }
