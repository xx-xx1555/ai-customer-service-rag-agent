from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from sqlalchemy import Select, delete, func, or_, select
from sqlalchemy.orm import Session

from app.db.models import Ticket


class TicketRepository:
    def __init__(self, session: Session):
        self.session = session

    @staticmethod
    def _apply_filters(
        statement: Select,
        *,
        status: Optional[str] = None,
        issue_type: Optional[str] = None,
        satisfaction_lte: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        keyword: Optional[str] = None,
    ) -> Select:
        if status:
            statement = statement.where(Ticket.status == status)
        if issue_type:
            statement = statement.where(Ticket.issue_type == issue_type)
        if satisfaction_lte is not None:
            statement = statement.where(Ticket.satisfaction <= satisfaction_lte)
        if date_from:
            statement = statement.where(Ticket.created_at >= date_from)
        if date_to:
            statement = statement.where(Ticket.created_at <= date_to)
        if keyword:
            pattern = f"%{keyword}%"
            statement = statement.where(
                or_(
                    Ticket.description.ilike(pattern),
                    Ticket.issue_type.ilike(pattern),
                    Ticket.ticket_id.ilike(pattern),
                    Ticket.user_id.ilike(pattern),
                )
            )
        return statement

    def create(self, ticket: Ticket) -> Ticket:
        self.session.add(ticket)
        self.session.flush()
        self.session.refresh(ticket)
        return ticket

    def get(self, ticket_id: str) -> Optional[Ticket]:
        return self.session.get(Ticket, ticket_id)

    def list(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        issue_type: Optional[str] = None,
        satisfaction_lte: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        keyword: Optional[str] = None,
    ) -> tuple[list[Ticket], int]:
        filters = {
            "status": status,
            "issue_type": issue_type,
            "satisfaction_lte": satisfaction_lte,
            "date_from": date_from,
            "date_to": date_to,
            "keyword": keyword,
        }
        statement = self._apply_filters(select(Ticket), **filters)
        count_statement = self._apply_filters(select(func.count()).select_from(Ticket), **filters)

        total = int(self.session.scalar(count_statement) or 0)
        items = list(
            self.session.scalars(
                statement.order_by(Ticket.created_at.desc(), Ticket.ticket_id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
        )
        return items, total

    def list_all(
        self,
        *,
        status: Optional[str] = None,
        issue_type: Optional[str] = None,
        satisfaction_lte: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        keyword: Optional[str] = None,
    ) -> list[Ticket]:
        statement = self._apply_filters(
            select(Ticket),
            status=status,
            issue_type=issue_type,
            satisfaction_lte=satisfaction_lte,
            date_from=date_from,
            date_to=date_to,
            keyword=keyword,
        )
        return list(self.session.scalars(statement.order_by(Ticket.created_at.desc())))

    def update(self, ticket: Ticket, values: dict) -> Ticket:
        for key, value in values.items():
            setattr(ticket, key, value)
        self.session.flush()
        self.session.refresh(ticket)
        return ticket

    def delete(self, ticket: Ticket) -> None:
        self.session.delete(ticket)
        self.session.flush()

    def delete_all(self) -> int:
        result = self.session.execute(delete(Ticket))
        return int(result.rowcount or 0)

    def count(self) -> int:
        return int(self.session.scalar(select(func.count()).select_from(Ticket)) or 0)

    def issue_type_counts(self) -> dict[str, int]:
        rows = self.session.execute(
            select(Ticket.issue_type, func.count(Ticket.ticket_id))
            .group_by(Ticket.issue_type)
            .order_by(func.count(Ticket.ticket_id).desc())
        ).all()
        return {str(name): int(count) for name, count in rows}

    def status_counts(self) -> dict[str, int]:
        rows = self.session.execute(
            select(Ticket.status, func.count(Ticket.ticket_id))
            .group_by(Ticket.status)
            .order_by(func.count(Ticket.ticket_id).desc())
        ).all()
        return {str(name): int(count) for name, count in rows}

    def daily_counts(self, *, date_from: Optional[datetime] = None) -> list[dict]:
        day = func.date(Ticket.created_at).label("day")
        statement = select(day, func.count(Ticket.ticket_id)).group_by(day).order_by(day)
        if date_from:
            statement = statement.where(Ticket.created_at >= date_from)
        return [
            {"date": str(row[0]), "count": int(row[1])}
            for row in self.session.execute(statement).all()
        ]

    def distinct_issue_types(self) -> list[str]:
        return [
            str(value)
            for value in self.session.scalars(
                select(Ticket.issue_type).distinct().order_by(Ticket.issue_type)
            )
        ]

    def distinct_statuses(self) -> list[str]:
        return [
            str(value)
            for value in self.session.scalars(
                select(Ticket.status).distinct().order_by(Ticket.status)
            )
        ]
