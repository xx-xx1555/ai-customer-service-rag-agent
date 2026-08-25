import csv
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock

from sqlalchemy import func, select, text

from app.core.config import settings
from app.db.base import Base
from app.db.models import Ticket
from app.db.session import SessionLocal, engine

logger = logging.getLogger(__name__)
_initialized = False
_init_lock = Lock()


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed


def seed_tickets_from_csv() -> int:
    path = Path(settings.TICKET_FILE)
    if not path.exists():
        logger.warning("工单种子文件不存在：%s", path)
        return 0

    with SessionLocal() as session:
        existing = session.scalar(select(func.count()).select_from(Ticket)) or 0
        if existing > 0:
            return 0

        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for item in csv.DictReader(file):
                rows.append(
                    Ticket(
                        ticket_id=item["ticket_id"],
                        user_id=item["user_id"],
                        issue_type=item["issue_type"],
                        description=item["description"],
                        status=item["status"],
                        created_at=_parse_datetime(item["created_at"]),
                        resolved_hours=float(item.get("resolved_hours") or 0),
                        satisfaction=int(float(item.get("satisfaction") or 0)),
                    )
                )
        session.add_all(rows)
        session.commit()
        logger.info("已从 CSV 导入 %s 条初始工单", len(rows))
        return len(rows)


def initialize_database(force: bool = False) -> None:
    global _initialized
    if _initialized and not force:
        return

    with _init_lock:
        if _initialized and not force:
            return
        if settings.AUTO_CREATE_TABLES:
            Base.metadata.create_all(bind=engine)
        if settings.AUTO_SEED_TICKETS:
            seed_tickets_from_csv()
        _initialized = True


def database_health() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("数据库健康检查失败")
        return False
