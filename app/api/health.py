from fastapi import APIRouter

from app.core.config import settings
from app.db.init_db import database_health


router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    database_ok = database_health()
    return {
        "status": "ok" if database_ok else "degraded",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "database": "ok" if database_ok else "unavailable",
    }
