import asyncio
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.conversations import router as conversation_router
from app.api.documents import router as document_router
from app.api.evaluation import router as eval_router
from app.api.health import router as health_router
from app.api.tickets import router as ticket_router
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.db.init_db import initialize_database
from app.services.vector_service import build_vector_index


setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        initialize_database()
    except Exception as exc:
        logger.exception("初始化数据库失败：%s", exc)

    if not settings.SKIP_INDEX_ON_STARTUP:
        retries = max(settings.INDEX_STARTUP_RETRIES, 1)
        for attempt in range(1, retries + 1):
            try:
                build_vector_index()
                break
            except Exception as exc:
                if attempt >= retries:
                    logger.exception("启动时构建检索索引失败：%s", exc)
                    break
                logger.warning(
                    "索引构建第 %s/%s 次失败，稍后重试：%s",
                    attempt,
                    retries,
                    exc,
                )
                await asyncio.sleep(settings.INDEX_RETRY_DELAY_SECONDS)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "基于 FastAPI、PostgreSQL、Qdrant、Cross-Encoder Reranker、"
        "LangGraph 多工具 Agent 与 Streamlit 的智能客服平台。"
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=settings.cors_origin_list != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router, prefix="/api")
app.include_router(conversation_router, prefix="/api")
app.include_router(document_router, prefix="/api")
app.include_router(ticket_router, prefix="/api")
app.include_router(agent_router, prefix="/api")
app.include_router(eval_router, prefix="/api")


@app.get("/")
def root():
    return {
        "message": "Smart Customer Support Knowledge Agent is running",
        "docs": "/docs",
        "health": "/health",
        "frontend": "http://localhost:8501",
    }
