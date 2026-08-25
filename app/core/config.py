from functools import cached_property
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_NAME: str = "智能客服知识库与工单分析平台"
    APP_VERSION: str = "0.7.0"
    DEBUG: bool = True
    SKIP_INDEX_ON_STARTUP: bool = False
    INDEX_STARTUP_RETRIES: int = 5
    INDEX_RETRY_DELAY_SECONDS: float = 2.0
    CORS_ORIGINS: str = "*"

    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    UPLOAD_DIR: str = "storage"
    TICKET_FILE: str = "data/tickets.csv"

    DATABASE_URL: str = "sqlite+pysqlite:///./data/app.db"
    DATABASE_ECHO: bool = False
    AUTO_CREATE_TABLES: bool = True
    AUTO_SEED_TICKETS: bool = True

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 80
    DEFAULT_TOP_K: int = 4
    DEFAULT_MIN_SCORE: float = 0.02

    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_QUERY_INSTRUCTION: str = "为这个句子生成表示以用于检索相关文章："

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "knowledge_chunks"
    QDRANT_TIMEOUT: float = 30
    QDRANT_UPLOAD_BATCH_SIZE: int = 64
    QDRANT_LOCAL_PATH: str = ""

    DENSE_WEIGHT: float = 0.70
    BM25_WEIGHT: float = 0.30

    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_DEVICE: str = "cpu"
    RERANKER_BATCH_SIZE: int = 16
    RERANKER_CANDIDATE_MULTIPLIER: int = 5
    RERANKER_WEIGHT: float = 0.85
    RERANKER_FAIL_OPEN: bool = True

    AGENT_MAX_TOOL_CALLS: int = 4
    AGENT_USE_LLM_PLANNER: bool = True
    AGENT_DEFAULT_RERANK: bool = True

    @cached_property
    def cors_origin_list(self) -> List[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [item.strip() for item in self.CORS_ORIGINS.split(",") if item.strip()]


settings = Settings()
