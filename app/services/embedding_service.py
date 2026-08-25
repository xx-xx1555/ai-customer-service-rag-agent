import logging
from functools import lru_cache
from typing import Iterable

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - 安装依赖后才会走真实模型
    SentenceTransformer = None


class EmbeddingService:
    """本地文本向量服务。

    BGE v1.5 在“短问题检索长文档”场景中，查询文本建议添加 instruction，
    文档文本不添加。所有向量做 L2 归一化，Qdrant 使用 Cosine 距离。
    """

    def __init__(self) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "缺少 sentence-transformers，请先执行：pip install sentence-transformers"
            )

        logger.info("正在加载 Embedding 模型：%s", settings.EMBEDDING_MODEL)
        try:
            self.model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE,
            )
        except Exception as exc:
            raise RuntimeError(
                "Embedding 模型加载失败。首次运行需要联网下载模型；"
                "也可以把 EMBEDDING_MODEL 改为本地模型目录。"
            ) from exc

        dimension = self.model.get_sentence_embedding_dimension()
        if not dimension:
            raise RuntimeError("无法读取 Embedding 模型维度")
        self.dimension = int(dimension)

    def encode_documents(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors = self.model.encode(
            values,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        prepared_query = f"{settings.EMBEDDING_QUERY_INSTRUCTION}{query.strip()}"
        vector = self.model.encode(
            prepared_query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vector, dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """延迟加载模型，避免 import 阶段就占用大量内存。"""
    return EmbeddingService()
