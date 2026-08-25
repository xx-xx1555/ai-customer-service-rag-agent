import logging
import uuid
from functools import lru_cache
from typing import Dict, Iterable, List

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient, models
except ImportError:  # pragma: no cover
    QdrantClient = None
    models = None


class QdrantRepository:
    """Qdrant 的最小访问层，让业务代码不直接依赖 SDK 细节。"""

    def __init__(self) -> None:
        if QdrantClient is None or models is None:
            raise RuntimeError("缺少 qdrant-client，请先执行：pip install qdrant-client")

        if settings.QDRANT_LOCAL_PATH == ":memory:":
            self.client = QdrantClient(":memory:")
        elif settings.QDRANT_LOCAL_PATH:
            self.client = QdrantClient(path=settings.QDRANT_LOCAL_PATH)
        else:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
                timeout=settings.QDRANT_TIMEOUT,
            )
        self.collection_name = settings.QDRANT_COLLECTION

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def recreate_collection(self, vector_size: int) -> None:
        if self.collection_exists():
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    @staticmethod
    def _point_id(chunk: Dict) -> str:
        stable_key = (
            f"{chunk['source']}:{chunk['chunk_id']}:"
            f"{chunk.get('start', 0)}:{chunk.get('end', 0)}"
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))

    def rebuild(self, chunks: List[Dict], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunk 数量与向量数量不一致")
        if not chunks:
            if self.collection_exists():
                self.client.delete_collection(self.collection_name)
            return 0

        self.recreate_collection(vector_size=int(vectors.shape[1]))

        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "start": chunk.get("start", 0),
                "end": chunk.get("end", 0),
                "page_number": chunk.get("page_number"),
                "title": chunk.get("title"),
            }
            points.append(
                models.PointStruct(
                    id=self._point_id(chunk),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=settings.QDRANT_UPLOAD_BATCH_SIZE,
            wait=True,
        )
        return len(points)

    def search(self, query_vector: np.ndarray, limit: int) -> List[Dict]:
        if not self.collection_exists():
            return []

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True,
        )

        results: List[Dict] = []
        for point in response.points:
            payload = dict(point.payload or {})
            payload["point_id"] = str(point.id)
            payload["score"] = float(point.score)
            results.append(payload)
        return results

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )
        return int(result.count)


@lru_cache(maxsize=1)
def get_qdrant_repository() -> QdrantRepository:
    return QdrantRepository()
