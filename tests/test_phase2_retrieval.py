import numpy as np

from app.core.config import settings
from app.repositories.qdrant_repository import get_qdrant_repository
from app.services import vector_service
from app.services.document_service import split_text_to_chunks


def test_chunk_metadata_and_ids():
    chunks = split_text_to_chunks(
        text="第一段内容。第二段内容。第三段内容。",
        source="manual.pdf",
        chunk_size=8,
        overlap=2,
        chunk_id_start=5,
        page_number=3,
        title="退款规则",
    )

    assert chunks[0]["chunk_id"] == 5
    assert chunks[-1]["chunk_id"] >= 5
    assert all(chunk["page_number"] == 3 for chunk in chunks)
    assert all(chunk["title"] == "退款规则" for chunk in chunks)


def test_qdrant_repository_local_mode():
    old_path = settings.QDRANT_LOCAL_PATH
    try:
        settings.QDRANT_LOCAL_PATH = ":memory:"
        get_qdrant_repository.cache_clear()
        repository = get_qdrant_repository()
        chunks = [
            {"source": "a.txt", "chunk_id": 1, "content": "退款政策", "start": 0, "end": 4},
            {"source": "b.txt", "chunk_id": 1, "content": "登录问题", "start": 0, "end": 4},
        ]
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        assert repository.rebuild(chunks, vectors) == 2
        hits = repository.search(np.asarray([1.0, 0.0], dtype=np.float32), limit=2)

        assert repository.count() == 2
        assert hits[0]["source"] == "a.txt"
    finally:
        repository.client.close()
        settings.QDRANT_LOCAL_PATH = old_path
        get_qdrant_repository.cache_clear()


def test_hybrid_fusion(monkeypatch):
    chunks = [
        {"source": "dense.txt", "chunk_id": 1, "content": "语义相关内容"},
        {"source": "keyword.txt", "chunk_id": 1, "content": "关键词精确内容"},
    ]
    vector_service.chunk_store = chunks

    monkeypatch.setattr(
        vector_service,
        "_dense_search",
        lambda question, candidate_k: (
            {("dense.txt", 1): 0.95, ("keyword.txt", 1): 0.40},
            {("dense.txt", 1): chunks[0], ("keyword.txt", 1): chunks[1]},
        ),
    )
    monkeypatch.setattr(
        vector_service,
        "_bm25_search",
        lambda question, candidate_k: (
            {("dense.txt", 1): 0.10, ("keyword.txt", 1): 1.00},
            {("dense.txt", 1): chunks[0], ("keyword.txt", 1): chunks[1]},
        ),
    )

    old_dense_weight = settings.DENSE_WEIGHT
    old_bm25_weight = settings.BM25_WEIGHT
    try:
        settings.DENSE_WEIGHT = 0.7
        settings.BM25_WEIGHT = 0.3
        results = vector_service.search_vector_chunks(
            question="测试",
            top_k=2,
            min_score=0.0,
            mode="hybrid",
        )
    finally:
        settings.DENSE_WEIGHT = old_dense_weight
        settings.BM25_WEIGHT = old_bm25_weight

    assert results[0]["source"] == "dense.txt"
    assert results[0]["retrieval_mode"] == "hybrid"
    assert results[0]["vector_score"] == 1.0
