import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Literal, Tuple

from app.core.config import settings
from app.repositories.qdrant_repository import get_qdrant_repository
from app.services.document_service import load_all_chunks
from app.services.embedding_service import get_embedding_service
from app.services.reranker_service import get_reranker_service

RetrievalMode = Literal["hybrid", "dense", "bm25"]
ChunkKey = Tuple[str, int]

chunk_store: List[Dict] = []
bm25_doc_tokens: List[List[str]] = []
bm25_doc_freq: Dict[str, int] = {}
bm25_avg_doc_len: float = 0.0
index_method: str = "not built"
indexed_chunk_count: int = 0

_PUNCT_PATTERN = re.compile(r"[\s\u3000，。！？、；：,.!?;:'\"()（）【】\[\]{}<>《》/\\|\-_=+*&#@`~]+")


def _chunk_key(chunk: Dict) -> ChunkKey:
    return str(chunk["source"]), int(chunk["chunk_id"])


def _tokenize(text: str) -> List[str]:
    """轻量中文 BM25 分词：英文按词、中文按字。它是 baseline，不冒充语义模型。"""
    text = text.lower()
    english_words = re.findall(r"[a-zA-Z0-9_]+", text)
    chinese_chars = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    other_chars = [ch for ch in _PUNCT_PATTERN.sub("", text) if ch and not ch.isascii()]
    return english_words + chinese_chars + other_chars


def _build_bm25_stats(chunks: List[Dict]) -> None:
    global bm25_doc_tokens, bm25_doc_freq, bm25_avg_doc_len

    bm25_doc_tokens = [_tokenize(chunk["content"]) for chunk in chunks]
    doc_freq = defaultdict(int)
    total_len = 0

    for tokens in bm25_doc_tokens:
        total_len += len(tokens)
        for token in set(tokens):
            doc_freq[token] += 1

    bm25_doc_freq = dict(doc_freq)
    bm25_avg_doc_len = total_len / max(len(bm25_doc_tokens), 1)


def _bm25_score(query: str, doc_index: int) -> float:
    if doc_index >= len(bm25_doc_tokens):
        return 0.0

    query_tokens = _tokenize(query)
    doc_tokens = bm25_doc_tokens[doc_index]
    if not query_tokens or not doc_tokens:
        return 0.0

    token_counts = Counter(doc_tokens)
    total_docs = len(bm25_doc_tokens)
    doc_len = len(doc_tokens)
    k1, b = 1.5, 0.75
    score = 0.0

    for token in query_tokens:
        df = bm25_doc_freq.get(token, 0)
        if df == 0:
            continue
        idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
        freq = token_counts.get(token, 0)
        numerator = freq * (k1 + 1)
        denominator = freq + k1 * (
            1 - b + b * doc_len / max(bm25_avg_doc_len, 1e-6)
        )
        score += idf * numerator / max(denominator, 1e-6)

    return float(score)


def _min_max_normalize(values: Dict[ChunkKey, float]) -> Dict[ChunkKey, float]:
    if not values:
        return {}
    min_v, max_v = min(values.values()), max(values.values())
    if abs(max_v - min_v) < 1e-9:
        return {key: (1.0 if value > 0 else 0.0) for key, value in values.items()}
    return {key: (value - min_v) / (max_v - min_v) for key, value in values.items()}


def build_vector_index() -> Dict:
    """重新读取全部文档，生成 BGE dense embedding，并写入 Qdrant。"""
    global chunk_store, index_method, indexed_chunk_count

    chunks = load_all_chunks()
    chunk_store = chunks
    _build_bm25_stats(chunks)

    if not chunks:
        try:
            get_qdrant_repository().rebuild([], vectors=[])
        except (RuntimeError, ConnectionError):
            pass
        index_method = "not built"
        indexed_chunk_count = 0
        return {
            "message": "没有可索引的文档 chunk",
            "total_chunks": 0,
            "dimension": None,
            "method": index_method,
        }

    embedding_service = get_embedding_service()
    vectors = embedding_service.encode_documents(chunk["content"] for chunk in chunks)
    total = get_qdrant_repository().rebuild(chunks, vectors)
    indexed_chunk_count = total

    index_method = "BGE dense embedding + Qdrant + local BM25 + Cross-Encoder reranker"
    return {
        "message": "Embedding、Qdrant 与本地 BM25 索引构建成功",
        "total_chunks": total,
        "dimension": embedding_service.dimension,
        "method": index_method,
    }


def get_index_status() -> Dict:
    return {
        "built": indexed_chunk_count > 0,
        "total_chunks": indexed_chunk_count,
        "method": index_method,
        "reranker_enabled": settings.RERANKER_ENABLED,
        "reranker_model": settings.RERANKER_MODEL,
    }


def _ensure_lexical_index() -> None:
    global chunk_store
    if chunk_store:
        return
    chunk_store = load_all_chunks()
    _build_bm25_stats(chunk_store)


def _dense_search(
    question: str,
    candidate_k: int,
) -> tuple[Dict[ChunkKey, float], Dict[ChunkKey, Dict]]:
    query_vector = get_embedding_service().encode_query(question)
    hits = get_qdrant_repository().search(query_vector=query_vector, limit=candidate_k)

    scores: Dict[ChunkKey, float] = {}
    chunks: Dict[ChunkKey, Dict] = {}
    for hit in hits:
        key = _chunk_key(hit)
        scores[key] = max(float(hit.get("score", 0.0)), 0.0)
        chunks[key] = hit
    return scores, chunks


def _bm25_search(
    question: str,
    candidate_k: int,
) -> tuple[Dict[ChunkKey, float], Dict[ChunkKey, Dict]]:
    _ensure_lexical_index()
    ranked = sorted(
        ((idx, _bm25_score(question, idx)) for idx in range(len(chunk_store))),
        key=lambda item: item[1],
        reverse=True,
    )[:candidate_k]

    scores: Dict[ChunkKey, float] = {}
    chunks: Dict[ChunkKey, Dict] = {}
    for idx, score in ranked:
        if score <= 0:
            continue
        chunk = chunk_store[idx]
        key = _chunk_key(chunk)
        scores[key] = score
        chunks[key] = chunk
    return scores, chunks


def _build_base_candidates(
    question: str,
    candidate_k: int,
    mode: RetrievalMode,
) -> List[Dict]:
    dense_raw: Dict[ChunkKey, float] = {}
    dense_chunks: Dict[ChunkKey, Dict] = {}
    bm25_raw: Dict[ChunkKey, float] = {}
    bm25_chunks: Dict[ChunkKey, Dict] = {}

    if mode in {"dense", "hybrid"}:
        dense_raw, dense_chunks = _dense_search(question, candidate_k)
    if mode in {"bm25", "hybrid"}:
        bm25_raw, bm25_chunks = _bm25_search(question, candidate_k)

    dense_scores = _min_max_normalize(dense_raw)
    bm25_scores = _min_max_normalize(bm25_raw)
    candidate_keys = set(dense_scores) | set(bm25_scores)
    results: List[Dict] = []

    for key in candidate_keys:
        chunk = dense_chunks.get(key) or bm25_chunks.get(key)
        if not chunk:
            continue

        dense_score = dense_scores.get(key, 0.0)
        bm25_score = bm25_scores.get(key, 0.0)

        if mode == "dense":
            base_score = dense_score
        elif mode == "bm25":
            base_score = bm25_score
        else:
            base_score = (
                settings.DENSE_WEIGHT * dense_score
                + settings.BM25_WEIGHT * bm25_score
            )

        results.append(
            {
                "source": chunk["source"],
                "chunk_id": int(chunk["chunk_id"]),
                "content": chunk["content"],
                "page_number": chunk.get("page_number"),
                "title": chunk.get("title"),
                "vector_score": round(dense_score, 4),
                "bm25_score": round(bm25_score, 4),
                "overlap_score": 0.0,
                "base_score": round(base_score, 4),
                "rerank_score": 0.0,
                "rerank_raw_score": None,
                "rerank_applied": False,
                "final_score": round(base_score, 4),
                "retrieval_mode": mode,
            }
        )

    results.sort(key=lambda item: item["base_score"], reverse=True)
    return results[:candidate_k]


def search_vector_chunks(
    question: str,
    top_k: int = 4,
    min_score: float = 0.02,
    mode: RetrievalMode = "hybrid",
    rerank: bool = True,
    candidate_k: int | None = None,
) -> List[Dict]:
    """先召回候选，再按需使用 Cross-Encoder 二阶段排序。"""
    if mode not in {"hybrid", "dense", "bm25"}:
        raise ValueError(f"不支持的检索模式：{mode}")
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")

    _ensure_lexical_index()
    if not chunk_store:
        return []

    default_candidates = max(
        top_k * max(settings.RERANKER_CANDIDATE_MULTIPLIER, 1),
        10,
    )
    candidate_limit = candidate_k or default_candidates
    candidate_limit = min(max(candidate_limit, top_k), len(chunk_store))

    candidates = _build_base_candidates(
        question=question,
        candidate_k=candidate_limit,
        mode=mode,
    )

    if rerank:
        ranked = get_reranker_service().rerank(
            question=question,
            candidates=candidates,
            top_k=top_k,
        )
    else:
        ranked = candidates[:top_k]

    return [item for item in ranked if float(item.get("final_score", 0.0)) >= min_score]
