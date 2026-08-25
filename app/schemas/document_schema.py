from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    message: str
    filename: str
    file_path: str
    index_message: str
    total_chunks: int
    supported_suffixes: List[str]


class DocumentItem(BaseModel):
    filename: str
    suffix: str
    size_bytes: int


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentItem]


class ChunkItem(BaseModel):
    source: str
    chunk_id: int
    content: str
    start: int = 0
    end: int = 0
    page_number: Optional[int] = None
    title: Optional[str] = None


class ChunkListResponse(BaseModel):
    total_chunks: int
    chunks: List[ChunkItem]


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(default=4, ge=1, le=20, description="返回最相关片段数量")
    min_score: float = Field(default=0.02, ge=0, le=1, description="最低融合分数")
    mode: Literal["hybrid", "dense", "bm25"] = Field(
        default="hybrid",
        description="dense=语义检索，bm25=关键词检索，hybrid=融合检索",
    )
    rerank: bool = Field(default=True, description="是否启用 Cross-Encoder 二阶段排序")
    candidate_k: Optional[int] = Field(
        default=None, ge=1, le=100, description="送入 Reranker 的候选数量；留空时自动计算"
    )


class SearchHit(BaseModel):
    source: str
    chunk_id: int
    content: str
    page_number: Optional[int] = None
    title: Optional[str] = None
    vector_score: float = 0.0
    bm25_score: float = 0.0
    overlap_score: float = 0.0
    base_score: float = 0.0
    rerank_score: float = 0.0
    rerank_raw_score: Optional[float] = None
    rerank_applied: bool = False
    final_score: float = 0.0
    retrieval_mode: str = "hybrid"


class SearchResponse(BaseModel):
    question: str
    top_k: int
    total: int
    results: List[SearchHit]


class IndexBuildResponse(BaseModel):
    message: str
    total_chunks: int
    dimension: Optional[int] = None
    method: str


class IndexStatusResponse(BaseModel):
    built: bool
    total_chunks: int
    method: str


class DocumentCatalogItem(BaseModel):
    document_id: str
    filename: str
    suffix: str
    storage_path: str
    size_bytes: int
    indexed: bool
    chunk_count: int
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentCatalogResponse(BaseModel):
    total: int
    documents: List[DocumentCatalogItem] = Field(default_factory=list)
