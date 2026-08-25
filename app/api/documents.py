from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.common_schema import MessageResponse
from app.schemas.document_schema import (
    ChunkListResponse,
    DocumentCatalogResponse,
    DocumentListResponse,
    IndexBuildResponse,
    IndexStatusResponse,
    SearchRequest,
    SearchResponse,
    UploadResponse,
)
from app.services.document_service import (
    SUPPORTED_SUFFIXES,
    delete_document,
    list_documents,
    load_all_chunks,
    save_upload_file,
    search_relevant_chunks,
)
from app.services.vector_service import build_vector_index, get_index_status, search_vector_chunks
from app.services.persistence_service import (
    delete_document_record,
    list_document_records,
    upsert_document_record,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


def _build_index_or_503():
    try:
        return build_vector_index()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"索引构建失败，请检查 Embedding 模型和 Qdrant：{exc}",
        ) from exc


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        saved = await save_upload_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    index_result = _build_index_or_503()
    source_chunk_count = sum(
        1 for chunk in load_all_chunks() if chunk.get("source") == saved["filename"]
    )
    upsert_document_record(
        filename=saved["filename"],
        storage_path=saved["file_path"],
        size_bytes=saved["size_bytes"],
        indexed=True,
        chunk_count=source_chunk_count,
    )
    return UploadResponse(
        message="文件上传成功，并已自动重建知识库索引",
        filename=saved["filename"],
        file_path=saved["file_path"],
        index_message=index_result.get("message", ""),
        total_chunks=index_result.get("total_chunks", 0),
        supported_suffixes=sorted(SUPPORTED_SUFFIXES),
    )


@router.get("/list", response_model=DocumentListResponse)
def get_documents():
    documents = list_documents()
    return DocumentListResponse(total=len(documents), documents=documents)


@router.get("/catalog", response_model=DocumentCatalogResponse)
def get_document_catalog():
    documents = list_document_records()
    return DocumentCatalogResponse(total=len(documents), documents=documents)


@router.delete("/{filename}", response_model=MessageResponse)
def remove_document(filename: str):
    ok = delete_document(filename)
    if not ok:
        raise HTTPException(status_code=404, detail="文件不存在")
    _build_index_or_503()
    delete_document_record(filename)
    return MessageResponse(message="文件已删除，并已重建索引")


@router.get("/chunks", response_model=ChunkListResponse)
def get_chunks():
    chunks = load_all_chunks()
    return ChunkListResponse(total_chunks=len(chunks), chunks=chunks)


@router.post("/search", response_model=SearchResponse)
def keyword_search(request: SearchRequest):
    """最简单的字符命中 baseline，主要用于教学对照。"""
    results = search_relevant_chunks(question=request.question, top_k=request.top_k)
    converted = [
        {
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "content": item["content"],
            "page_number": item.get("page_number"),
            "title": item.get("title"),
            "vector_score": 0.0,
            "bm25_score": 0.0,
            "overlap_score": item.get("score", 0.0),
            "base_score": item.get("score", 0.0),
            "rerank_score": 0.0,
            "rerank_raw_score": None,
            "rerank_applied": False,
            "final_score": item.get("score", 0.0),
            "retrieval_mode": "character-baseline",
        }
        for item in results
    ]
    return SearchResponse(
        question=request.question,
        top_k=request.top_k,
        total=len(converted),
        results=converted,
    )


@router.post("/vector/build", response_model=IndexBuildResponse)
def build_index():
    return _build_index_or_503()


@router.get("/vector/status", response_model=IndexStatusResponse)
def index_status():
    return get_index_status()


@router.post("/vector/search", response_model=SearchResponse)
def vector_search(request: SearchRequest):
    try:
        results = search_vector_chunks(
            question=request.question,
            top_k=request.top_k,
            min_score=request.min_score,
            mode=request.mode,
            rerank=request.rerank,
            candidate_k=request.candidate_k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"检索失败，请检查索引、Embedding 模型和 Qdrant：{exc}",
        ) from exc

    return SearchResponse(
        question=request.question,
        top_k=request.top_k,
        total=len(results),
        results=results,
    )
