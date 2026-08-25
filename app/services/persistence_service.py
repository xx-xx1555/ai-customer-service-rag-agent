from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import delete, select

from app.db.init_db import initialize_database
from app.db.models import ChatMessage, ChatSession, DocumentRecord, EvaluationRun
from app.db.session import session_scope


def _now() -> datetime:
    return datetime.now(timezone.utc)


def upsert_document_record(
    *,
    filename: str,
    storage_path: str,
    size_bytes: int,
    indexed: bool,
    chunk_count: int,
) -> Dict[str, Any]:
    initialize_database()
    with session_scope() as session:
        record = session.scalar(select(DocumentRecord).where(DocumentRecord.filename == filename))
        if record is None:
            record = DocumentRecord(
                document_id=f"DOC-{uuid4().hex}",
                filename=filename,
                suffix=Path(filename).suffix.lower(),
                storage_path=storage_path,
                size_bytes=size_bytes,
                indexed=indexed,
                chunk_count=chunk_count,
            )
            session.add(record)
        else:
            record.storage_path = storage_path
            record.size_bytes = size_bytes
            record.indexed = indexed
            record.chunk_count = chunk_count
            record.updated_at = _now()
        session.flush()
        return document_record_to_dict(record)


def delete_document_record(filename: str) -> bool:
    initialize_database()
    with session_scope() as session:
        record = session.scalar(select(DocumentRecord).where(DocumentRecord.filename == filename))
        if record is None:
            return False
        session.delete(record)
        return True


def list_document_records() -> List[Dict[str, Any]]:
    initialize_database()
    with session_scope() as session:
        records = list(session.scalars(select(DocumentRecord).order_by(DocumentRecord.created_at.desc())))
        return [document_record_to_dict(item) for item in records]


def document_record_to_dict(record: DocumentRecord) -> Dict[str, Any]:
    return {
        "document_id": record.document_id,
        "filename": record.filename,
        "suffix": record.suffix,
        "storage_path": record.storage_path,
        "size_bytes": record.size_bytes,
        "indexed": record.indexed,
        "chunk_count": record.chunk_count,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def save_chat_exchange(
    *,
    question: str,
    answer: str,
    sources: List[str],
    session_id: Optional[str] = None,
) -> str:
    initialize_database()
    resolved_session_id = session_id or f"CHAT-{uuid4().hex}"
    with session_scope() as session:
        chat_session = session.get(ChatSession, resolved_session_id)
        if chat_session is None:
            chat_session = ChatSession(
                session_id=resolved_session_id,
                title=question.strip()[:80] or "新会话",
            )
            session.add(chat_session)
            session.flush()
        chat_session.updated_at = _now()
        session.add_all(
            [
                ChatMessage(
                    session_id=resolved_session_id,
                    role="user",
                    content=question,
                    sources=[],
                ),
                ChatMessage(
                    session_id=resolved_session_id,
                    role="assistant",
                    content=answer,
                    sources=sources,
                ),
            ]
        )
    return resolved_session_id


def list_chat_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    initialize_database()
    with session_scope() as session:
        records = list(
            session.scalars(
                select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(max(1, min(limit, 200)))
            )
        )
        return [
            {
                "session_id": item.session_id,
                "title": item.title,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in records
        ]


def get_chat_session(session_id: str) -> Optional[Dict[str, Any]]:
    initialize_database()
    with session_scope() as session:
        chat_session = session.get(ChatSession, session_id)
        if chat_session is None:
            return None
        messages = list(
            session.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at, ChatMessage.message_id)
            )
        )
        return {
            "session_id": chat_session.session_id,
            "title": chat_session.title,
            "created_at": chat_session.created_at,
            "updated_at": chat_session.updated_at,
            "messages": [
                {
                    "message_id": item.message_id,
                    "role": item.role,
                    "content": item.content,
                    "sources": item.sources or [],
                    "created_at": item.created_at,
                }
                for item in messages
            ],
        }


def delete_chat_session(session_id: str) -> bool:
    initialize_database()
    with session_scope() as session:
        chat_session = session.get(ChatSession, session_id)
        if chat_session is None:
            return False
        session.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
        session.delete(chat_session)
        return True


def save_evaluation_run(
    *,
    evaluation_type: str,
    configuration: Dict[str, Any],
    result: Dict[str, Any],
) -> str:
    initialize_database()
    run_id = f"EVAL-{uuid4().hex}"
    metrics = {
        key: value
        for key, value in result.items()
        if key not in {"cases", "results"}
    }
    cases = result.get("cases", [])
    if "results" in result:
        cases = [
            {"pipeline": name, "result": pipeline_result}
            for name, pipeline_result in result["results"].items()
        ]
    with session_scope() as session:
        session.add(
            EvaluationRun(
                run_id=run_id,
                evaluation_type=evaluation_type,
                configuration=configuration,
                metrics=metrics,
                cases=cases,
            )
        )
    return run_id


def list_evaluation_runs(limit: int = 50) -> List[Dict[str, Any]]:
    initialize_database()
    with session_scope() as session:
        records = list(
            session.scalars(
                select(EvaluationRun)
                .order_by(EvaluationRun.created_at.desc())
                .limit(max(1, min(limit, 200)))
            )
        )
        return [
            {
                "run_id": item.run_id,
                "evaluation_type": item.evaluation_type,
                "configuration": item.configuration,
                "metrics": item.metrics,
                "created_at": item.created_at,
            }
            for item in records
        ]


def get_evaluation_run(run_id: str) -> Optional[Dict[str, Any]]:
    initialize_database()
    with session_scope() as session:
        item = session.get(EvaluationRun, run_id)
        if item is None:
            return None
        return {
            "run_id": item.run_id,
            "evaluation_type": item.evaluation_type,
            "configuration": item.configuration,
            "metrics": item.metrics,
            "cases": item.cases,
            "created_at": item.created_at,
        }
