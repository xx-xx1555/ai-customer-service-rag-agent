import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import UploadFile

from app.core.config import settings

SUPPORTED_SUFFIXES = {".txt", ".md", ".csv", ".pdf", ".docx"}


def ensure_upload_dir() -> None:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


def _safe_filename(filename: str) -> str:
    return Path(filename).name


async def save_upload_file(file: UploadFile) -> Dict:
    ensure_upload_dir()
    safe_name = _safe_filename(file.filename or "unknown.txt")
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型：{suffix}，当前支持：{sorted(SUPPORTED_SUFFIXES)}")

    content = await file.read()
    if not content:
        raise ValueError("文件内容为空")

    file_path = os.path.join(settings.UPLOAD_DIR, safe_name)
    with open(file_path, "wb") as file_handle:
        file_handle.write(content)

    return {"filename": safe_name, "file_path": file_path, "size_bytes": len(content)}


def list_documents() -> List[Dict]:
    ensure_upload_dir()
    documents = []
    for filename in sorted(os.listdir(settings.UPLOAD_DIR)):
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        if not os.path.isfile(file_path):
            continue
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        documents.append(
            {
                "filename": filename,
                "suffix": suffix,
                "size_bytes": os.path.getsize(file_path),
            }
        )
    return documents


def delete_document(filename: str) -> bool:
    safe_name = _safe_filename(filename)
    file_path = os.path.join(settings.UPLOAD_DIR, safe_name)
    if not os.path.exists(file_path):
        return False
    os.remove(file_path)
    return True


def _read_text_file(file_path: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(file_path, "r", encoding=encoding) as file_handle:
                return file_handle.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file_handle:
        return file_handle.read()


def _read_csv_file(file_path: str) -> str:
    dataframe = pd.read_csv(file_path)
    lines = []
    for idx, row in dataframe.iterrows():
        parts = [f"{column}: {row[column]}" for column in dataframe.columns]
        lines.append(f"第{idx + 1}行，" + "；".join(parts))
    return "\n".join(lines)


def _read_pdf_sections(file_path: str) -> List[Dict]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return []

    reader = PdfReader(file_path)
    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append({"content": text, "page_number": page_number})
    return sections


def _read_docx_file(file_path: str) -> str:
    try:
        from docx import Document
    except ImportError:
        return ""

    document = Document(file_path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n".join(paragraphs)


def _read_markdown_sections(file_path: str) -> List[Dict]:
    text = _read_text_file(file_path)
    sections: List[Dict] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            sections.append({"content": content, "title": current_title})
        current_lines = []

    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            flush()
            current_title = match.group(1).strip()
            current_lines.append(line)
        else:
            current_lines.append(line)
    flush()
    return sections or [{"content": text}]


def load_document_sections(file_path: str) -> List[Dict]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_sections(file_path)
    if suffix == ".md":
        return _read_markdown_sections(file_path)
    if suffix == ".csv":
        return [{"content": _read_csv_file(file_path)}]
    if suffix == ".docx":
        return [{"content": _read_docx_file(file_path)}]
    if suffix == ".txt":
        return _read_text_sections(file_path)
    return []


def load_document_text(file_path: str) -> str:
    return "\n\n".join(section["content"] for section in load_document_sections(file_path))


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_to_chunks(
    text: str,
    source: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    chunk_id_start: int = 1,
    page_number: int | None = None,
    title: str | None = None,
) -> List[Dict]:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap if overlap is not None else settings.CHUNK_OVERLAP
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")

    text = _clean_text(text)
    if not text:
        return []

    chunks: List[Dict] = []
    start = 0
    chunk_id = chunk_id_start
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        content = text[start:end].strip()
        if content:
            chunks.append(
                {
                    "source": source,
                    "chunk_id": chunk_id,
                    "content": content,
                    "start": start,
                    "end": end,
                    "page_number": page_number,
                    "title": title,
                }
            )
            chunk_id += 1

        if end >= text_len:
            break
        start = end - overlap

    return chunks


def load_all_chunks() -> List[Dict]:
    ensure_upload_dir()
    all_chunks: List[Dict] = []

    for filename in sorted(os.listdir(settings.UPLOAD_DIR)):
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        if not os.path.isfile(file_path):
            continue
        if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        next_chunk_id = 1
        for section in load_document_sections(file_path):
            chunks = split_text_to_chunks(
                text=section["content"],
                source=filename,
                chunk_id_start=next_chunk_id,
                page_number=section.get("page_number"),
                title=section.get("title"),
            )
            all_chunks.extend(chunks)
            next_chunk_id += len(chunks)

    return all_chunks


def search_relevant_chunks(question: str, top_k: int = 4) -> List[Dict]:
    """最简单的字符命中 baseline，仅用于教学对照。"""
    chunks = load_all_chunks()
    question_chars = set(re.sub(r"\s+", "", question))
    scored = []

    for chunk in chunks:
        content = chunk["content"]
        hit_count = sum(1 for char in question_chars if char and char in content)
        score = hit_count / max(len(question_chars), 1)
        if score > 0:
            item = dict(chunk)
            item["score"] = round(score, 4)
            scored.append(item)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]

def _read_text_sections(file_path: str) -> List[Dict]:
    text = _read_text_file(file_path)

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    return [{"content": paragraph} for paragraph in paragraphs]