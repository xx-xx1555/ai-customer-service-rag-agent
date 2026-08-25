from typing import Dict, List

from app.services.llm_service import chat_completion, llm_available
from app.services.vector_service import search_vector_chunks

RAG_SYSTEM_PROMPT = """
你是一个严谨的企业知识库问答助手，必须根据给定的【知识库片段】回答。

规则：
1. 只能使用知识库片段中的信息，不要编造。
2. 如果片段中没有答案，回答：知识库中没有找到相关信息。
3. 先给结论，再给依据。
4. 回答末尾必须列出“参考来源”，优先包含文件名、页码和 chunk 编号。
5. 语言简洁，适合项目演示。
""".strip()


def _source_label(hit: Dict) -> str:
    parts = [str(hit["source"])]
    if hit.get("page_number"):
        parts.append(f"page-{hit['page_number']}")
    if hit.get("title"):
        parts.append(str(hit["title"]))
    parts.append(f"chunk-{hit['chunk_id']}")
    return "#".join(parts)


def _build_context(hits: List[Dict]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(
            f"来源：{_source_label(hit)}\n"
            f"检索模式：{hit.get('retrieval_mode', 'hybrid')}\n"
            f"相关分数：{hit['final_score']}\n"
            f"内容：\n{hit['content']}"
        )
    return "\n\n---\n\n".join(blocks)


def _fallback_answer(question: str, hits: List[Dict]) -> str:
    if not hits:
        return "知识库中没有找到相关信息。"

    top = hits[0]
    sources = "、".join(_source_label(hit) for hit in hits)
    return (
        "结论：已在知识库中找到相关内容。\n"
        f"依据：最相关片段来自 {_source_label(top)}，内容为：{top['content'][:300]}。\n"
        f"参考来源：{sources}\n"
        "说明：当前未配置 DEEPSEEK_API_KEY，所以返回的是本地抽取式兜底结果。"
    )


def answer_question(
    question: str,
    top_k: int = 4,
    min_score: float = 0.02,
    mode: str = "hybrid",
    rerank: bool = True,
    candidate_k: int | None = None,
) -> Dict:
    hits = search_vector_chunks(
        question=question,
        top_k=top_k,
        min_score=min_score,
        mode=mode,
        rerank=rerank,
        candidate_k=candidate_k,
    )
    sources = [_source_label(hit) for hit in hits]

    if not hits:
        return {
            "question": question,
            "answer": "知识库中没有找到相关信息。",
            "sources": [],
            "contexts": [],
        }

    if not llm_available():
        return {
            "question": question,
            "answer": _fallback_answer(question, hits),
            "sources": sources,
            "contexts": hits,
        }

    context = _build_context(hits)
    user_prompt = f"""
【用户问题】
{question}

【知识库片段】
{context}
""".strip()

    answer = chat_completion(
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    if not answer:
        answer = _fallback_answer(question, hits)

    if "知识库中没有找到相关信息" in answer:
        sources = []

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "contexts": hits,
    }
