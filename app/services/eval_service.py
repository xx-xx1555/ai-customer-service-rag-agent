import json
import os
from typing import Dict, List

from app.services.agent_service import run_agent
from app.services.vector_service import search_vector_chunks

DEFAULT_EVAL_FILE = "data/eval_questions.json"
DEFAULT_AGENT_EVAL_FILE = "data/agent_eval_questions.json"


def _load_json_list(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    return data if isinstance(data, list) else []


def load_default_eval_items() -> List[Dict]:
    return _load_json_list(DEFAULT_EVAL_FILE)


def load_default_agent_eval_items() -> List[Dict]:
    return _load_json_list(DEFAULT_AGENT_EVAL_FILE)


def evaluate_rag(
    items: List[Dict],
    top_k: int = 4,
    mode: str = "hybrid",
    rerank: bool = True,
) -> Dict:
    """评估召回命中率、MRR 和关键词覆盖率。"""
    if not items:
        items = load_default_eval_items()

    cases = []
    source_hits = []
    reciprocal_ranks = []
    keyword_coverages = []
    top_scores = []
    evidence_hits = []
    evidence_reciprocal_ranks = []
    evidence_keyword_coverages = []

    for item in items:
        question = item.get("question", "")
        expected_source = item.get("expected_source")
        expected_keywords = item.get("expected_keywords", [])
        hits = search_vector_chunks(
            question=question,
            top_k=top_k,
            mode=mode,
            rerank=rerank,
        )

        top_sources = [f"{hit['source']}#chunk-{hit['chunk_id']}" for hit in hits]
        joined_content = "\n".join(hit["content"] for hit in hits)

        source_rank = None
        if expected_source:
            for index, source in enumerate(top_sources, start=1):
                if expected_source in source:
                    source_rank = index
                    break

        hit_source = source_rank is not None
        reciprocal_rank = 1.0 / source_rank if source_rank else 0.0

        if expected_keywords:
            matched = sum(keyword in joined_content for keyword in expected_keywords)
            keyword_coverage = matched / len(expected_keywords)
        else:
            keyword_coverage = 0.0

        evidence_rank = None
        best_evidence_keyword_coverage = 0.0

        for index, hit in enumerate(hits, start=1):
            content = str(hit.get("content", ""))

            if expected_keywords:
                matched = sum(
                    keyword in content
                    for keyword in expected_keywords
                )
                hit_keyword_coverage = matched / len(expected_keywords)
            else:
                hit_keyword_coverage = 1.0

            source_match = (
                not expected_source
                or str(hit.get("source", "")) == expected_source
            )

            if source_match:
                best_evidence_keyword_coverage = max(
                    best_evidence_keyword_coverage,
                    hit_keyword_coverage,
                )

            if (
                evidence_rank is None
                and source_match
                and hit_keyword_coverage >= 1.0
            ):
                evidence_rank = index

        evidence_hit = evidence_rank is not None
        evidence_reciprocal_rank = (
            1.0 / evidence_rank if evidence_rank else 0.0
        )
        top_score = float(hits[0]["final_score"]) if hits else 0.0
        source_hits.append(float(hit_source))
        reciprocal_ranks.append(reciprocal_rank)
        keyword_coverages.append(keyword_coverage)
        top_scores.append(top_score)
        evidence_hits.append(float(evidence_hit))
        evidence_reciprocal_ranks.append(evidence_reciprocal_rank)
        evidence_keyword_coverages.append(best_evidence_keyword_coverage)

        cases.append(
            {
                "question": question,
                "expected_source": expected_source,
                "hit_source": hit_source,
                "source_rank": source_rank,
                "reciprocal_rank": round(reciprocal_rank, 4),
                "keyword_coverage": round(keyword_coverage, 4),
                "top_sources": top_sources,
                "top_score": round(top_score, 4),
                "evidence_hit": evidence_hit,
                "evidence_rank": evidence_rank,
                "evidence_reciprocal_rank": round(evidence_reciprocal_rank, 4),
                "evidence_keyword_coverage": round(
                    best_evidence_keyword_coverage,
                    4,
                ),
            }
        )

    total = len(cases)
    hit_rate = round(sum(source_hits) / total, 4) if total else 0.0
    evidence_hit_rate = round(sum(evidence_hits) / total, 4) if total else 0.0

    evidence_mrr = (
        round(sum(evidence_reciprocal_ranks) / total, 4)
        if total
        else 0.0
    )

    avg_evidence_keyword_coverage = (
        round(sum(evidence_keyword_coverages) / total, 4)
        if total
        else 0.0
    )
    pipeline = f"{mode}{'+rerank' if rerank else ''}"
    return {
        "mode": mode,
        "rerank": rerank,
        "pipeline": pipeline,
        "total": total,
        "source_hit_rate": hit_rate,
        "hit_rate_at_k": hit_rate,
        "mrr_at_k": round(sum(reciprocal_ranks) / total, 4) if total else 0.0,
        "avg_keyword_coverage": round(sum(keyword_coverages) / total, 4) if total else 0.0,
        "avg_top_score": round(sum(top_scores) / total, 4) if total else 0.0,
        "cases": cases,
        "evidence_hit_rate_at_k": evidence_hit_rate,
        "evidence_mrr_at_k": evidence_mrr,
        "avg_evidence_keyword_coverage": avg_evidence_keyword_coverage,
    }


def compare_retrieval_modes(items: List[Dict], top_k: int = 4) -> Dict:
    """兼容第二阶段接口：对比三种一阶段检索，不启用 Reranker。"""
    return {
        "top_k": top_k,
        "results": {
            mode: evaluate_rag(items=items, top_k=top_k, mode=mode, rerank=False)
            for mode in ("bm25", "dense", "hybrid")
        },
    }


def compare_retrieval_pipelines(items: List[Dict], top_k: int = 4) -> Dict:
    """第三阶段核心对比：BM25、Dense、Hybrid、Hybrid + Reranker。"""
    configurations = {
        "bm25": {"mode": "bm25", "rerank": False},
        "dense": {"mode": "dense", "rerank": False},
        "hybrid": {"mode": "hybrid", "rerank": False},
        "hybrid_rerank": {"mode": "hybrid", "rerank": True},
    }
    return {
        "top_k": top_k,
        "results": {
            name: evaluate_rag(items=items, top_k=top_k, **config)
            for name, config in configurations.items()
        },
    }


def evaluate_agent(
    items: List[Dict],
    top_k: int = 4,
    rerank: bool = True,
    max_tool_calls: int = 4,
) -> Dict:
    if not items:
        items = load_default_agent_eval_items()

    cases = []
    intent_scores = []
    tool_recalls = []
    exact_matches = []
    answer_coverages = []

    for item in items:
        result = run_agent(
            question=item.get("question", ""),
            top_k=top_k,
            rerank=rerank,
            max_tool_calls=max_tool_calls,
        )
        expected_intent = item.get("expected_intent")
        expected_tools = list(item.get("expected_tools") or [])
        expected_keywords = list(item.get("expected_answer_keywords") or [])
        actual_tools = list(result.get("selected_tools") or [])
        actual_intent = str(result.get("intent", "unknown"))
        answer = str(result.get("answer", ""))

        intent_correct = not expected_intent or actual_intent == expected_intent
        expected_set = set(expected_tools)
        actual_set = set(actual_tools)
        tool_recall = (
            len(expected_set & actual_set) / len(expected_set) if expected_set else 1.0
        )
        exact_match = expected_set == actual_set if expected_set else not actual_set
        answer_coverage = (
            sum(keyword in answer for keyword in expected_keywords) / len(expected_keywords)
            if expected_keywords
            else 1.0
        )

        intent_scores.append(float(intent_correct))
        tool_recalls.append(tool_recall)
        exact_matches.append(float(exact_match))
        answer_coverages.append(answer_coverage)
        cases.append(
            {
                "question": item.get("question", ""),
                "expected_intent": expected_intent,
                "actual_intent": actual_intent,
                "intent_correct": intent_correct,
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "tool_recall": round(tool_recall, 4),
                "exact_tool_match": exact_match,
                "answer_keyword_coverage": round(answer_coverage, 4),
            }
        )

    total = len(cases)
    return {
        "total": total,
        "intent_accuracy": round(sum(intent_scores) / total, 4) if total else 0.0,
        "avg_tool_recall": round(sum(tool_recalls) / total, 4) if total else 0.0,
        "exact_tool_match_rate": round(sum(exact_matches) / total, 4) if total else 0.0,
        "avg_answer_keyword_coverage": round(sum(answer_coverages) / total, 4)
        if total
        else 0.0,
        "cases": cases,
    }
