from app.core.config import settings
from app.services import agent_service, eval_service, vector_service
from app.services.reranker_service import RerankerService
from app.services.ticket_service import compare_ticket_periods, create_faq_candidates


def test_reranker_reorders_candidates(monkeypatch):
    service = RerankerService()
    monkeypatch.setattr(service, "score_pairs", lambda question, passages: [0.1, 2.5])

    old_enabled = settings.RERANKER_ENABLED
    old_weight = settings.RERANKER_WEIGHT
    try:
        settings.RERANKER_ENABLED = True
        settings.RERANKER_WEIGHT = 0.85
        results = service.rerank(
            "退款怎么办",
            [
                {
                    "source": "a",
                    "content": "普通内容",
                    "base_score": 1.0,
                    "final_score": 1.0,
                },
                {
                    "source": "b",
                    "content": "退款处理流程",
                    "base_score": 0.2,
                    "final_score": 0.2,
                },
            ],
            top_k=2,
        )
    finally:
        settings.RERANKER_ENABLED = old_enabled
        settings.RERANKER_WEIGHT = old_weight

    assert results[0]["source"] == "b"
    assert results[0]["rerank_applied"] is True
    assert results[0]["rerank_score"] == 1.0


def test_vector_search_calls_reranker(monkeypatch):
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

    class FakeReranker:
        def rerank(self, question, candidates, top_k):
            candidates = list(reversed(candidates))
            for index, candidate in enumerate(candidates):
                candidate["rerank_applied"] = True
                candidate["rerank_score"] = 1.0 - index * 0.1
                candidate["final_score"] = candidate["rerank_score"]
            return candidates[:top_k]

    monkeypatch.setattr(vector_service, "get_reranker_service", lambda: FakeReranker())
    results = vector_service.search_vector_chunks(
        question="测试",
        top_k=2,
        min_score=0.0,
        mode="hybrid",
        rerank=True,
    )

    assert results[0]["rerank_applied"] is True
    assert len(results) == 2


def test_langgraph_multi_tool_plan(monkeypatch):
    def fake_execute(tool_name, question, top_k, arguments, rerank):
        if tool_name == "knowledge_base_rag":
            return {
                "answer": "根据退款规则，应先核对订单与支付记录。",
                "sources": ["manual.md#退款规则#chunk-1"],
            }
        if tool_name == "top_issue_types":
            return {"top_issue_types": {"支付问题": 3}}
        if tool_name == "low_satisfaction_tickets":
            return {"tickets": [{"ticket_id": "T005"}]}
        if tool_name == "ticket_ai_report":
            return {"ai_report": "优先处理支付投诉，并补充标准答复。"}
        return {}

    monkeypatch.setattr(agent_service, "execute_tool", fake_execute)
    monkeypatch.setattr(agent_service, "llm_available", lambda: False)
    agent_service.get_agent_graph.cache_clear()

    result = agent_service.run_agent(
        "分析投诉问题，并结合知识库给出客服处理建议。",
        top_k=3,
        rerank=True,
        max_tool_calls=4,
    )

    assert result["intent"] == "ticket_knowledge_analysis"
    assert result["selected_tools"] == [
        "top_issue_types",
        "low_satisfaction_tickets",
        "knowledge_base_rag",
        "ticket_ai_report",
    ]
    assert "manual.md" in result["sources"][0]
    assert len(result["steps"]) == 6


def test_agent_evaluation_metrics(monkeypatch):
    monkeypatch.setattr(
        eval_service,
        "run_agent",
        lambda **kwargs: {
            "intent": "ticket_analysis",
            "selected_tools": ["unresolved_tickets"],
            "answer": "当前存在未解决工单。",
        },
    )
    result = eval_service.evaluate_agent(
        items=[
            {
                "question": "有哪些未解决工单？",
                "expected_intent": "ticket_analysis",
                "expected_tools": ["unresolved_tickets"],
                "expected_answer_keywords": ["未解决"],
            }
        ]
    )

    assert result["intent_accuracy"] == 1.0
    assert result["avg_tool_recall"] == 1.0
    assert result["exact_tool_match_rate"] == 1.0


def test_rag_evaluation_distinguishes_source_and_evidence(monkeypatch):
    fake_hits = [
        {
            "source": "test_knowledge.txt",
            "chunk_id": 2,
            "content": "RAG 包含文档上传、向量检索和大模型生成。",
            "final_score": 0.9,
        },
        {
            "source": "PHASE3_GUIDE.md",
            "chunk_id": 2,
            "content": "第三阶段增加检索评测。",
            "final_score": 0.8,
        },
        {
            "source": "test_knowledge.txt",
            "chunk_id": 5,
            "content": ("Agent 通常包含意图识别、工具选择、" "工具调用和结果汇总。"),
            "final_score": 0.7,
        },
    ]

    monkeypatch.setattr(
        eval_service,
        "search_vector_chunks",
        lambda **kwargs: fake_hits[: kwargs["top_k"]],
    )

    items = [
        {
            "question": "智能体通常经历哪四个环节？",
            "expected_source": "test_knowledge.txt",
            "expected_keywords": [
                "意图识别",
                "工具选择",
                "工具调用",
                "结果汇总",
            ],
        }
    ]

    top3_result = eval_service.evaluate_rag(
        items=items,
        top_k=3,
    )
    top3_case = top3_result["cases"][0]

    assert top3_case["hit_source"] is True
    assert top3_case["source_rank"] == 1

    assert top3_case["evidence_hit"] is True
    assert top3_case["evidence_rank"] == 3
    assert top3_case["evidence_reciprocal_rank"] == 0.3333

    assert top3_result["hit_rate_at_k"] == 1.0
    assert top3_result["evidence_hit_rate_at_k"] == 1.0
    assert top3_result["evidence_mrr_at_k"] == 0.3333

    top1_result = eval_service.evaluate_rag(
        items=items,
        top_k=1,
    )

    assert top1_result["hit_rate_at_k"] == 1.0
    assert top1_result["evidence_hit_rate_at_k"] == 0.0


def test_ticket_period_comparison_has_changes():
    result = compare_ticket_periods(days=5)
    assert "current_period" in result
    assert "previous_period" in result
    assert "changes" in result


def test_create_faq_candidates():
    result = create_faq_candidates(top_k=3)
    assert len(result["candidates"]) == 3
    assert all(item["evidence_count"] > 0 for item in result["candidates"])


def test_reranker_passage_contains_metadata(monkeypatch):
    service = RerankerService()
    captured = {}

    def fake_score_pairs(question, passages):
        captured["question"] = question
        captured["passages"] = passages
        return [1.0]

    monkeypatch.setattr(service, "score_pairs", fake_score_pairs)

    old_enabled = settings.RERANKER_ENABLED
    try:
        settings.RERANKER_ENABLED = True
        service.rerank(
            question="第二阶段使用什么模型？",
            candidates=[
                {
                    "source": "PHASE2_GUIDE.md",
                    "title": "启动",
                    "content": "需要下载 BAAI/bge-small-zh-v1.5。",
                    "base_score": 0.5,
                    "final_score": 0.5,
                }
            ],
            top_k=1,
        )
    finally:
        settings.RERANKER_ENABLED = old_enabled

    passage = captured["passages"][0]

    assert "来源：PHASE2_GUIDE.md" in passage
    assert "标题：启动" in passage
    assert "内容：需要下载 BAAI/bge-small-zh-v1.5。" in passage
