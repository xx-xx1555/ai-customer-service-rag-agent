from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from app.services.rag_service import answer_question
from app.services.ticket_ai_service import generate_ticket_ai_report
from app.services.ticket_service import (
    compare_ticket_periods,
    create_faq_candidates,
    get_low_satisfaction_tickets,
    get_ticket_summary,
    get_top_issue_types,
    get_unresolved_tickets,
    search_tickets_by_keyword,
)
from app.services.vector_service import search_vector_chunks


@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, str]
    func: Callable[..., Any]


def _rag_tool(
    question: str,
    top_k: int = 4,
    rerank: bool = True,
    mode: str = "hybrid",
) -> Dict:
    return answer_question(
        question=question,
        top_k=top_k,
        rerank=rerank,
        mode=mode,
    )


def _vector_search_tool(
    question: str,
    top_k: int = 4,
    rerank: bool = True,
    mode: str = "hybrid",
) -> Dict:
    return {
        "question": question,
        "results": search_vector_chunks(
            question=question,
            top_k=top_k,
            rerank=rerank,
            mode=mode,
        ),
    }


def _ticket_keyword_tool(question: str, keyword: str = "") -> Dict:
    if not keyword:
        for candidate in ["退款", "支付", "登录", "验证码", "上传", "加载", "搜索", "订单", "优惠券"]:
            if candidate in question:
                keyword = candidate
                break
    if not keyword:
        keyword = question[:10]

    return {
        "keyword": keyword,
        "tickets": search_tickets_by_keyword(keyword),
    }


TOOLS: Dict[str, Tool] = {
    "knowledge_base_rag": Tool(
        name="knowledge_base_rag",
        description="根据知识库片段回答问题，返回答案、来源和检索上下文。",
        input_schema={"question": "str", "top_k": "int", "rerank": "bool", "mode": "str"},
        func=_rag_tool,
    ),
    "vector_search": Tool(
        name="vector_search",
        description="只检索知识库片段，不生成回答，适合检查召回和 Reranker 排序。",
        input_schema={"question": "str", "top_k": "int", "rerank": "bool", "mode": "str"},
        func=_vector_search_tool,
    ),
    "ticket_summary": Tool(
        name="ticket_summary",
        description="统计工单总数、类型、状态、满意度和风险工单。",
        input_schema={},
        func=get_ticket_summary,
    ),
    "unresolved_tickets": Tool(
        name="unresolved_tickets",
        description="查询未解决或处理中的工单。",
        input_schema={},
        func=lambda: {"tickets": get_unresolved_tickets()},
    ),
    "top_issue_types": Tool(
        name="top_issue_types",
        description="查询最高频的问题类型。",
        input_schema={"top_k": "int"},
        func=get_top_issue_types,
    ),
    "low_satisfaction_tickets": Tool(
        name="low_satisfaction_tickets",
        description="查询低满意度工单，用于风险识别和投诉分析。",
        input_schema={"threshold": "int"},
        func=lambda threshold=2: {"tickets": get_low_satisfaction_tickets(threshold=threshold)},
    ),
    "ticket_keyword_search": Tool(
        name="ticket_keyword_search",
        description="按关键词搜索工单描述。",
        input_schema={"question": "str", "keyword": "str"},
        func=_ticket_keyword_tool,
    ),
    "ticket_ai_report": Tool(
        name="ticket_ai_report",
        description="基于工单统计数据生成业务分析报告。",
        input_schema={},
        func=generate_ticket_ai_report,
    ),
    "compare_ticket_periods": Tool(
        name="compare_ticket_periods",
        description="对比最近一个周期与上一周期的工单数量、满意度和解决时长。",
        input_schema={"days": "int"},
        func=compare_ticket_periods,
    ),
    "create_faq_candidates": Tool(
        name="create_faq_candidates",
        description="从高频工单中提取值得沉淀到知识库的 FAQ 候选问题。",
        input_schema={"top_k": "int"},
        func=create_faq_candidates,
    ),
}


def execute_tool(
    tool_name: str,
    question: str,
    top_k: int,
    arguments: Dict[str, Any] | None = None,
    rerank: bool = True,
) -> Any:
    """统一执行入口，负责过滤和补齐各工具参数。"""
    if tool_name not in TOOLS:
        raise ValueError(f"未知工具：{tool_name}")

    arguments = dict(arguments or {})
    tool = TOOLS[tool_name]

    if tool_name in {"knowledge_base_rag", "vector_search"}:
        return tool.func(
            question=str(arguments.get("question") or question),
            top_k=int(arguments.get("top_k", top_k)),
            rerank=bool(arguments.get("rerank", rerank)),
            mode=str(arguments.get("mode", "hybrid")),
        )
    if tool_name == "top_issue_types":
        return tool.func(top_k=int(arguments.get("top_k", top_k)))
    if tool_name == "low_satisfaction_tickets":
        return tool.func(threshold=int(arguments.get("threshold", 2)))
    if tool_name == "ticket_keyword_search":
        return tool.func(
            question=question,
            keyword=str(arguments.get("keyword", "")),
        )
    if tool_name == "compare_ticket_periods":
        return tool.func(days=int(arguments.get("days", 30)))
    if tool_name == "create_faq_candidates":
        return tool.func(top_k=int(arguments.get("top_k", top_k)))

    return tool.func()


def list_tools() -> List[Dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in TOOLS.values()
    ]
