import json
from functools import lru_cache
from typing import Any, Dict, List, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.core.config import settings
from app.services.llm_service import (
    chat_completion,
    json_chat_completion,
    llm_available,
)
from app.services.tools import TOOLS, execute_tool


class AgentState(TypedDict, total=False):
    question: str
    top_k: int
    rerank: bool
    max_tool_calls: int
    intent: str
    planner: str
    plan: List[Dict[str, Any]]
    next_tool_index: int
    selected_tools: List[str]
    tool_results: List[Dict[str, Any]]
    sources: List[str]
    steps: List[Dict[str, Any]]
    answer: str


PlanRoute = Literal["execute", "synthesize"]

TICKET_ANALYSIS_KEYWORDS = [
    "工单",
    "投诉",
    "满意度",
    "未解决",
    "处理中",
    "高频问题",
    "问题类型",
    "客服数据",
    "差评",
    "趋势",
    "环比",
]
KNOWLEDGE_KEYWORDS = [
    "知识库",
    "文档",
    "资料",
    "规定",
    "规则",
    "说明书",
    "参考来源",
    "检索片段",
    "结合知识",
]


def _step(
    steps: List[Dict[str, Any]], name: str, detail: str, status: str = "completed"
) -> List[Dict[str, Any]]:
    updated = list(steps)
    updated.append(
        {
            "step": len(updated) + 1,
            "name": name,
            "detail": detail,
            "status": status,
        }
    )
    return updated


def detect_intent(question: str) -> str:
    has_ticket = any(keyword in question for keyword in TICKET_ANALYSIS_KEYWORDS)
    has_knowledge = any(keyword in question for keyword in KNOWLEDGE_KEYWORDS)

    if has_ticket and has_knowledge:
        return "ticket_knowledge_analysis"
    if has_ticket:
        return "ticket_analysis"
    return "knowledge_qa"


def _plan_item(
    name: str, reason: str, arguments: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    return {
        "name": name,
        "arguments": arguments or {},
        "reason": reason,
    }


def _deduplicate_plan(
    plan: List[Dict[str, Any]], max_tool_calls: int
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in plan:
        name = str(item.get("name", ""))
        if name not in TOOLS or name in seen:
            continue
        seen.add(name)
        result.append(
            {
                "name": name,
                "arguments": dict(item.get("arguments") or {}),
                "reason": str(item.get("reason") or TOOLS[name].description),
            }
        )
        if len(result) >= max_tool_calls:
            break
    return result


def _rule_plan(
    question: str, intent: str, top_k: int, max_tool_calls: int
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []

    if intent == "knowledge_qa":
        if any(
            word in question.lower()
            for word in ["检索", "片段", "来源", "chunk", "召回", "rerank", "分数"]
        ):
            plan.append(
                _plan_item(
                    "vector_search", "检查知识库召回与排序结果", {"top_k": top_k}
                )
            )
        else:
            plan.append(
                _plan_item(
                    "knowledge_base_rag",
                    "从知识库检索并生成引用式回答",
                    {"top_k": top_k},
                )
            )
        return _deduplicate_plan(plan, max_tool_calls)

    if intent == "ticket_knowledge_analysis":
        plan.extend(
            [
                _plan_item(
                    "top_issue_types", "识别最常见的客服问题", {"top_k": min(top_k, 5)}
                ),
                _plan_item(
                    "low_satisfaction_tickets",
                    "识别投诉和低满意度风险",
                    {"threshold": 2},
                ),
                _plan_item(
                    "knowledge_base_rag", "结合知识库生成处理依据", {"top_k": top_k}
                ),
            ]
        )
        if any(word in question for word in ["报告", "分析", "建议", "优化"]):
            plan.append(_plan_item("ticket_ai_report", "汇总工单数据并生成业务建议"))
        return _deduplicate_plan(plan, max_tool_calls)

    # ticket_analysis
    if any(
        word in question.lower()
        for word in ["faq", "常见问题", "沉淀知识库", "知识库候选"]
    ):
        plan.extend(
            [
                _plan_item(
                    "top_issue_types", "确认高频问题类型", {"top_k": min(top_k, 5)}
                ),
                _plan_item(
                    "create_faq_candidates",
                    "生成待维护的 FAQ 候选",
                    {"top_k": min(top_k, 5)},
                ),
            ]
        )

    if any(word in question for word in ["趋势", "对比", "环比", "同比", "最近"]):
        plan.append(
            _plan_item("compare_ticket_periods", "比较前后两个时间周期", {"days": 30})
        )
    if any(word in question for word in ["未解决", "没解决", "处理中"]):
        plan.append(_plan_item("unresolved_tickets", "查看尚未解决的工单"))
    if any(word in question for word in ["低满意度", "差评", "不满意", "投诉"]):
        plan.append(
            _plan_item(
                "low_satisfaction_tickets", "识别低满意度风险工单", {"threshold": 2}
            )
        )
    if any(word in question for word in ["高频", "最多", "主要问题", "问题类型"]):
        plan.append(
            _plan_item("top_issue_types", "统计高频问题类型", {"top_k": min(top_k, 5)})
        )
    if any(word in question for word in ["概况", "统计", "总数", "平均"]):
        plan.append(_plan_item("ticket_summary", "汇总工单核心指标"))
    if any(word in question for word in ["报告", "分析", "建议", "优化"]):
        if not plan:
            plan.extend(
                [
                    _plan_item("ticket_summary", "获取整体工单指标"),
                    _plan_item(
                        "top_issue_types", "识别主要问题", {"top_k": min(top_k, 5)}
                    ),
                    _plan_item(
                        "low_satisfaction_tickets", "识别风险工单", {"threshold": 2}
                    ),
                ]
            )
        plan.append(_plan_item("ticket_ai_report", "生成业务分析报告"))

    if not plan:
        plan.append(_plan_item("ticket_keyword_search", "按用户问题中的关键词查询工单"))

    return _deduplicate_plan(plan, max_tool_calls)


def _llm_plan(
    question: str, intent: str, top_k: int, max_tool_calls: int
) -> List[Dict[str, Any]]:
    if not (settings.AGENT_USE_LLM_PLANNER and llm_available()):
        return []

    tool_descriptions = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in TOOLS.values()
    ]
    prompt = f"""
你是客服知识库 Agent 的任务规划器。把用户问题拆成最多 {max_tool_calls} 个必要工具调用。
工具可以连续调用，但不要选择重复工具，不要选择与问题无关的工具。
只输出合法 JSON：
{{
  "intent": "knowledge_qa|ticket_analysis|ticket_knowledge_analysis",
  "tools": [
    {{"name": "工具名", "arguments": {{}}, "reason": "选择原因"}}
  ]
}}

用户问题：{question}
初步意图：{intent}
默认 top_k：{top_k}
工具列表：
{json.dumps(tool_descriptions, ensure_ascii=False, indent=2)}
""".strip()

    data = json_chat_completion(
        messages=[
            {"role": "system", "content": "你只输出合法 JSON，不输出 Markdown。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return _deduplicate_plan(list(data.get("tools") or []), max_tool_calls)


def _plan_node(state: AgentState) -> AgentState:
    question = state["question"]
    intent = detect_intent(question)
    max_tool_calls = max(
        int(state.get("max_tool_calls", settings.AGENT_MAX_TOOL_CALLS)), 1
    )
    top_k = int(state.get("top_k", settings.DEFAULT_TOP_K))

    plan = _llm_plan(question, intent, top_k, max_tool_calls)
    planner = "llm" if plan else "rule"
    if not plan:
        plan = _rule_plan(question, intent, top_k, max_tool_calls)

    details = " → ".join(item["name"] for item in plan) or "无工具"
    return {
        "intent": intent,
        "planner": planner,
        "plan": plan,
        "next_tool_index": 0,
        "selected_tools": [],
        "tool_results": [],
        "sources": [],
        "steps": _step(
            state.get("steps", []),
            "task_planning",
            f"意图：{intent}；规划器：{planner}；工具链：{details}",
        ),
    }


def _source_labels(tool_name: str, result: Any) -> List[str]:
    if not isinstance(result, dict):
        return []
    if tool_name == "knowledge_base_rag":
        return [str(item) for item in result.get("sources", [])]
    if tool_name == "vector_search":
        labels = []
        for item in result.get("results", []):
            label = str(item.get("source", "unknown"))
            if item.get("page_number"):
                label += f"#page-{item['page_number']}"
            label += f"#chunk-{item.get('chunk_id', 0)}"
            labels.append(label)
        return labels
    return []


def _execute_tool_node(state: AgentState) -> AgentState:
    plan = state.get("plan", [])
    index = int(state.get("next_tool_index", 0))
    if index >= len(plan):
        return {}

    call = plan[index]
    tool_name = call["name"]
    arguments = dict(call.get("arguments") or {})
    selected_tools = list(state.get("selected_tools", []))
    tool_results = list(state.get("tool_results", []))
    sources = list(state.get("sources", []))
    steps = list(state.get("steps", []))

    try:
        result = execute_tool(
            tool_name=tool_name,
            question=state["question"],
            top_k=int(state.get("top_k", settings.DEFAULT_TOP_K)),
            arguments=arguments,
            rerank=bool(state.get("rerank", settings.AGENT_DEFAULT_RERANK)),
        )
        status = "completed"
        detail = f"工具 {tool_name} 执行完成"
    except Exception as exc:
        result = {"error": str(exc)}
        status = "failed"
        detail = f"工具 {tool_name} 执行失败：{exc}"

    selected_tools.append(tool_name)
    tool_results.append(
        {
            "tool_name": tool_name,
            "arguments": arguments,
            "reason": call.get("reason", ""),
            "status": status,
            "result": result,
        }
    )
    for source in _source_labels(tool_name, result):
        if source not in sources:
            sources.append(source)

    return {
        "next_tool_index": index + 1,
        "selected_tools": selected_tools,
        "tool_results": tool_results,
        "sources": sources,
        "steps": _step(steps, "tool_execution", detail, status=status),
    }


def _route_after_plan(state: AgentState) -> PlanRoute:
    return "execute" if state.get("plan") else "synthesize"


def _route_after_execution(state: AgentState) -> PlanRoute:
    if int(state.get("next_tool_index", 0)) < len(state.get("plan", [])):
        return "execute"
    return "synthesize"


def _fallback_synthesis(
    question: str, tool_results: List[Dict[str, Any]], sources: List[str]
) -> str:
    if not tool_results:
        return "没有找到可执行的工具，请换一种说法。"

    sections: List[str] = []
    for record in tool_results:
        name = record["tool_name"]
        result = record.get("result")
        if record.get("status") == "failed":
            sections.append(f"{name}：执行失败，{result.get('error', '未知错误')}。")
            continue

        if name == "knowledge_base_rag":
            sections.append(str(result.get("answer", "知识库未返回答案。")))
        elif name == "vector_search":
            hits = result.get("results", [])
            sections.append(f"知识库检索返回 {len(hits)} 个片段，已按相关性排序。")
        elif name == "ticket_summary":
            sections.append(
                f"工单概况：共 {result.get('total_tickets', 0)} 条，"
                f"平均满意度 {result.get('avg_satisfaction', 0)}，"
                f"平均解决时长 {result.get('avg_resolved_hours', 0)} 小时。"
            )
        elif name == "unresolved_tickets":
            sections.append(
                f"未解决或处理中工单共 {len(result.get('tickets', []))} 条。"
            )
        elif name == "top_issue_types":
            sections.append(f"高频问题类型：{result.get('top_issue_types', {})}。")
        elif name == "low_satisfaction_tickets":
            sections.append(f"低满意度风险工单共 {len(result.get('tickets', []))} 条。")
        elif name == "ticket_keyword_search":
            sections.append(
                f"关键词“{result.get('keyword', '')}”命中 {len(result.get('tickets', []))} 条工单。"
            )
        elif name == "ticket_ai_report":
            sections.append(str(result.get("ai_report", "工单报告生成完成。")))
        elif name == "compare_ticket_periods":
            sections.append(f"周期变化：{result.get('changes', {})}。")
        elif name == "create_faq_candidates":
            candidates = result.get("candidates", [])
            questions = [item.get("suggested_question", "") for item in candidates]
            sections.append(
                f"建议沉淀的 FAQ：{'；'.join(questions) if questions else '暂无'}。"
            )
        else:
            sections.append(f"{name} 已执行，结果：{result}")

    answer = "\n\n".join(section for section in sections if section)
    if sources and "参考来源" not in answer:
        answer += "\n\n参考来源：" + "、".join(sources)
    return answer


def _synthesize_node(state: AgentState) -> AgentState:
    tool_results = state.get("tool_results", [])
    sources = state.get("sources", [])
    question = state["question"]
    fallback = _fallback_synthesis(question, tool_results, sources)
    answer = fallback

    if llm_available() and tool_results:
        prompt = f"""
用户问题：{question}

工具执行结果：
{json.dumps(tool_results, ensure_ascii=False, indent=2, default=str)}

参考来源：{json.dumps(sources, ensure_ascii=False)}

请生成最终回答：
1. 先给结论，再给证据和行动建议。
2. 只能使用工具结果，不得编造。
3. 多工具结果要合并，而不是逐项机械复读。
4. 有来源时在末尾列出参考来源。
""".strip()
        generated = chat_completion(
            messages=[
                {"role": "system", "content": "你是严谨的客服知识库分析 Agent。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        if generated:
            answer = generated

    return {
        "answer": answer,
        "steps": _step(
            state.get("steps", []), "final_synthesis", "汇总全部工具结果并生成最终回答"
        ),
    }


@lru_cache(maxsize=1)
def get_agent_graph():
    builder = StateGraph(AgentState)
    builder.add_node("plan", _plan_node)
    builder.add_node("execute", _execute_tool_node)
    builder.add_node("synthesize", _synthesize_node)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan",
        _route_after_plan,
        {"execute": "execute", "synthesize": "synthesize"},
    )
    builder.add_conditional_edges(
        "execute",
        _route_after_execution,
        {"execute": "execute", "synthesize": "synthesize"},
    )
    builder.add_edge("synthesize", END)
    return builder.compile()


def run_agent(
    question: str,
    top_k: int = 4,
    rerank: bool | None = None,
    max_tool_calls: int | None = None,
) -> Dict[str, Any]:
    """LangGraph 主流程：规划 → 多工具循环执行 → 统一汇总。"""
    if not question.strip():
        raise ValueError("question 不能为空")

    max_calls = max_tool_calls or settings.AGENT_MAX_TOOL_CALLS
    initial_state: AgentState = {
        "question": question.strip(),
        "top_k": top_k,
        "rerank": settings.AGENT_DEFAULT_RERANK if rerank is None else rerank,
        "max_tool_calls": max(max_calls, 1),
        "steps": [],
    }
    result = get_agent_graph().invoke(
        initial_state,
        config={"recursion_limit": max(max_calls * 3 + 5, 20)},
    )

    selected_tools = list(result.get("selected_tools", []))
    tool_records = list(result.get("tool_results", []))
    if len(tool_records) == 1:
        compat_tool_result: Any = tool_records[0].get("result")
    else:
        compat_tool_result = {
            record["tool_name"]: record.get("result") for record in tool_records
        }

    return {
        "question": question,
        "intent": result.get("intent", "unknown"),
        "planner": result.get("planner", "rule"),
        "selected_tool": selected_tools[0] if selected_tools else "",
        "selected_tools": selected_tools,
        "plan": result.get("plan", []),
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "tool_result": compat_tool_result,
        "tool_results": tool_records,
        "steps": result.get("steps", []),
    }
