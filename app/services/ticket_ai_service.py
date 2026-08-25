import json
from typing import Dict

from app.services.llm_service import chat_completion, llm_available
from app.services.ticket_service import build_ticket_analysis_context


TICKET_REPORT_PROMPT = """
你是一个企业客服工单数据分析助手。
请基于给定的工单统计数据生成业务分析报告。

要求：
1. 只能基于给定数据，不要编造。
2. 输出四个部分：核心结论、主要问题、优先级建议、下一步动作。
3. 每个部分最多 3 条。
4. 语言要像项目演示汇报，不要写成论文。
""".strip()


def _fallback_report(context: Dict) -> str:
    summary = context.get("summary", {})
    top_issues = context.get("top_issues", {}).get("top_issue_types", {})
    unresolved = context.get("unresolved_tickets", [])
    low_rating = context.get("low_satisfaction_tickets", [])

    return (
        "一、核心结论\n"
        f"1. 当前共有 {summary.get('total_tickets', 0)} 条工单，平均满意度为 {summary.get('avg_satisfaction', 0)}。\n"
        f"2. 高频问题类型为：{top_issues}。\n"
        "二、主要问题\n"
        f"1. 未完全解决工单数量为 {len(unresolved)}。\n"
        f"2. 低满意度工单数量为 {len(low_rating)}。\n"
        "三、优先级建议\n"
        "1. 优先处理未解决和低满意度工单。\n"
        "2. 对高频问题类型建立知识库标准答复。\n"
        "四、下一步动作\n"
        "1. 将重复问题沉淀为 FAQ。\n"
        "2. 对处理时长和满意度持续监控。"
    )


def generate_ticket_ai_report() -> Dict:
    context = build_ticket_analysis_context()

    if not llm_available():
        return {
            "analysis_context": context,
            "ai_report": _fallback_report(context),
            "llm_used": False,
        }

    prompt = f"""
【工单统计数据】
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()

    report = chat_completion(
        messages=[
            {"role": "system", "content": TICKET_REPORT_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return {
        "analysis_context": context,
        "ai_report": report or _fallback_report(context),
        "llm_used": bool(report),
    }
