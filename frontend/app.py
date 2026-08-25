from datetime import date, datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from api_client import API_BASE_URL, ApiError, delete, get, patch, post


st.set_page_config(
    page_title="智能客服知识库平台",
    page_icon="🧠",
    layout="wide",
)


STATUS_OPTIONS = ["待处理", "处理中", "未解决", "已解决"]


def show_error(exc: Exception) -> None:
    st.error(str(exc))


def format_ticket_rows(items: List[Dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    frame = pd.DataFrame(items)
    preferred = [
        "ticket_id",
        "user_id",
        "issue_type",
        "description",
        "status",
        "created_at",
        "resolved_hours",
        "satisfaction",
    ]
    columns = [item for item in preferred if item in frame.columns]
    return frame[columns]


def render_dashboard() -> None:
    st.header("客服运营看板")
    days = st.slider("周期长度（天）", min_value=7, max_value=90, value=30, step=1)
    try:
        data = get("/api/tickets/dashboard", params={"days": days})
    except ApiError as exc:
        show_error(exc)
        return

    summary = data["summary"]
    columns = st.columns(5)
    columns[0].metric("工单总数", summary["total_tickets"])
    columns[1].metric("平均满意度", summary["avg_satisfaction"])
    columns[2].metric("平均解决时长", f"{summary['avg_resolved_hours']} 小时")
    columns[3].metric("未解决率", f"{summary['unresolved_rate'] * 100:.1f}%")
    columns[4].metric("低满意度率", f"{summary['low_satisfaction_rate'] * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("问题类型分布")
        issue_frame = pd.DataFrame(
            list(summary["issue_type_counts"].items()), columns=["问题类型", "数量"]
        ).set_index("问题类型")
        st.bar_chart(issue_frame)
    with right:
        st.subheader("工单状态分布")
        status_frame = pd.DataFrame(
            list(summary["status_counts"].items()), columns=["状态", "数量"]
        ).set_index("状态")
        st.bar_chart(status_frame)

    st.subheader("每日工单趋势")
    trend_frame = pd.DataFrame(data.get("trends", []))
    if trend_frame.empty:
        st.info("暂无趋势数据。")
    else:
        trend_frame["date"] = pd.to_datetime(trend_frame["date"])
        st.line_chart(trend_frame.set_index("date")["count"])

    st.subheader("周期对比")
    comparison = data.get("period_comparison", {})
    current = comparison.get("current_period", {})
    previous = comparison.get("previous_period", {})
    changes = comparison.get("changes", {})
    compare_frame = pd.DataFrame(
        [
            {"周期": "当前周期", **current},
            {"周期": "上一周期", **previous},
        ]
    )
    if not compare_frame.empty:
        st.dataframe(compare_frame, use_container_width=True, hide_index=True)
    st.caption(f"变化值：{changes}")

    st.subheader("风险工单")
    st.dataframe(
        format_ticket_rows(summary.get("risk_tickets", [])),
        use_container_width=True,
        hide_index=True,
    )


def render_knowledge_base() -> None:
    st.header("知识库管理与 RAG 问答")
    upload_tab, documents_tab, chat_tab, history_tab, search_tab = st.tabs(
        ["上传文档", "文档列表", "智能问答", "会话历史", "检索调试"]
    )

    with upload_tab:
        file = st.file_uploader("上传 TXT、Markdown、PDF 或 DOCX", type=["txt", "md", "pdf", "docx"])
        if st.button("上传并重建索引", disabled=file is None, type="primary"):
            try:
                result = post(
                    "/api/documents/upload",
                    files={"file": (file.name, file.getvalue(), file.type)},
                    timeout=300,
                )
                st.success(result["message"])
                st.json(result)
            except ApiError as exc:
                show_error(exc)

        if st.button("手动重建知识库索引"):
            try:
                with st.spinner("正在生成向量并写入 Qdrant……"):
                    result = post("/api/documents/vector/build", timeout=600)
                st.success(result["message"])
                st.json(result)
            except ApiError as exc:
                show_error(exc)

    with documents_tab:
        try:
            result = get("/api/documents/list")
            documents = result.get("documents", [])
            st.dataframe(pd.DataFrame(documents), use_container_width=True, hide_index=True)
            try:
                catalog = get("/api/documents/catalog")
                if catalog.get("documents"):
                    with st.expander("查看数据库中的文档索引记录"):
                        st.dataframe(pd.DataFrame(catalog["documents"]), use_container_width=True)
            except ApiError:
                pass
            if documents:
                selected = st.selectbox("选择要删除的文档", [item["filename"] for item in documents])
                if st.button("删除文档", type="secondary"):
                    delete(f"/api/documents/{selected}")
                    st.success("文档已删除，索引已重建。")
                    st.rerun()
        except ApiError as exc:
            show_error(exc)

    with chat_tab:
        if "knowledge_session_id" not in st.session_state:
            st.session_state.knowledge_session_id = None
        session_col, reset_col = st.columns([4, 1])
        session_col.caption(
            f"当前会话：{st.session_state.knowledge_session_id or '尚未创建'}"
        )
        if reset_col.button("新会话"):
            st.session_state.knowledge_session_id = None
            st.rerun()
        question = st.text_area("输入知识库问题", placeholder="例如：退款需要哪些材料？")
        col1, col2, col3 = st.columns(3)
        mode = col1.selectbox("检索模式", ["hybrid", "dense", "bm25"])
        top_k = col2.slider("Top-K", 1, 10, 4)
        rerank = col3.checkbox("启用 Reranker", value=True)
        if st.button("生成回答", disabled=not question.strip(), type="primary"):
            try:
                with st.spinner("正在检索和生成回答……"):
                    result = post(
                        "/api/chat/",
                        json={
                            "question": question,
                            "session_id": st.session_state.knowledge_session_id,
                            "top_k": top_k,
                            "mode": mode,
                            "rerank": rerank,
                            "min_score": 0.02,
                        },
                        timeout=300,
                    )
                st.session_state.knowledge_session_id = result.get("session_id")
                st.markdown(result["answer"])
                if result.get("sources"):
                    st.caption("参考来源：" + "、".join(result["sources"]))
                with st.expander("查看检索上下文"):
                    st.dataframe(pd.DataFrame(result.get("contexts", [])), use_container_width=True)
            except ApiError as exc:
                show_error(exc)


    with history_tab:
        try:
            sessions = get("/api/conversations")
            if not sessions:
                st.info("暂无持久化会话。完成一次知识库问答后，这里会出现记录。")
            else:
                labels = {f"{item['title']}｜{item['session_id'][-8:]}": item for item in sessions}
                selected_label = st.selectbox("选择会话", list(labels))
                selected_session = labels[selected_label]
                detail = get(f"/api/conversations/{selected_session['session_id']}")
                for message in detail.get("messages", []):
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if message.get("sources"):
                            st.caption("来源：" + "、".join(message["sources"]))
                col_use, col_delete = st.columns(2)
                if col_use.button("继续该会话"):
                    st.session_state.knowledge_session_id = selected_session["session_id"]
                    st.success("已切换当前会话。")
                if col_delete.button("删除该会话"):
                    delete(f"/api/conversations/{selected_session['session_id']}")
                    if st.session_state.knowledge_session_id == selected_session["session_id"]:
                        st.session_state.knowledge_session_id = None
                    st.rerun()
        except ApiError as exc:
            show_error(exc)

    with search_tab:
        query = st.text_input("输入检索查询", key="search_query")
        mode = st.radio("模式", ["hybrid", "dense", "bm25"], horizontal=True)
        rerank = st.checkbox("二阶段排序", value=True, key="search_rerank")
        if st.button("执行检索", disabled=not query.strip()):
            try:
                result = post(
                    "/api/documents/vector/search",
                    json={
                        "question": query,
                        "top_k": 8,
                        "min_score": 0.0,
                        "mode": mode,
                        "rerank": rerank,
                    },
                    timeout=300,
                )
                st.dataframe(pd.DataFrame(result["results"]), use_container_width=True)
            except ApiError as exc:
                show_error(exc)


def render_agent() -> None:
    st.header("多工具客服 Agent")
    st.caption("Agent 会规划并执行知识库检索、工单统计、风险识别等工具。")

    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("例如：分析投诉问题，并结合知识库给出处理建议")
    if not question:
        return

    st.session_state.agent_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Agent 正在规划和调用工具……"):
                result = post(
                    "/api/agent/run",
                    json={
                        "question": question,
                        "top_k": 4,
                        "rerank": True,
                        "max_tool_calls": 6,
                    },
                    timeout=300,
                )
            st.markdown(result["answer"])
            st.caption(
                f"规划器：{result.get('planner')} ｜ 工具：{', '.join(result.get('selected_tools', [])) or '无'}"
            )
            with st.expander("查看执行轨迹"):
                st.dataframe(pd.DataFrame(result.get("steps", [])), use_container_width=True)
            with st.expander("查看工具结果"):
                st.json(result.get("tool_results", []))
            if result.get("sources"):
                st.caption("来源：" + "、".join(result["sources"]))
        st.session_state.agent_messages.append({"role": "assistant", "content": result["answer"]})
    except ApiError as exc:
        show_error(exc)


def render_tickets() -> None:
    st.header("工单管理")
    list_tab, create_tab, manage_tab, report_tab = st.tabs(
        ["工单列表", "创建工单", "更新与删除", "AI 分析报告"]
    )

    try:
        options = get("/api/tickets/filter-options")
    except ApiError:
        options = {"issue_types": [], "statuses": []}

    with list_tab:
        col1, col2, col3, col4 = st.columns(4)
        status_filter = col1.selectbox("状态", ["全部", *options.get("statuses", [])])
        issue_filter = col2.selectbox("问题类型", ["全部", *options.get("issue_types", [])])
        keyword = col3.text_input("关键词")
        satisfaction = col4.selectbox("满意度不高于", ["不限", 1, 2, 3, 4, 5])

        params: Dict[str, Any] = {"page": 1, "page_size": 100}
        if status_filter != "全部":
            params["status"] = status_filter
        if issue_filter != "全部":
            params["issue_type"] = issue_filter
        if keyword:
            params["keyword"] = keyword
        if satisfaction != "不限":
            params["satisfaction_lte"] = satisfaction

        try:
            result = get("/api/tickets", params=params)
            st.caption(f"共 {result['total']} 条")
            st.dataframe(format_ticket_rows(result["items"]), use_container_width=True, hide_index=True)
        except ApiError as exc:
            show_error(exc)

    with create_tab:
        with st.form("create_ticket"):
            col1, col2 = st.columns(2)
            user_id = col1.text_input("用户编号", value="U013")
            issue_type = col2.text_input("问题类型", value="知识库问题")
            description = st.text_area("问题描述")
            col3, col4, col5 = st.columns(3)
            status_value = col3.selectbox("状态", STATUS_OPTIONS)
            satisfaction = col4.slider("满意度", 1, 5, 3)
            resolved_hours = col5.number_input("解决时长（小时）", min_value=0.0, value=0.0)
            submitted = st.form_submit_button("创建工单", type="primary")
        if submitted:
            try:
                result = post(
                    "/api/tickets",
                    json={
                        "user_id": user_id,
                        "issue_type": issue_type,
                        "description": description,
                        "status": status_value,
                        "created_at": datetime.now().isoformat(),
                        "resolved_hours": resolved_hours,
                        "satisfaction": satisfaction,
                    },
                )
                st.success(f"已创建工单 {result['ticket_id']}")
                st.json(result)
            except ApiError as exc:
                show_error(exc)

    with manage_tab:
        ticket_id = st.text_input("工单编号", placeholder="例如 T001")
        if st.button("读取工单", disabled=not ticket_id.strip()):
            try:
                st.session_state.edit_ticket = get(f"/api/tickets/{ticket_id.strip()}")
            except ApiError as exc:
                show_error(exc)

        ticket = st.session_state.get("edit_ticket")
        if ticket:
            st.json(ticket)
            with st.form("update_ticket"):
                status_value = st.selectbox(
                    "更新状态",
                    STATUS_OPTIONS,
                    index=STATUS_OPTIONS.index(ticket["status"])
                    if ticket["status"] in STATUS_OPTIONS
                    else 0,
                )
                satisfaction = st.slider("更新满意度", 1, 5, int(ticket["satisfaction"]))
                resolved_hours = st.number_input(
                    "更新解决时长",
                    min_value=0.0,
                    value=float(ticket["resolved_hours"]),
                )
                description = st.text_area("更新描述", value=ticket["description"])
                update_submitted = st.form_submit_button("保存更新")
            if update_submitted:
                try:
                    updated = patch(
                        f"/api/tickets/{ticket['ticket_id']}",
                        json={
                            "status": status_value,
                            "satisfaction": satisfaction,
                            "resolved_hours": resolved_hours,
                            "description": description,
                        },
                    )
                    st.session_state.edit_ticket = updated
                    st.success("工单已更新。")
                    st.rerun()
                except ApiError as exc:
                    show_error(exc)

            if st.button("删除该工单", type="secondary"):
                try:
                    delete(f"/api/tickets/{ticket['ticket_id']}")
                    st.session_state.pop("edit_ticket", None)
                    st.success("工单已删除。")
                    st.rerun()
                except ApiError as exc:
                    show_error(exc)

    with report_tab:
        if st.button("生成工单 AI 报告", type="primary"):
            try:
                with st.spinner("正在分析工单数据……"):
                    result = get("/api/tickets/ai-report")
                st.markdown(result["ai_report"])
                with st.expander("查看分析上下文"):
                    st.json(result["analysis_context"])
            except ApiError as exc:
                show_error(exc)


def render_evaluation() -> None:
    st.header("RAG 与 Agent 评测")
    retrieval_tab, agent_tab = st.tabs(["检索 Pipeline", "Agent 工具规划"])

    with retrieval_tab:
        top_k = st.slider("检索 Top-K", 1, 10, 4, key="eval_top_k")
        if st.button("运行检索对比评测", type="primary"):
            try:
                with st.spinner("正在比较 BM25、Dense、Hybrid 和 Reranker……"):
                    result = post(
                        "/api/eval/retrieval/compare",
                        json={"items": [], "top_k": top_k, "mode": "hybrid", "rerank": True},
                        timeout=600,
                    )
                st.json(result)
                rows = []
                for pipeline, metrics in result.get("results", {}).items():
                    rows.append({
                        "pipeline": pipeline,
                        **{key: value for key, value in metrics.items() if key != "cases"},
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            except ApiError as exc:
                show_error(exc)

    with agent_tab:
        if st.button("运行 Agent 默认评测", type="primary"):
            try:
                with st.spinner("正在评估意图和工具规划……"):
                    result = get("/api/eval/agent/default", params={"max_tool_calls": 6})
                metrics = st.columns(4)
                metrics[0].metric("意图准确率", result.get("intent_accuracy", 0))
                metrics[1].metric("工具召回率", result.get("avg_tool_recall", 0))
                metrics[2].metric("完全匹配率", result.get("exact_tool_match_rate", 0))
                metrics[3].metric("关键词覆盖率", result.get("avg_answer_keyword_coverage", 0))
                st.dataframe(pd.DataFrame(result.get("cases", [])), use_container_width=True)
                try:
                    runs = get("/api/eval/runs", params={"limit": 10})
                    with st.expander("最近评测记录"):
                        st.dataframe(pd.DataFrame(runs.get("runs", [])), use_container_width=True)
                except ApiError:
                    pass
            except ApiError as exc:
                show_error(exc)


st.sidebar.title("智能客服平台")
page = st.sidebar.radio(
    "功能导航",
    ["运营看板", "知识库", "Agent 助手", "工单管理", "评测中心"],
)
st.sidebar.caption(f"Backend: {API_BASE_URL}")
try:
    health = get("/health")
    if health.get("status") == "ok":
        st.sidebar.success("后端与数据库正常")
    else:
        st.sidebar.warning(f"服务状态：{health.get('status')}")
except ApiError:
    st.sidebar.error("后端未连接")

if page == "运营看板":
    render_dashboard()
elif page == "知识库":
    render_knowledge_base()
elif page == "Agent 助手":
    render_agent()
elif page == "工单管理":
    render_tickets()
else:
    render_evaluation()
