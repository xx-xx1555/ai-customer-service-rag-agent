# 第三阶段改造指南：Reranker + LangGraph Agent + 评测

## 一、改造目标

第三阶段不再只做“检索后直接生成”，而是增加两个关键闭环：

```text
检索闭环：候选召回 -> Reranker -> 指标评测
Agent 闭环：任务规划 -> 多工具执行 -> 汇总 -> Agent 评测
```

## 二、建议阅读顺序

按以下顺序看代码：

1. `app/services/reranker_service.py`
2. `app/services/vector_service.py`
3. `app/services/tools.py`
4. `app/services/agent_service.py`
5. `app/services/eval_service.py`
6. `tests/test_phase3_reranker_agent.py`

## 三、Reranker 修改点

### 1. 候选数量

Reranker 不能只接收最终 `top_k`，否则没有重新排序的空间。

默认候选数：

```text
max(top_k * RERANKER_CANDIDATE_MULTIPLIER, 10)
```

### 2. 分数

- `base_score`：Dense/BM25 融合结果。
- `rerank_raw_score`：模型原始输出。
- `rerank_score`：候选集内归一化分数。
- `final_score`：Reranker 与基础分数加权结果。

### 3. 降级

模型无法加载时：

```text
rerank_applied=false
rerank_fallback_reason=unavailable
final_score=base_score
```

## 四、LangGraph 修改点

AgentState 保存：

```text
question
intent
plan
next_tool_index
selected_tools
tool_results
sources
steps
answer
```

状态图：

```text
plan -> execute -> execute -> ... -> synthesize
```

不要在一个节点里把所有事情做完，否则用了 LangGraph 也只是把普通函数塞进了图里，属于技术栈贴纸。

## 五、增加自定义工具

### 周期对比

```text
compare_ticket_periods
```

以 CSV 中的最新日期为锚点，对比当前周期与上一周期，避免演示数据因为历史日期导致全部落空。

### FAQ 候选

```text
create_faq_candidates
```

只生成候选问题和证据，不自动编造标准答案。标准答案应由知识库管理员确认。

## 六、评测

### 检索

```http
POST /api/eval/retrieval/compare
```

比较：

```text
bm25
dense
hybrid
hybrid_rerank
```

### Agent

```http
GET /api/eval/agent/default
```

评测文件：

```text
data/agent_eval_questions.json
```

## 七、测试与提交

```bash
pytest -q
```

建议提交顺序：

```text
feat: add cross encoder reranker service
refactor: add retrieve and rerank pipeline
feat: add langgraph multi tool agent workflow
feat: add ticket period comparison and faq candidate tools
feat: add retrieval and agent evaluation metrics
test: add phase3 reranker and agent tests
docs: add phase3 guide and api examples
```
