# RAG 检索与重排序实验报告

## 1. 实验目标

本实验用于验证智能客服知识库中的四条检索管线：

- BM25：关键词检索；
- Dense：BGE Embedding 与 Qdrant 向量检索；
- Hybrid：合并 Dense 与 BM25 候选并进行归一化、加权排序；
- Hybrid + Rerank：在 Hybrid 候选集上使用 Cross-Encoder 进行二阶段排序。

实验重点不是判断“是否找到了正确文件”，而是判断“是否找到了真正包含答案的 Chunk”，并验证改造在未参与调参的问题上是否仍然有效。

## 2. 原始问题

早期评测仅以 `expected_source` 判断命中，并把 Top-K 的多个 Chunk 正文拼接后计算关键词覆盖率。这会产生两类误判：

1. 检索到了正确文件中的错误 Chunk，仍会被记录为命中；
2. 多个错误 Chunk 分别包含部分关键词，拼接后可能形成虚假的完整证据。

实际案例中，问题“一个智能体处理任务时通常会经历哪四个环节？”曾将同一文件中介绍 RAG 流程的 Chunk 排在第一，而真正包含“意图识别、工具选择、工具调用、结果汇总”的 Agent Chunk 排在第三。旧指标会得到 Source MRR=1，但真实 Evidence MRR 只有 1/3。

## 3. 关键改造

### 3.1 TXT 按段落切分

TXT 文档由“整篇读取后固定字符切块”改为“先按空行分段，再对段落切块”，降低不同主题落入同一 Chunk 的概率。

### 3.2 Evidence 级评测

在原有 Source 指标之外增加：

- `evidence_hit_rate_at_k`：Top-K 中是否存在来源正确且完整包含预期关键词的单个 Chunk；
- `evidence_mrr_at_k`：第一个正确证据 Chunk 的平均倒数排名；
- `avg_evidence_keyword_coverage`：正确来源的候选中，单个 Chunk 的最佳关键词覆盖率。

当前 Evidence 命中条件为：来源匹配，并且同一个 Chunk 对 `expected_keywords` 的覆盖率达到 1.0。

### 3.3 Reranker 输入增加 Metadata

原先 Cross-Encoder 只接收 Chunk 正文，无法利用“第二阶段”“PHASE2_GUIDE.md”等文档级线索。改造后输入为：

```text
来源：{source}
标题：{title}
内容：{content}
```

这让 Reranker 可以结合来源、章节标题和正文判断问题与候选证据的相关性。

### 3.4 自动化回归测试

新增测试覆盖：

- 正确文件、错误 Chunk 与正确证据 Chunk 的区分；
- Source Hit 与 Evidence Hit 的差异；
- Reranker 实际输入是否包含 source、title 和 content；
- 改造后的语法和既有检索流程是否正常。

## 4. 数据集设计

### 4.1 开发集

- 题目数量：12；
- 用途：发现问题、调整切块、评测器和 Reranker 输入；
- 文件：`data/eval_questions.json`；
- 注意：该集合参与过调参，不能作为最终泛化结论的唯一依据。

### 4.2 封存测试集

- 题目数量：12；
- 用途：代码冻结后进行一次独立验证；
- 文件：`data/eval_questions_holdout.json`；
- 结果：`data/holdout_result.json`；
- 原则：如果继续根据这12题修改系统，该集合将转化为开发集，必须重新建立新的封存测试集。

两套数据均以 `expected_source` 标注正确文件，以 `expected_keywords` 标注一个正确 Chunk 应包含的核心证据。

## 5. 开发集实验结果

### 5.1 Metadata 改造前

| 管线 | Evidence Hit@3 | Evidence MRR@3 |
|---|---:|---:|
| BM25 | 0.7500 | 0.6250 |
| Dense | 0.6667 | 0.5833 |
| Hybrid | 0.6667 | 0.5833 |
| Hybrid + Rerank | 0.9167 | 0.7639 |

### 5.2 Metadata 改造后

Hybrid + Rerank 的结果变为：

| 指标 | 改造前 | 改造后 | 提升 |
|---|---:|---:|---:|
| Evidence Hit@3 | 0.9167 | 1.0000 | +0.0833（8.33个百分点） |
| Evidence MRR@3 | 0.7639 | 0.8889 | +0.1250 |
| Evidence 关键词覆盖率 | 0.9167 | 1.0000 | +0.0833（8.33个百分点） |

具体变化包括：

- BGE 模型名称问题：未命中变为正确证据第3名；
- 前端四条检索管线问题：正确证据由第2名升至第1名；
- Agent 四个环节问题：正确证据由第3名升至第1名；
- 开发集中没有出现 Evidence 排名回退。

开发集达到100%只能说明改造成功解决了当前已知问题，不能证明生产环境准确率为100%。

## 6. 封存测试集结果

| 管线 | Evidence Hit@3 | Evidence MRR@3 | Evidence Coverage | Source Hit@3 | Source MRR@3 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.7500 | 0.6667 | 0.7500 | 1.0000 | 0.8194 |
| Dense | 0.5833 | 0.5000 | 0.5833 | 0.7500 | 0.6111 |
| Hybrid | 0.6667 | 0.5139 | 0.6667 | 0.8333 | 0.6389 |
| Hybrid + Rerank | **0.8333** | **0.7917** | **0.8333** | **0.9167** | **0.8750** |

### 6.1 相比普通 Hybrid

- Evidence Hit@3：0.6667 → 0.8333，提升16.66个百分点；
- Evidence MRR@3：0.5139 → 0.7917，提升0.2778；
- 说明 Metadata 增强的 Cross-Encoder 在未见问题上仍然改善了正确证据的召回与排序。

### 6.2 相比 BM25

- Evidence Hit@3：0.7500 → 0.8333，提升8.33个百分点；
- Evidence MRR@3：0.6667 → 0.7917，提升0.1250；
- BM25 是本轮最强的单路检索，但 Hybrid + Rerank 的综合表现更好。

### 6.3 Source 与 Evidence 的差距

Hybrid + Rerank 的 Source Hit@3 为 11/12，Evidence Hit@3 为 10/12。这说明其中一道题虽然找到了正确文件，但没有找到该文件中真正包含完整答案的 Chunk。

## 7. 结论与边界

在12道未见问题的封存测试集上，Hybrid + Rerank 相比普通 Hybrid，Evidence Hit@3 从0.6667提升到0.8333，Evidence MRR@3 从0.5139提升到0.7917。

该结果说明：

- 二阶段排序不仅在参与调参的开发集上有效，在未见问题上也表现出一定泛化能力；
- Metadata 能为 Cross-Encoder 补充阶段、文件和章节信息；
- Evidence 级指标比单纯 Source 命中更接近“检索结果能否支持回答”。

当前不能宣称达到生产级准确率，原因包括：

- 封存集只有12道题，统计置信度有限；
- 文档和题目集中在单一项目领域；
- Evidence 标签依赖人工设计的精确关键词；
- 尚未系统评估端到端答案正确率、拒答质量、P95延迟、吞吐量和成本；
- 当前结果只验证 Top-3 检索证据，不代表大模型最终回答一定正确。

## 8. 可复现实验

在项目根目录执行：

```powershell
$itemsJson = Get-Content .\data\eval_questions_holdout.json -Raw -Encoding UTF8
$body = '{"items":' + $itemsJson + ',"top_k":3,"mode":"hybrid","rerank":true}'

$response = Invoke-RestMethod `
    -Uri "http://localhost:8000/api/eval/retrieval/compare" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body $body

$response |
    ConvertTo-Json -Depth 30 |
    Set-Content .\data\holdout_result.json -Encoding UTF8
```

为保证封存集独立性，正式报告中的结果应直接使用已经保存的 `holdout_result.json`，不应反复执行并根据结果调参。

## 9. 简历表述

> 为 Dense + BM25 混合检索链路增加 Metadata 增强的 Cross-Encoder 二阶段排序，并设计 Source/Evidence 双层评测与独立封存测试集；在12道未见问题上将 Evidence Hit@3 从66.67%提升至83.33%，Evidence MRR@3从0.5139提升至0.7917，同时通过自动化测试验证降级和证据判定逻辑。

## 10. 面试讲解框架

1. **问题**：只判断文件命中会把“正确文件中的错误 Chunk”当成正确答案。
2. **诊断**：真实案例出现 Source MRR=1、Evidence MRR=1/3。
3. **改造**：TXT段落切分、Evidence级指标、Reranker加入 source/title metadata。
4. **验证**：开发集用于调参，代码冻结后用12道新题建立封存测试集。
5. **结果**：封存集上 Hybrid + Rerank 相比 Hybrid 的 Hit@3提升16.66个百分点，MRR提升0.2778。
6. **边界**：样本仍小，尚未覆盖端到端生成质量、延迟、吞吐量和成本。
