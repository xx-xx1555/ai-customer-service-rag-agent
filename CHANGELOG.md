# Changelog

## v0.7.0

- TXT 文档改为段落优先切分，减少不同主题混入同一 Chunk
- Reranker 输入增加 source、title 与 content 元数据
- 增加 Source / Evidence 双层检索评测
- 增加 12 道未参与调参的封存测试题与实验结果
- Hybrid + Rerank 在封存集上的 Evidence Hit@3 达到 83.33%，Evidence MRR@3 达到 0.7917
- 自动化测试提升至 21 项

## v0.6.0

- PostgreSQL + SQLAlchemy 数据持久化
- 工单 CRUD、筛选、趋势与周期对比
- 文档目录、RAG 会话和评测历史持久化
- Streamlit 运营看板、知识库、Agent、工单和评测页面
- Docker Compose 增加 PostgreSQL 和 Frontend
- 启动阶段增加 Qdrant 索引重试
- 自动化测试提升至 19 项

## v0.5.0

- Cross-Encoder Reranker
- LangGraph 多工具 Agent
- Agent 和检索评测

## v0.4.0

- BGE Embedding
- Qdrant
- BM25 + Dense 混合检索
