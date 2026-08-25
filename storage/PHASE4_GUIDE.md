# 第四阶段：PostgreSQL 与业务闭环

## 目标

把原先依赖 CSV 的工单 Demo 改为可持续运行的业务系统，并持久化文档目录、RAG 会话和评测结果。

## 核心变化

### 1. 配置管理

`app/core/config.py` 已改为 `pydantic-settings`，新增：

```env
DATABASE_URL=
DATABASE_ECHO=false
AUTO_CREATE_TABLES=true
AUTO_SEED_TICKETS=true
```

### 2. 数据库层

新增：

```text
app/db/base.py
app/db/models.py
app/db/session.py
app/db/init_db.py
```

Docker 使用 PostgreSQL；测试使用 SQLite 内存数据库。

### 3. 数据表

- `tickets`
- `documents`
- `chat_sessions`
- `chat_messages`
- `evaluation_runs`

Qdrant 继续负责向量和 chunk metadata，PostgreSQL 不做重复存储。

### 4. 工单 Repository 与 Service

新增：

```text
app/repositories/ticket_repository.py
```

`ticket_service.py` 已从 Pandas/CSV 查询改为 SQLAlchemy 数据访问，同时保留原 Agent 工具调用函数名，避免第三阶段接口被破坏。

### 5. CRUD

```http
POST   /api/tickets
GET    /api/tickets
GET    /api/tickets/{ticket_id}
PATCH  /api/tickets/{ticket_id}
DELETE /api/tickets/{ticket_id}
```

列表接口支持：

```text
status
issue_type
satisfaction_lte
date_from
date_to
keyword
page
page_size
```

### 6. 持久化业务记录

- 上传文档后写入 `documents`
- RAG 问答保存到 `chat_sessions` 和 `chat_messages`
- RAG/Agent 评测保存到 `evaluation_runs`

## 数据初始化

数据库为空且 `AUTO_SEED_TICKETS=true` 时，会从 `data/tickets.csv` 导入示例数据。

这意味着 CSV 只是种子，不再是运行时数据库。

## 验证

```bash
pytest -q tests/test_phase4_database_api.py
```

手动验证：

```bash
curl http://localhost:8000/api/tickets/dashboard
```
