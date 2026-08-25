# 第五阶段：Streamlit 前端

## 目标

让项目从“只能看 Swagger”变成可以完整演示业务流程的产品原型。

## 文件

```text
frontend/app.py
frontend/api_client.py
frontend/requirements.txt
frontend/Dockerfile
```

## 页面

### 运营看板

- 工单总数
- 平均满意度
- 平均解决时长
- 未解决率
- 低满意度率
- 问题类型和状态分布
- 每日工单趋势
- 周期对比
- 风险工单

### 知识库

- 上传与删除文档
- 重建 Qdrant 索引
- 查看数据库文档目录
- Dense/BM25/Hybrid 检索调试
- RAG 问答
- 会话历史与继续对话

### Agent

- 多工具规划
- 工具执行轨迹
- 工具原始结果
- 引用来源

### 工单管理

- 工单筛选
- 创建工单
- 更新工单
- 删除工单
- AI 业务报告

### 评测中心

- BM25、Dense、Hybrid、Hybrid+Reranker 对比
- Agent 意图和工具规划评测
- 最近评测历史

## 启动

Docker：

```bash
docker compose up --build
```

本地：

```bash
API_BASE_URL=http://localhost:8000 streamlit run frontend/app.py
```

访问：

```text
http://localhost:8501
```
