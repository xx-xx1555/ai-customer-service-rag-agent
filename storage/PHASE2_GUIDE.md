# 第二阶段改造指南：Embedding + Qdrant + 混合检索

## 目标

把原来的“TF-IDF + FAISS”替换为真正的语义检索链路，同时保留 BM25 作为关键词 baseline：

```text
文档 -> 分块与 metadata -> BGE embedding -> Qdrant

用户问题 -> query embedding -> dense top-k
用户问题 -> BM25 -----------> lexical top-k
                         合并、归一化、加权融合 -> RAG
```

## 新增文件

- `app/services/embedding_service.py`：加载 BGE、生成文档/查询向量。
- `app/repositories/qdrant_repository.py`：封装 collection、upsert、search、count。
- `tests/test_phase2_retrieval.py`：验证 metadata、Qdrant、本地融合逻辑。

## 主要修改

- `vector_service.py`：支持 `dense`、`bm25`、`hybrid` 三种模式。
- `document_service.py`：PDF 保留页码，Markdown 保留标题。
- `docker-compose.yml`：新增 Qdrant 和持久化 volume。
- `/api/eval/rag/compare`：一次比较三种召回策略。

## 启动

推荐使用 Python 3.11 或 Docker。先复制环境变量：

```bash
cp .env.example .env
```

Windows CMD：

```bat
copy .env.example .env
```

Docker 启动：

```bash
docker compose up --build
```

第一次启动需要下载 `BAAI/bge-small-zh-v1.5`，之后模型缓存保存在 Docker volume。

## 接口测试顺序

1. `POST /api/documents/vector/build`
2. `POST /api/documents/vector/search`，分别传 `mode=bm25/dense/hybrid`
3. `POST /api/chat/`，使用 `mode=hybrid`
4. `POST /api/eval/rag/compare`

示例：

```json
{
  "question": "RAG 包含哪些步骤？",
  "top_k": 4,
  "min_score": 0.02,
  "mode": "hybrid"
}
```

## 建议提交顺序

```text
feat: add BGE embedding service
feat: add Qdrant repository and persistence
refactor: replace TF-IDF dense retrieval with Qdrant
feat: add dense bm25 hybrid retrieval modes
feat: preserve PDF page and Markdown title metadata
test: add phase2 retrieval tests
```
