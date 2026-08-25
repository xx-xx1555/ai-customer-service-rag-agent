from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.schemas.eval_schema import (
    AgentEvalRequest,
    AgentEvalResponse,
    EvalCompareResponse,
    EvalRequest,
    EvalResponse,
)
from app.services.eval_service import (
    compare_retrieval_modes,
    compare_retrieval_pipelines,
    evaluate_agent,
    evaluate_rag,
)
from app.services.persistence_service import (
    get_evaluation_run,
    list_evaluation_runs,
    save_evaluation_run,
)

router = APIRouter(prefix="/eval", tags=["Evaluation"])


def _save(kind: str, configuration: dict, result: dict) -> dict:
    save_evaluation_run(
        evaluation_type=kind,
        configuration=configuration,
        result=result,
    )
    return result


@router.post("/rag", response_model=EvalResponse)
def eval_rag(request: EvalRequest):
    items = [item.model_dump() for item in request.items]
    result = evaluate_rag(
        items=items,
        top_k=request.top_k,
        mode=request.mode,
        rerank=request.rerank,
    )
    return _save("rag", request.model_dump(exclude={"items"}), result)


@router.get("/rag/default", response_model=EvalResponse)
def eval_rag_default(
    top_k: int = 4,
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    rerank: bool = True,
):
    result = evaluate_rag(items=[], top_k=top_k, mode=mode, rerank=rerank)
    return _save("rag_default", {"top_k": top_k, "mode": mode, "rerank": rerank}, result)


@router.post("/rag/compare", response_model=EvalCompareResponse)
def eval_rag_compare(request: EvalRequest):
    items = [item.model_dump() for item in request.items]
    result = compare_retrieval_modes(items=items, top_k=request.top_k)
    return _save("retrieval_modes", {"top_k": request.top_k}, result)


@router.post("/retrieval/compare", response_model=EvalCompareResponse)
def eval_retrieval_pipelines(request: EvalRequest):
    items = [item.model_dump() for item in request.items]
    result = compare_retrieval_pipelines(items=items, top_k=request.top_k)
    return _save("retrieval_pipelines", {"top_k": request.top_k}, result)


@router.post("/agent", response_model=AgentEvalResponse)
def eval_agent(request: AgentEvalRequest):
    items = [item.model_dump() for item in request.items]
    result = evaluate_agent(
        items=items,
        top_k=request.top_k,
        rerank=request.rerank,
        max_tool_calls=request.max_tool_calls,
    )
    return _save("agent", request.model_dump(exclude={"items"}), result)


@router.get("/agent/default", response_model=AgentEvalResponse)
def eval_agent_default(
    top_k: int = 4,
    rerank: bool = True,
    max_tool_calls: int = 4,
):
    result = evaluate_agent(
        items=[],
        top_k=top_k,
        rerank=rerank,
        max_tool_calls=max_tool_calls,
    )
    return _save(
        "agent_default",
        {"top_k": top_k, "rerank": rerank, "max_tool_calls": max_tool_calls},
        result,
    )


@router.get("/runs")
def evaluation_runs(limit: int = Query(default=50, ge=1, le=200)):
    return {"runs": list_evaluation_runs(limit=limit)}


@router.get("/runs/{run_id}")
def evaluation_run_detail(run_id: str):
    result = get_evaluation_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="评测记录不存在")
    return result
