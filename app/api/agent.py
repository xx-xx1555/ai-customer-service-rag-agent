from typing import List

from fastapi import APIRouter

from app.schemas.agent_schema import AgentRequest, AgentResponse, ToolInfo
from app.services.agent_service import run_agent
from app.services.tools import list_tools


router = APIRouter(prefix="/agent", tags=["Agent"])


@router.get("/tools", response_model=List[ToolInfo])
def get_tools():
    return list_tools()


@router.post("/run", response_model=AgentResponse)
def agent_run(request: AgentRequest):
    return AgentResponse(**run_agent(
        question=request.question,
        top_k=request.top_k,
        rerank=request.rerank,
        max_tool_calls=request.max_tool_calls,
    ))


@router.post("/", response_model=AgentResponse)
def agent_chat_compat(request: AgentRequest):
    """兼容旧接口：/api/agent/。"""
    return AgentResponse(**run_agent(
        question=request.question,
        top_k=request.top_k,
        rerank=request.rerank,
        max_tool_calls=request.max_tool_calls,
    ))
