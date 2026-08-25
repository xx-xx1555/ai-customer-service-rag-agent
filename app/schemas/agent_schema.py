from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)
    rerank: bool = True
    max_tool_calls: int = Field(default=4, ge=1, le=8)


class AgentStep(BaseModel):
    step: int
    name: str
    detail: str
    status: str = "completed"


class AgentPlanItem(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class AgentToolResult(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    status: str
    result: Any


class AgentResponse(BaseModel):
    question: str
    intent: str
    planner: str = "rule"
    selected_tool: str = ""
    selected_tools: List[str] = Field(default_factory=list)
    plan: List[AgentPlanItem] = Field(default_factory=list)
    answer: str
    sources: List[str] = Field(default_factory=list)
    tool_result: Optional[Any] = None
    tool_results: List[AgentToolResult] = Field(default_factory=list)
    steps: List[AgentStep] = Field(default_factory=list)


class ToolInfo(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, str]
