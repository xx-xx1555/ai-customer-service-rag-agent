from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class EvalItem(BaseModel):
    question: str
    expected_source: Optional[str] = None
    expected_keywords: List[str] = Field(default_factory=list)


class EvalRequest(BaseModel):
    items: List[EvalItem] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=20)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    rerank: bool = True


class EvalCaseResult(BaseModel):
    evidence_hit: bool = False
    evidence_rank: Optional[int] = None
    evidence_reciprocal_rank: float = 0.0
    evidence_keyword_coverage: float = 0.0
    question: str
    expected_source: Optional[str]
    hit_source: bool
    source_rank: Optional[int] = None
    reciprocal_rank: float = 0.0
    keyword_coverage: float
    top_sources: List[str]
    top_score: float


class EvalResponse(BaseModel):
    evidence_hit_rate_at_k: float = 0.0
    evidence_mrr_at_k: float = 0.0
    avg_evidence_keyword_coverage: float = 0.0
    mode: str
    rerank: bool = False
    pipeline: str = ""
    total: int
    source_hit_rate: float
    hit_rate_at_k: float = 0.0
    mrr_at_k: float = 0.0
    avg_keyword_coverage: float
    avg_top_score: float
    cases: List[EvalCaseResult]


class EvalCompareResponse(BaseModel):
    top_k: int
    results: Dict[str, EvalResponse]


class AgentEvalItem(BaseModel):
    question: str
    expected_intent: Optional[str] = None
    expected_tools: List[str] = Field(default_factory=list)
    expected_answer_keywords: List[str] = Field(default_factory=list)


class AgentEvalRequest(BaseModel):
    items: List[AgentEvalItem] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=10)
    rerank: bool = True
    max_tool_calls: int = Field(default=4, ge=1, le=8)


class AgentEvalCaseResult(BaseModel):
    question: str
    expected_intent: Optional[str]
    actual_intent: str
    intent_correct: bool
    expected_tools: List[str]
    actual_tools: List[str]
    tool_recall: float
    exact_tool_match: bool
    answer_keyword_coverage: float


class AgentEvalResponse(BaseModel):
    total: int
    intent_accuracy: float
    avg_tool_recall: float
    exact_tool_match_rate: float
    avg_answer_keyword_coverage: float
    cases: List[AgentEvalCaseResult]
