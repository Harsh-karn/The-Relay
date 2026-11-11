"""Pipeline schemas, trace models, and tool definitions."""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from the_relay.models import AnswerGrade, RequestStatus, SourceType
from the_relay.retrieval.engine import RetrievedChunk


class PipelineBranch(str, Enum):
    GROUNDED_ANSWER = "grounded_answer"
    REFUSAL = "refusal"
    DEGRADED_ANSWER = "degraded_answer"


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    output_summary: str
    duration_ms: float


class StepExecutionRecord(BaseModel):
    step_number: int
    step_name: str
    duration_ms: float
    details: Dict[str, Any] = Field(default_factory=dict)


class LatencyBreakdown(BaseModel):
    """Detailed latency dissection required for Section 4: Where the time goes."""
    waiting_ms: float = 0.0   # Time spent in queue before execution starts
    working_ms: float = 0.0   # Time spent in pipeline computation and model calls
    retrying_ms: float = 0.0  # Time spent waiting on retry backoffs
    first_token_ms: float = 0.0
    total_ms: float = 0.0
    delay_before_first_token: bool = True  # Whether predominant latency is before TTFT


class PipelineResult(BaseModel):
    """End-to-end execution result for a single question."""
    request_id: str
    question: str
    answer: str
    citations: List[str]
    branch_taken: PipelineBranch
    loop_count: int
    steps_executed: List[StepExecutionRecord]
    tool_calls: List[ToolCallRecord]
    latency: LatencyBreakdown
    status: RequestStatus
    source: SourceType
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    retrieval_setting_name: str = ""
    grade: Optional[AnswerGrade] = None
