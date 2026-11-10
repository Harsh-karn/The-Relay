"""Data models and schemas for The Relay."""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    REAL = "real"
    STAND_IN = "stand-in"


class RequestStatus(str, Enum):
    OK = "ok"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    ERROR = "error"


class AnswerGrade(str, Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    REFUSED_CORRECTLY = "refused_correctly"
    REFUSED_WRONGLY = "refused_wrongly"


class ChatMessage(BaseModel):
    role: str
    content: str


class StreamChunk(BaseModel):
    """A streaming token chunk emitted during generation."""
    delta: str
    index: int
    elapsed_ms: float
    is_first_token: bool = False
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    """Result of a model invocation (real or stand-in)."""
    text: str
    tokens_in: int
    tokens_out: int
    first_token_ms: float
    total_ms: float
    status: RequestStatus
    source: SourceType
    retries: int = 0
    error_message: Optional[str] = None
    rate_limit_headers: Optional[Dict[str, str]] = None


class LimitFoundRecord(BaseModel):
    """One row in Table 1: Limits you found in the real service."""
    limit_name: str
    value_found: str
    how_found: str
    confidence: str  # "high", "medium", "low"


class LatencyStats(BaseModel):
    """Percentile summary for latencies."""
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    sample_size: int


class ResultsCsvRow(BaseModel):
    """Strict schema corresponding to results.csv specified in the brief."""
    request_id: str
    scenario: str
    source: SourceType
    start_time: str
    first_token_ms: float
    total_ms: float
    status: RequestStatus
    retries: int
    tokens_in: int
    tokens_out: int
    retrieval_setting: str
    answer_grade: AnswerGrade

    def to_csv_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "scenario": self.scenario,
            "source": self.source.value,
            "start_time": self.start_time,
            "first_token_ms": round(self.first_token_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "status": self.status.value,
            "retries": self.retries,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "retrieval_setting": self.retrieval_setting,
            "answer_grade": self.answer_grade.value,
        }

