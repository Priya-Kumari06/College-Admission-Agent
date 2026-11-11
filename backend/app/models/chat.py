from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class QueryCategory(str, Enum):
    ADMISSION_POLICY = "admission_policy"
    ELIGIBILITY = "eligibility"
    COURSE_INFO = "course_info"
    FEE_STRUCTURE = "fee_structure"
    SCHOLARSHIP = "scholarship"
    DEADLINE = "deadline"
    GENERAL_FAQ = "general_faq"
    RESTRICTED = "restricted"
    OUT_OF_SCOPE = "out_of_scope"


class GuardrailStatus(str, Enum):
    PASSED = "passed"
    BLOCKED_INJECTION = "blocked_injection"
    BLOCKED_TOXICITY = "blocked_toxicity"
    BLOCKED_PII = "blocked_pii"
    BLOCKED_RESTRICTED = "blocked_restricted"
    LOW_CONFIDENCE = "low_confidence"
    HUMAN_HANDOFF = "human_handoff"


class SourceReference(BaseModel):
    document_name: str
    page_number: Optional[int] = None
    chunk_id: str
    relevance_score: float


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=2000)
    chat_history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    session_id: str
    message: str
    sources: List[SourceReference] = []
    confidence_score: float
    query_category: QueryCategory
    guardrail_status: GuardrailStatus
    human_handoff_required: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QueryLog(BaseModel):
    id: str
    session_id: str
    user_id: Optional[str]
    query: str
    response: str
    query_category: QueryCategory
    guardrail_status: GuardrailStatus
    confidence_score: float
    sources: List[SourceReference]
    human_handoff: bool
    timestamp: datetime
    response_time_ms: int
