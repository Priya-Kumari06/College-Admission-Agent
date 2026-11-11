from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentCategory(str, Enum):
    ADMISSION_POLICY = "admission_policy"
    ELIGIBILITY = "eligibility"
    COURSE_INFO = "course_info"
    FEE_STRUCTURE = "fee_structure"
    SCHOLARSHIP = "scholarship"
    DEADLINE = "deadline"
    FAQ = "faq"
    GENERAL = "general"


class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    original_name: str
    category: DocumentCategory
    status: DocumentStatus
    uploaded_by: str
    uploaded_at: datetime
    chunk_count: Optional[int] = 0
    file_size_kb: Optional[float] = 0


class DocumentListResponse(BaseModel):
    documents: List[DocumentUploadResponse]
    total: int
    page: int
    page_size: int


class FAQCreate(BaseModel):
    question: str = Field(..., min_length=10)
    answer: str = Field(..., min_length=10)
    category: DocumentCategory = DocumentCategory.FAQ
    tags: List[str] = []


class FAQResponse(BaseModel):
    id: str
    question: str
    answer: str
    category: DocumentCategory
    tags: List[str]
    created_at: datetime
    updated_at: datetime
