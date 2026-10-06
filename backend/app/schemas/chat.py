"""
schemas/chat.py — Pydantic schemas cho Chat.
"""
import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# ── Citation / Source ─────────────────────────────────────────────────────────

class CitationSource(BaseModel):
    document_id: uuid.UUID
    document_name: str
    chunk_index: int
    page_number: int | None
    content_snippet: str   # đoạn trích ngắn làm bằng chứng
    relevance_score: float


# ── Query / Answer ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: uuid.UUID | None = None          # None → tạo session mới
    document_ids: list[uuid.UUID] | None = None  # None → tìm trên tất cả tài liệu có quyền
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    session_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    sources: list[CitationSource]
    model_used: str


# ── Session / Message ─────────────────────────────────────────────────────────

class ChatSessionCreate(BaseModel):
    title: str | None = None


class ChatSessionRead(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatMessageCreate(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatMessageRead(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    sources: list[CitationSource] | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatHistoryResponse(BaseModel):
    session: ChatSessionRead
    messages: list[ChatMessageRead]
    memory_percent: int = 0   # mức đầy của bộ nhớ hội thoại (0–100)
