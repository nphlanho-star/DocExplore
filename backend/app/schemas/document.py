"""
schemas/document.py — Pydantic schemas cho Document.
"""
import uuid
from datetime import datetime

from pydantic import BaseModel


class DocumentRead(BaseModel):
    id: uuid.UUID
    original_filename: str
    file_extension: str
    file_size_bytes: int
    mime_type: str | None
    status: str
    page_count: int | None
    word_count: int | None
    language: str | None
    created_at: datetime
    processed_at: datetime | None

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    original_filename: str
    status: str
    celery_task_id: str | None
    message: str


class DocumentListResponse(BaseModel):
    items: list[DocumentRead]
    total: int
    page: int
    page_size: int


class DocumentStatusResponse(BaseModel):
    document_id: uuid.UUID
    status: str
    error_message: str | None
    processed_at: datetime | None


class ChunkRead(BaseModel):
    id: uuid.UUID
    chunk_index: int
    content: str
    page_number: int | None
    chunk_metadata: dict | None

    model_config = {"from_attributes": True}


class ChunkListResponse(BaseModel):
    document_id: uuid.UUID
    total: int
    items: list[ChunkRead]
