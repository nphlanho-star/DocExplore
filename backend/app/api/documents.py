"""
api/documents.py — Upload, quản lý tài liệu, theo dõi trạng thái xử lý.
"""
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config import get_settings
from app.database import get_db
from app.models.document import Document, ProcessingStatus
from app.models.permission import Permission
from app.models.user import User
from app.schemas.document import (
    DocumentListResponse,
    DocumentRead,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from app.services.minio_service import minio_service
from app.workers.tasks import delete_document_data, process_document_pipeline

settings = get_settings()
router = APIRouter(prefix="/api/documents", tags=["Documents"])


# ── Helpers ───────────────────────────────────────────────────────────────────

MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


def _validate_upload(file: UploadFile) -> str:
    """Kiểm tra định dạng và trả về extension. Raise nếu không hợp lệ."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Định dạng '{ext}' không được hỗ trợ. Hỗ trợ: {settings.ALLOWED_EXTENSIONS}",
        )
    return ext


async def _check_document_access(
    doc_id: uuid.UUID,
    user: User,
    db: AsyncSession,
    require_owner: bool = False,
) -> Document:
    """Lấy document và kiểm tra quyền truy cập."""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Không tìm thấy tài liệu.")

    is_owner = doc.owner_id == user.id
    if require_owner and not is_owner and not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Không có quyền thực hiện.")

    if not is_owner and not user.is_admin:
        perm = await db.execute(
            select(Permission).where(
                Permission.document_id == doc_id,
                Permission.user_id == user.id,
            )
        )
        if perm.scalar_one_or_none() is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Không có quyền truy cập tài liệu này.")

    return doc


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload tài liệu:
      1. Validate định dạng & dung lượng.
      2. Lưu metadata vào PostgreSQL.
      3. Upload file gốc lên MinIO.
      4. Gửi Celery task để xử lý bất đồng bộ.
    """
    ext = _validate_upload(file)

    file_bytes = await file.read()
    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File vượt quá giới hạn {settings.MAX_FILE_SIZE_MB} MB.",
        )

    # Tạo bản ghi Document trước khi upload MinIO
    doc = Document(
        owner_id=current_user.id,
        original_filename=file.filename or "unknown",
        file_extension=ext,
        file_size_bytes=len(file_bytes),
        mime_type=file.content_type,
        status=ProcessingStatus.PENDING,
    )
    db.add(doc)
    await db.flush()   # lấy doc.id

    # Upload file gốc lên MinIO
    minio_path = minio_service.upload_original(
        user_id=current_user.id,
        doc_id=doc.id,
        filename=file.filename or "file",
        data=file_bytes,
        content_type=file.content_type or "application/octet-stream",
    )
    doc.minio_original_path = minio_path

    # Gửi Celery task (bất đồng bộ)
    task = process_document_pipeline.delay(str(doc.id))
    doc.celery_task_id = task.id

    return DocumentUploadResponse(
        document_id=doc.id,
        original_filename=doc.original_filename,
        status=doc.status,
        celery_task_id=task.id,
        message="Tài liệu đã được tiếp nhận và đang được xử lý.",
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Lấy danh sách tài liệu của user hiện tại (hoặc tất cả nếu là admin)."""
    offset = (page - 1) * page_size

    if current_user.is_admin:
        base_filter = select(Document)
        count_query = select(func.count()).select_from(Document)
    else:
        base_filter = select(Document).where(Document.owner_id == current_user.id)
        count_query = select(func.count()).select_from(Document).where(
            Document.owner_id == current_user.id
        )

    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    paginated = base_filter.offset(offset).limit(page_size)
    result = await db.execute(paginated)
    docs = result.scalars().all()

    return DocumentListResponse(items=docs, total=total, page=page, page_size=page_size)


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Lấy chi tiết một tài liệu."""
    return await _check_document_access(document_id, current_user, db)


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Theo dõi trạng thái xử lý của tài liệu."""
    doc = await _check_document_access(document_id, current_user, db)
    return DocumentStatusResponse(
        document_id=doc.id,
        status=doc.status,
        error_message=doc.error_message,
        processed_at=doc.processed_at,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Xóa tài liệu (chỉ owner hoặc admin). Cleanup MinIO + Qdrant bất đồng bộ."""
    doc = await _check_document_access(document_id, current_user, db, require_owner=True)

    # Xóa bản ghi PostgreSQL (CASCADE xóa chunks, permissions, …)
    await db.delete(doc)

    # Cleanup MinIO + Qdrant qua Celery (không block API response)
    delete_document_data.delay(str(doc.id), str(doc.owner_id))


@router.post("/{document_id}/share/{target_user_id}", status_code=status.HTTP_201_CREATED)
async def share_document(
    document_id: uuid.UUID,
    target_user_id: uuid.UUID,
    access_level: str = "read",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Chia sẻ tài liệu cho user khác."""
    await _check_document_access(document_id, current_user, db, require_owner=True)

    # Kiểm tra target_user tồn tại
    target = await db.execute(select(User).where(User.id == target_user_id))
    if target.scalar_one_or_none() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Người dùng không tồn tại.")

    perm = Permission(
        user_id=target_user_id,
        document_id=document_id,
        access_level=access_level,
    )
    db.add(perm)
    return {"message": f"Đã chia sẻ tài liệu cho user {target_user_id}."}
