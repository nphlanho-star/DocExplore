"""
workers/tasks.py — Celery tasks cho toàn bộ document processing pipeline.

Pipeline sau khi upload:
  process_document_pipeline
      ├─ step_parse_document       (Docling → text + markdown)
      ├─ step_ocr_if_needed        (detect ảnh → PaddleOCR nếu có)
      ├─ step_chunk_document       (SentenceSplitter)
      ├─ step_embed_chunks         (BGE-M3 batch embedding)
      └─ step_index_to_qdrant      (upsert vào Qdrant)
"""
import os

# ── Tắt kiểm tra mạng HuggingFace Hub (version-check) mỗi lần load model ──────
# Phải set TRƯỚC bất kỳ import nào có thể kéo theo huggingface_hub/transformers.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio
import sys
import uuid
from datetime import datetime, timezone

# ── Windows fix: asyncio.run() trong Celery cần SelectorEventLoop ────────────
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from celery.utils.log import get_task_logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.workers.celery_app import celery_app
from app.config import get_settings
from app.models.document import Document, DocumentChunk, ProcessingStatus
from app.services.minio_service import minio_service
from app.services.document_parser import document_parser
from app.services.ocr_service import ocr_service
from app.services.chunking_service import chunking_service
from app.services.embedding_service import embedding_service
from app.services.qdrant_service import qdrant_service

logger = get_task_logger(__name__)

# ── Engine riêng cho Celery worker ────────────────────────────────────────────
# Không dùng chung AsyncSessionLocal của FastAPI: mỗi task ở đây gọi asyncio.run()
# tạo event loop MỚI, trong khi engine có connection pool tái sử dụng connection
# giữa các loop → lỗi "Event loop is closed" / "attached to a different loop".
# NullPool đảm bảo mỗi lần dùng xong connection sẽ đóng hẳn, không giữ lại pool.
_settings = get_settings()
_worker_engine = create_async_engine(_settings.DATABASE_URL, poolclass=NullPool)
WorkerSessionLocal = async_sessionmaker(_worker_engine, expire_on_commit=False)


# ── Helper: cập nhật trạng thái document trong DB ─────────────────────────────

def _sync_update_status(doc_id: str, status: str, error: str | None = None) -> None:
    """
    Celery tasks là sync (không phải async), nên dùng sync SQLAlchemy session.
    Tạo event loop tạm thời để chạy coroutine.
    """
    import asyncio

    async def _update():
        async with WorkerSessionLocal() as session:
            values = {"status": status}
            if error:
                values["error_message"] = error
            if status == ProcessingStatus.COMPLETED:
                values["processed_at"] = datetime.now(timezone.utc)
            await session.execute(
                update(Document)
                .where(Document.id == uuid.UUID(doc_id))
                .values(**values)
            )
            await session.commit()

    asyncio.run(_update())


def _sync_get_document(doc_id: str) -> dict:
    """Lấy thông tin document từ DB."""
    import asyncio

    async def _fetch():
        async with WorkerSessionLocal() as session:
            result = await session.execute(
                select(Document).where(Document.id == uuid.UUID(doc_id))
            )
            doc = result.scalar_one_or_none()
            if doc is None:
                raise ValueError(f"Document {doc_id} không tồn tại.")
            return {
                "id": str(doc.id),
                "owner_id": str(doc.owner_id),
                "original_filename": doc.original_filename,
                "file_extension": doc.file_extension,
                "minio_original_path": doc.minio_original_path,
            }

    return asyncio.run(_fetch())


def _sync_save_chunks(doc_id: str, chunks_data: list[dict]) -> list[str]:
    """Lưu DocumentChunk vào PostgreSQL, trả về danh sách chunk UUID string."""
    import asyncio

    async def _save():
        async with WorkerSessionLocal() as session:
            chunk_ids = []
            for c in chunks_data:
                chunk = DocumentChunk(
                    document_id=uuid.UUID(doc_id),
                    chunk_index=c["chunk_index"],
                    content=c["content"],
                    page_number=c.get("page_number"),
                    qdrant_point_id=c.get("qdrant_point_id"),
                    chunk_metadata=c.get("metadata"),
                )
                session.add(chunk)
                chunk_ids.append(str(chunk.id))
            await session.commit()
            return chunk_ids

    return asyncio.run(_save())


# ── Main pipeline task ────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.process_document_pipeline",
    max_retries=3,
    default_retry_delay=15,
)
def process_document_pipeline(self, document_id: str) -> dict:
    """
    Task tổng hợp — chạy toàn bộ pipeline xử lý tài liệu.

    Parameters
    ----------
    document_id : str  (UUID string)
    """
    logger.info(f"[Pipeline] Bắt đầu xử lý document {document_id}")

    try:
        # ── Cập nhật trạng thái PROCESSING ────────────────────────────
        _sync_update_status(document_id, ProcessingStatus.PROCESSING)
        doc = _sync_get_document(document_id)

        # ── Bước 1: Tải file từ MinIO ──────────────────────────────────
        logger.info(f"[1/5] Tải file từ MinIO: {doc['minio_original_path']}")
        file_bytes = minio_service.download_original(doc["minio_original_path"])

        # ── Bước 2: Parse bằng Docling ────────────────────────────────
        logger.info(f"[2/5] Parse tài liệu bằng Docling ({doc['file_extension']})")
        parse_result = document_parser.parse(file_bytes, doc["file_extension"])

        base_text = parse_result.text
        markdown_text = parse_result.markdown

        # ── Bước 3: OCR (chỉ khi Word có ảnh nhúng) ───────────────────
        logger.info("[3/5] Kiểm tra OCR…")
        final_text = base_text

        if doc["file_extension"].lower() in (".docx", ".doc"):
            ocr_result = ocr_service.process_docx(file_bytes)
            # ocr_service.process_docx() trả None nếu không có ảnh → skip
            if ocr_result is not None:
                final_text = ocr_service.merge_ocr_into_text(base_text, ocr_result)
                # Gộp cả vào markdown_text — đây mới là nguồn chính dùng để
                # chunking (chunk_document ưu tiên markdown_text). Trước đây
                # chỉ gộp vào final_text nên chữ OCR bị bỏ sót khi index.
                markdown_text = ocr_service.merge_ocr_into_text(markdown_text, ocr_result)
                # Lưu kết quả OCR lên MinIO
                minio_service.upload_ocr_result(
                    uuid.UUID(doc["owner_id"]),
                    uuid.UUID(doc["id"]),
                    ocr_result.combined_text,
                )
                logger.info(
                    f"  OCR: {ocr_result.image_count} ảnh, "
                    f"{'có' if ocr_result.has_content else 'không có'} chữ."
                )
            else:
                logger.info("  Bỏ qua OCR (Word không có ảnh nhúng).")
        else:
            logger.info(f"  Bỏ qua OCR (không phải Word: {doc['file_extension']}).")

        # Lưu processed text lên MinIO
        minio_service.upload_processed_text(
            uuid.UUID(doc["owner_id"]),
            uuid.UUID(doc["id"]),
            final_text,
        )

        # ── Bước 4: Chunking ──────────────────────────────────────────
        # chunk_document() tự chọn chiến lược phù hợp nhất theo thứ tự ưu
        # tiên: cấu trúc luật (Chương/Điều/Khoản) → heading Markdown →
        # cắt toàn văn bản theo câu.
        logger.info("[4/5] Chunking…")
        chunks = chunking_service.chunk_document(
            markdown_text=markdown_text,
            plain_text=final_text,
            document_id=uuid.UUID(document_id),
            extra_metadata={
                "original_filename": doc["original_filename"],
                "file_extension": doc["file_extension"],
                "owner_id": doc["owner_id"],
            },
        )

        logger.info(f"  {len(chunks)} chunk được tạo.")

        # ── Bước 5: Embedding + Index vào Qdrant ──────────────────────
        logger.info("[5/5] Embedding + Qdrant indexing…")
        texts = [c.content for c in chunks]
        vectors = embedding_service.embed_texts(texts)

        # Tạo UUID cho mỗi chunk (dùng làm Qdrant point ID)
        qdrant_ids = [str(uuid.uuid4()) for _ in chunks]

        payloads = [
            {
                "document_id": document_id,
                "owner_id": doc["owner_id"],
                "original_filename": doc["original_filename"],
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "content": c.content,
                **c.metadata,
            }
            for c in chunks
        ]

        qdrant_service.index_chunks(qdrant_ids, vectors, payloads)

        # Lưu chunk vào PostgreSQL
        chunks_data = [
            {
                "chunk_index": c.chunk_index,
                "content": c.content,
                "page_number": c.page_number,
                "metadata": c.metadata,
                "qdrant_point_id": q_id,
            }
            for c, q_id in zip(chunks, qdrant_ids)
        ]
        _sync_save_chunks(document_id, chunks_data)

        # ── Hoàn tất ──────────────────────────────────────────────────
        _sync_update_status(document_id, ProcessingStatus.COMPLETED)
        logger.info(f"[Pipeline] Document {document_id} xử lý thành công.")

        return {
            "status": "completed",
            "document_id": document_id,
            "chunks_count": len(chunks),
            "page_count": parse_result.page_count,
            "word_count": parse_result.word_count,
        }

    except Exception as exc:
        error_msg = str(exc)
        logger.error(f"[Pipeline] Lỗi xử lý document {document_id}: {error_msg}")
        _sync_update_status(document_id, ProcessingStatus.FAILED, error=error_msg)

        # Retry tự động
        raise self.retry(exc=exc)


# ── Task xóa document (cleanup) ───────────────────────────────────────────────

@celery_app.task(name="tasks.delete_document_data")
def delete_document_data(document_id: str, owner_id: str) -> dict:
    """
    Xóa toàn bộ dữ liệu của một document:
      - Vectors trong Qdrant
      - Files trong MinIO
    (PostgreSQL records được xóa bởi CASCADE trong DB)
    """
    logger.info(f"[Cleanup] Xóa dữ liệu document {document_id}")

    qdrant_service.delete_by_document(uuid.UUID(document_id))
    minio_service.delete_document(uuid.UUID(owner_id), uuid.UUID(document_id))

    logger.info(f"[Cleanup] Hoàn tất xóa document {document_id}")
    return {"status": "deleted", "document_id": document_id}
