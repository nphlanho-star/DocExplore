"""
services/minio_service.py — Local filesystem storage (dev mode).

Giữ nguyên interface như MinIO SDK để không phải sửa documents.py / tasks.py.
Files được lưu vào: backend/storage/<bucket>/<user_id>/<doc_id>/<filename>
"""
import shutil
import uuid
from pathlib import Path

from loguru import logger

from app.config import get_settings

settings = get_settings()

# Thư mục gốc: D:\RAG-2\backend\storage\
STORAGE_ROOT = Path(__file__).parent.parent.parent / "storage"


class MinIOService:
    """Local filesystem storage — development mode."""

    def __init__(self) -> None:
        self._ensure_buckets()

    # ── Private ───────────────────────────────────────────────────────

    def _ensure_buckets(self) -> None:
        for bucket in (
            settings.MINIO_BUCKET_ORIGINAL,
            settings.MINIO_BUCKET_PROCESSED,
            settings.MINIO_BUCKET_OCR,
        ):
            (STORAGE_ROOT / bucket).mkdir(parents=True, exist_ok=True)
        logger.info(f"Local storage sẵn sàng tại: {STORAGE_ROOT.resolve()}")

    @staticmethod
    def _build_object_name(user_id: uuid.UUID, doc_id: uuid.UUID, filename: str) -> str:
        return f"{user_id}/{doc_id}/{filename}"

    def _object_path(self, bucket: str, object_name: str) -> Path:
        p = STORAGE_ROOT / bucket / object_name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ── Public ────────────────────────────────────────────────────────

    def upload_file(
        self,
        bucket: str,
        user_id: uuid.UUID,
        doc_id: uuid.UUID,
        filename: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        object_name = self._build_object_name(user_id, doc_id, filename)
        path = self._object_path(bucket, object_name)
        path.write_bytes(data)
        logger.debug(f"Saved {len(data)} bytes → {path}")
        return object_name

    def upload_original(
        self,
        user_id: uuid.UUID,
        doc_id: uuid.UUID,
        filename: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        return self.upload_file(
            settings.MINIO_BUCKET_ORIGINAL, user_id, doc_id, filename, data, content_type
        )

    def upload_processed_text(
        self,
        user_id: uuid.UUID,
        doc_id: uuid.UUID,
        text: str,
    ) -> str:
        return self.upload_file(
            settings.MINIO_BUCKET_PROCESSED,
            user_id,
            doc_id,
            "processed.txt",
            text.encode("utf-8"),
            "text/plain; charset=utf-8",
        )

    def upload_ocr_result(
        self,
        user_id: uuid.UUID,
        doc_id: uuid.UUID,
        text: str,
    ) -> str:
        return self.upload_file(
            settings.MINIO_BUCKET_OCR,
            user_id,
            doc_id,
            "ocr_result.txt",
            text.encode("utf-8"),
            "text/plain; charset=utf-8",
        )

    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        path = self._object_path(bucket, object_name)
        if not path.exists():
            raise FileNotFoundError(f"Object không tồn tại: {bucket}/{object_name}")
        data = path.read_bytes()
        logger.debug(f"Read {len(data)} bytes ← {path}")
        return data

    def download_original(self, object_name: str) -> bytes:
        return self.download_bytes(settings.MINIO_BUCKET_ORIGINAL, object_name)

    def delete_document(self, user_id: uuid.UUID, doc_id: uuid.UUID) -> None:
        for bucket in (
            settings.MINIO_BUCKET_ORIGINAL,
            settings.MINIO_BUCKET_PROCESSED,
            settings.MINIO_BUCKET_OCR,
        ):
            folder = STORAGE_ROOT / bucket / str(user_id) / str(doc_id)
            if folder.exists():
                shutil.rmtree(folder)
                logger.debug(f"Deleted folder: {folder}")

    def get_presigned_url(self, bucket: str, object_name: str, expires_seconds: int = 3600) -> str:
        return f"http://localhost:8000/storage/{bucket}/{object_name}"


# Singleton
minio_service = MinIOService()
