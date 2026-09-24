"""
services/minio_service.py — Upload / download file từ MinIO.
"""
import io
import uuid
from pathlib import Path

from minio import Minio
from minio.error import S3Error
from loguru import logger

from app.config import get_settings

settings = get_settings()


class MinIOService:
    def __init__(self) -> None:
        self._client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self._ensure_buckets()

    # ── Private ───────────────────────────────────────────────────────

    def _ensure_buckets(self) -> None:
        """Tạo bucket nếu chưa tồn tại khi khởi động."""
        for bucket in (
            settings.MINIO_BUCKET_ORIGINAL,
            settings.MINIO_BUCKET_PROCESSED,
            settings.MINIO_BUCKET_OCR,
        ):
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
                logger.info(f"Created MinIO bucket: {bucket}")

    @staticmethod
    def _build_object_name(user_id: uuid.UUID, doc_id: uuid.UUID, filename: str) -> str:
        """Tổ chức object theo user/document để dễ quản lý."""
        return f"{user_id}/{doc_id}/{filename}"

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
        """Upload bytes lên MinIO, trả về object path."""
        object_name = self._build_object_name(user_id, doc_id, filename)
        self._client.put_object(
            bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        logger.debug(f"Uploaded {object_name} → bucket={bucket}")
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
        """Tải nội dung file về dạng bytes."""
        try:
            response = self._client.get_object(bucket, object_name)
            return response.read()
        except S3Error as exc:
            logger.error(f"MinIO download error: {exc}")
            raise

    def download_original(self, object_name: str) -> bytes:
        return self.download_bytes(settings.MINIO_BUCKET_ORIGINAL, object_name)

    def delete_document(self, user_id: uuid.UUID, doc_id: uuid.UUID) -> None:
        """Xóa tất cả object liên quan đến một document."""
        prefix = f"{user_id}/{doc_id}/"
        for bucket in (
            settings.MINIO_BUCKET_ORIGINAL,
            settings.MINIO_BUCKET_PROCESSED,
            settings.MINIO_BUCKET_OCR,
        ):
            objects = self._client.list_objects(bucket, prefix=prefix, recursive=True)
            for obj in objects:
                self._client.remove_object(bucket, obj.object_name)
                logger.debug(f"Deleted {obj.object_name} from bucket={bucket}")

    def get_presigned_url(self, bucket: str, object_name: str, expires_seconds: int = 3600) -> str:
        """Tạo presigned URL để frontend tải trực tiếp từ MinIO."""
        from datetime import timedelta
        return self._client.presigned_get_object(
            bucket, object_name, expires=timedelta(seconds=expires_seconds)
        )


# Singleton instance
minio_service = MinIOService()
