"""
services/qdrant_service.py — Tương tác với Qdrant (lưu & tìm kiếm vector).
"""
import uuid
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings

settings = get_settings()


class QdrantService:
    def __init__(self) -> None:
        self._client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=30,
        )
        self._ensure_collection()

    # ── Setup ─────────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Tạo collection nếu chưa tồn tại."""
        existing = [c.name for c in self._client.get_collections().collections]
        if settings.QDRANT_COLLECTION not in existing:
            self._client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=settings.VECTOR_SIZE,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(f"Tạo Qdrant collection: {settings.QDRANT_COLLECTION}")

    # ── Index ─────────────────────────────────────────────────────────

    def index_chunks(
        self,
        chunk_ids: list[str],          # UUID string cho mỗi chunk
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """
        Lưu batch vectors vào Qdrant.

        Parameters
        ----------
        chunk_ids : list[str]
            Danh sách UUID string (point ID trong Qdrant).
        vectors : list[list[float]]
            Danh sách embedding vector tương ứng.
        payloads : list[dict]
            Metadata kèm theo mỗi vector (document_id, chunk_index, page, …).
        """
        if not chunk_ids:
            return

        points = [
            qmodels.PointStruct(
                id=cid,
                vector=vec,
                payload=meta,
            )
            for cid, vec, meta in zip(chunk_ids, vectors, payloads)
        ]

        self._client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
            wait=True,
        )
        logger.info(f"Indexed {len(points)} điểm vào Qdrant.")

    # ── Search ────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        document_ids: list[uuid.UUID] | None = None,
        user_id: uuid.UUID | None = None,
    ) -> list[dict[str, Any]]:
        """
        Tìm kiếm vector gần nhất, có filter theo document/user nếu truyền vào.

        Parameters
        ----------
        query_vector : list[float]
        top_k : int
        document_ids : list[uuid.UUID], optional
            Giới hạn tìm trong danh sách tài liệu cụ thể.
        user_id : uuid.UUID, optional
            Giới hạn tìm các chunk thuộc về user (quyền truy cập).

        Returns
        -------
        list[dict] — mỗi dict chứa id, score, payload.
        """
        search_filter = self._build_filter(document_ids, user_id)

        results = self._client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results
        ]

    @staticmethod
    def _build_filter(
        document_ids: list[uuid.UUID] | None,
        user_id: uuid.UUID | None,
    ) -> qmodels.Filter | None:
        """Xây dựng filter Qdrant theo document_ids và/hoặc user_id."""
        conditions: list[qmodels.Condition] = []

        if document_ids:
            conditions.append(
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchAny(any=[str(d) for d in document_ids]),
                )
            )

        if user_id:
            conditions.append(
                qmodels.FieldCondition(
                    key="owner_id",
                    match=qmodels.MatchValue(value=str(user_id)),
                )
            )

        if not conditions:
            return None

        return qmodels.Filter(must=conditions)

    # ── Delete ────────────────────────────────────────────────────────

    def delete_by_document(self, document_id: uuid.UUID) -> None:
        """Xóa tất cả vectors của một document."""
        self._client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
        )
        logger.info(f"Đã xóa vectors của document {document_id} khỏi Qdrant.")


# Singleton
qdrant_service = QdrantService()
