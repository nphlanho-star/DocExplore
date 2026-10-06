"""
services/qdrant_service.py — Tương tác với Qdrant (lưu & tìm kiếm vector).

Hỗ trợ 2 chế độ:
  - Dense-only  (HYBRID_SEARCH_ENABLED=False, mặc định): tương thích ngược hoàn toàn.
  - Hybrid      (HYBRID_SEARCH_ENABLED=True):  dense + sparse (BGE-M3 lexical),
                kết hợp bằng Reciprocal Rank Fusion (RRF) của Qdrant.

    ⚠️  Bật Hybrid lần đầu sẽ DROP và tạo lại collection → cần re-upload tài liệu.
"""
import uuid
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings

settings = get_settings()

# Tên named vector cho dense (chỉ dùng khi hybrid enabled — collection v2)
_DENSE_NAME = "dense"
_SPARSE_NAME = "sparse"


class QdrantService:
    def __init__(self) -> None:
        self._client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=30,
        )
        self._hybrid = settings.HYBRID_SEARCH_ENABLED
        self._ensure_collection()

    # ── Setup ─────────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """
        Tạo / kiểm tra collection.

        - Dense-only : collection dùng unnamed default vector (tương thích cũ).
        - Hybrid     : collection dùng named vectors {dense, sparse}.
                       Nếu collection hiện tại là dense-only → DROP và tạo lại.
        """
        existing_names = [c.name for c in self._client.get_collections().collections]
        col = settings.QDRANT_COLLECTION

        if col in existing_names:
            if self._hybrid and not self._collection_has_sparse(col):
                logger.warning(
                    f"HYBRID_SEARCH_ENABLED=True nhưng collection '{col}' chưa có "
                    "sparse vectors → XÓA và tạo lại. Cần re-upload toàn bộ tài liệu!"
                )
                self._client.delete_collection(col)
                existing_names.remove(col)
            elif self._hybrid:
                logger.info(f"Collection '{col}' đã có sparse vectors — OK.")
                return
            else:
                # Dense-only, collection đã tồn tại → không làm gì.
                return

        if self._hybrid:
            self._client.create_collection(
                collection_name=col,
                vectors_config={
                    _DENSE_NAME: qmodels.VectorParams(
                        size=settings.VECTOR_SIZE,
                        distance=qmodels.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    _SPARSE_NAME: qmodels.SparseVectorParams(
                        index=qmodels.SparseIndexParams(on_disk=False),
                    ),
                },
            )
            logger.info(f"Tạo Qdrant collection (hybrid): '{col}'")
        else:
            self._client.create_collection(
                collection_name=col,
                vectors_config=qmodels.VectorParams(
                    size=settings.VECTOR_SIZE,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(f"Tạo Qdrant collection (dense-only): '{col}'")

    def _collection_has_sparse(self, col: str) -> bool:
        """Kiểm tra collection có cấu hình sparse vectors không."""
        try:
            info = self._client.get_collection(col)
            return bool(info.config.params.sparse_vectors)
        except Exception:
            return False

    # ── Index ─────────────────────────────────────────────────────────

    def index_chunks(
        self,
        chunk_ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        sparse_vectors: list[dict[int, float]] | None = None,
    ) -> None:
        """
        Lưu batch vectors vào Qdrant.

        Parameters
        ----------
        chunk_ids      : list[str]               — UUID string mỗi chunk.
        vectors        : list[list[float]]        — dense embedding.
        payloads       : list[dict]               — metadata mỗi chunk.
        sparse_vectors : list[dict[int,float]]   — lexical sparse (khi hybrid).
        """
        if not chunk_ids:
            return

        if self._hybrid and sparse_vectors:
            points = [
                qmodels.PointStruct(
                    id=cid,
                    vector={
                        _DENSE_NAME: vec,
                        _SPARSE_NAME: qmodels.SparseVector(
                            indices=list(sv.keys()),
                            values=list(sv.values()),
                        ),
                    },
                    payload=meta,
                )
                for cid, vec, meta, sv in zip(chunk_ids, vectors, payloads, sparse_vectors)
            ]
        elif self._hybrid:
            # Hybrid collection nhưng không có sparse → dùng dense với named vector
            logger.warning("Hybrid collection nhưng sparse_vectors=None — chỉ index dense.")
            points = [
                qmodels.PointStruct(
                    id=cid,
                    vector={_DENSE_NAME: vec},
                    payload=meta,
                )
                for cid, vec, meta in zip(chunk_ids, vectors, payloads)
            ]
        else:
            # Dense-only: unnamed default vector (tương thích cũ)
            points = [
                qmodels.PointStruct(id=cid, vector=vec, payload=meta)
                for cid, vec, meta in zip(chunk_ids, vectors, payloads)
            ]

        self._client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
            wait=True,
        )
        logger.info(f"Indexed {len(points)} điểm vào Qdrant (hybrid={self._hybrid}).")

    # ── Search ────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        document_ids: list[uuid.UUID] | None = None,
        user_id: uuid.UUID | None = None,
    ) -> list[dict[str, Any]]:
        """
        Dense-only search (interface cũ — vẫn giữ để không break code khác).
        Khi hybrid enabled, dùng named vector 'dense'.
        """
        search_filter = self._build_filter(document_ids, user_id)

        if self._hybrid:
            # Named vector search
            results = self._client.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=(  _DENSE_NAME, query_vector),
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False,
            )
        else:
            results = self._client.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False,
            )

        return [
            {"id": str(hit.id), "score": hit.score, "payload": hit.payload}
            for hit in results
        ]

    def hybrid_search(
        self,
        query_vector: list[float],
        query_sparse: dict[int, float],
        top_k: int = 20,
        document_ids: list[uuid.UUID] | None = None,
        user_id: uuid.UUID | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search: kết hợp dense + sparse bằng RRF (Reciprocal Rank Fusion).

        Chỉ hoạt động khi HYBRID_SEARCH_ENABLED=True và collection có sparse.
        Nếu không → fallback về dense-only search.

        Tại sao RRF tốt hơn weighted sum:
        - Dense tốt với câu hỏi ngữ nghĩa ("điều kiện mua nhà")
        - Sparse tốt với thuật ngữ pháp luật chính xác ("Điều 8", "hộ gia đình")
        - RRF kết hợp rank từ cả 2 nguồn, robust hơn với tham số α
        """
        if not self._hybrid:
            logger.debug("Hybrid search gọi nhưng hybrid disabled → fallback dense.")
            return self.search(query_vector, top_k, document_ids, user_id)

        search_filter = self._build_filter(document_ids, user_id)
        sparse_vec = qmodels.SparseVector(
            indices=list(query_sparse.keys()),
            values=list(query_sparse.values()),
        )

        # Lấy nhiều hơn top_k ở mỗi nhánh để RRF có đủ ứng viên
        prefetch_k = min(top_k * 2, 60)

        prefetch = [
            qmodels.Prefetch(
                query=query_vector,
                using=_DENSE_NAME,
                limit=prefetch_k,
                filter=search_filter,
            ),
            qmodels.Prefetch(
                query=sparse_vec,
                using=_SPARSE_NAME,
                limit=prefetch_k,
                filter=search_filter,
            ),
        ]

        results = self._client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            prefetch=prefetch,
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {"id": str(hit.id), "score": hit.score, "payload": hit.payload}
            for hit in results.points
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
