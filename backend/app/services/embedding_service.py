"""
services/embedding_service.py — Tạo vector embedding bằng BGE-M3.

BGE-M3 hỗ trợ đa ngôn ngữ (tiếng Việt + tiếng Anh), output 1024 chiều.
Dùng FlagEmbedding để có thể dùng dense + sparse + colbert nếu cần.
"""
from __future__ import annotations

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from loguru import logger

from app.config import get_settings

settings = get_settings()


class EmbeddingService:
    """Singleton wrapper quanh BGE-M3 model."""

    def __init__(self) -> None:
        self._model: BGEM3FlagModel | None = None

    def _load(self) -> BGEM3FlagModel:
        if self._model is None:
            logger.info(f"Đang tải embedding model: {settings.EMBEDDING_MODEL}")
            self._model = BGEM3FlagModel(
                settings.EMBEDDING_MODEL,
                use_fp16=True,        # tiết kiệm VRAM/RAM
                device=settings.EMBEDDING_DEVICE,
            )
            logger.info("Embedding model đã sẵn sàng.")
        return self._model

    # ── Public ────────────────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Tạo embedding cho danh sách text.

        Parameters
        ----------
        texts : list[str]
            Danh sách đoạn văn bản (chunk hoặc query).

        Returns
        -------
        list[list[float]]
            Danh sách vector, mỗi vector có shape (1024,).
        """
        if not texts:
            return []

        model = self._load()

        # Chia thành batch để tránh OOM
        batch_size = settings.EMBEDDING_BATCH_SIZE
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = model.encode(
                batch,
                batch_size=len(batch),
                max_length=8192,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            dense: np.ndarray = result["dense_vecs"]  # shape (N, 1024)
            all_embeddings.extend(dense.tolist())
            logger.debug(f"  Embedded batch {i // batch_size + 1}: {len(batch)} texts.")

        logger.info(f"Đã tạo {len(all_embeddings)} embedding vectors.")
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Tạo embedding cho một câu hỏi đơn."""
        vectors = self.embed_texts([query])
        if not vectors:
            raise ValueError("Không thể tạo embedding cho query.")
        return vectors[0]

    @staticmethod
    def validate_vector_size(vector: list[float]) -> bool:
        expected = settings.VECTOR_SIZE  # 1024
        if len(vector) != expected:
            logger.error(f"Vector size mismatch: got {len(vector)}, expected {expected}.")
            return False
        return True


# Singleton
embedding_service = EmbeddingService()
