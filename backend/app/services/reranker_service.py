"""
services/reranker_service.py — Rerank kết quả retrieval bằng bge-reranker-v2-m3.

Luồng:
  1. Nhận query + danh sách candidate chunks từ Qdrant.
  2. Tính relevance score giữa query ↔ mỗi chunk.
  3. Sắp xếp lại theo score giảm dần.
  4. Trả về top-K chunk sau rerank.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from app.config import get_settings

settings = get_settings()


@dataclass
class RankedChunk:
    """Chunk sau khi rerank, kèm score mới."""

    original_id: str           # Qdrant point ID
    content: str
    rerank_score: float
    payload: dict[str, Any]


class RerankerService:
    def __init__(self) -> None:
        self._model = None   # lazy init

    def _load(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker
            logger.info(f"Đang tải reranker model: {settings.RERANKER_MODEL}")
            self._model = FlagReranker(
                settings.RERANKER_MODEL,
                use_fp16=True,
            )
            logger.info("Reranker model đã sẵn sàng.")
        return self._model

    # ── Public ────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[RankedChunk]:
        """
        Rerank danh sách candidate chunks.

        Parameters
        ----------
        query : str
            Câu hỏi của người dùng.
        candidates : list[dict]
            Output từ QdrantService.search() — mỗi dict có: id, score, payload.
            payload phải chứa trường "content" (text của chunk).
        top_k : int, optional
            Số chunk giữ lại sau rerank. Mặc định dùng settings.RERANKER_TOP_K.

        Returns
        -------
        list[RankedChunk] — đã sort theo rerank_score giảm dần.
        """
        if not candidates:
            return []

        top_k = top_k or settings.RERANKER_TOP_K
        model = self._load()

        # Chuẩn bị pairs [query, chunk_content]
        pairs = [
            [query, c["payload"].get("content", "")]
            for c in candidates
        ]

        # Tính scores — trả về list[float]
        scores: list[float] = model.compute_score(pairs, normalize=True)

        # Kết hợp và sắp xếp
        ranked: list[RankedChunk] = []
        for candidate, score in zip(candidates, scores):
            ranked.append(
                RankedChunk(
                    original_id=candidate["id"],
                    content=candidate["payload"].get("content", ""),
                    rerank_score=float(score),
                    payload=candidate["payload"],
                )
            )

        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        top = ranked[:top_k]

        logger.info(
            f"Rerank: {len(candidates)} → {len(top)} chunk "
            f"(top score={top[0].rerank_score:.4f} nếu có)"
            if top else "Rerank: không có kết quả."
        )

        return top

    def filter_by_threshold(
        self,
        ranked: list[RankedChunk],
        threshold: float = 0.3,
    ) -> list[RankedChunk]:
        """
        Loại bỏ các chunk có rerank_score thấp hơn threshold.
        Giúp tránh đưa ngữ cảnh không liên quan vào LLM.
        """
        filtered = [c for c in ranked if c.rerank_score >= threshold]
        if len(filtered) < len(ranked):
            logger.debug(
                f"Reranker filter: {len(ranked)} → {len(filtered)} chunk "
                f"(threshold={threshold})."
            )
        return filtered


# Singleton
reranker_service = RerankerService()
