"""
services/llm_service.py — Gọi Qwen3.5:0.8b qua Ollama để sinh câu trả lời RAG.

Luồng:
  1. Nhận query + danh sách chunk đã rerank.
  2. Xây dựng prompt RAG (context + instruction).
  3. Gọi Ollama HTTP API (streaming hoặc non-streaming).
  4. Trả về câu trả lời + metadata nguồn để frontend hiển thị citation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx
from loguru import logger

from app.config import get_settings
from app.services.reranker_service import RankedChunk

settings = get_settings()


@dataclass
class LLMAnswer:
    answer: str
    model: str
    sources_used: list[dict] = field(default_factory=list)   # chunk metadata dùng làm context


# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp tài liệu thông minh.
Nhiệm vụ của bạn là trả lời câu hỏi **CHỈ DỰA TRÊN** các đoạn tài liệu được cung cấp dưới đây.

Quy tắc:
1. Nếu câu trả lời có trong tài liệu → trả lời chính xác, ngắn gọn, rõ ràng.
2. Nếu không tìm thấy thông tin → trả lời: "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
3. KHÔNG suy đoán hoặc thêm thông tin ngoài tài liệu.
4. KHÔNG bịa đặt nguồn.
5. Trả lời bằng cùng ngôn ngữ với câu hỏi.
"""


def _build_context_block(chunks: list[RankedChunk]) -> str:
    """Định dạng các chunk thành context block cho prompt."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        doc_name = chunk.payload.get("original_filename", "Tài liệu không tên")
        page = chunk.payload.get("page_number")
        page_info = f" — Trang {page}" if page else ""
        lines.append(f"[Nguồn {i}] {doc_name}{page_info}\n{chunk.content}")
    return "\n\n".join(lines)


def _build_user_message(question: str, context: str) -> str:
    return (
        f"Dưới đây là các đoạn tài liệu liên quan:\n\n"
        f"{context}\n\n"
        f"---\n"
        f"Câu hỏi: {question}\n\n"
        f"Hãy trả lời câu hỏi trên dựa vào tài liệu đã cung cấp."
    )


# ── Service ───────────────────────────────────────────────────────────────────

class LLMService:
    def __init__(self) -> None:
        self._base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self._model = settings.OLLAMA_MODEL
        self._generate_url = f"{self._base_url}/api/chat"

    # ── Public ────────────────────────────────────────────────────────

    async def answer(
        self,
        question: str,
        chunks: list[RankedChunk],
    ) -> LLMAnswer:
        """
        Sinh câu trả lời đầy đủ (non-streaming).

        Parameters
        ----------
        question : str
        chunks : list[RankedChunk]
            Các chunk đã rerank — dùng làm context.

        Returns
        -------
        LLMAnswer
        """
        if not chunks:
            return LLMAnswer(
                answer="Tôi không tìm thấy thông tin liên quan trong tài liệu.",
                model=self._model,
            )

        context = _build_context_block(chunks)
        user_msg = _build_user_message(question, context)

        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "options": {
                "temperature": 0.1,   # thấp → ít sáng tạo, bám sát tài liệu hơn
                "top_p": 0.9,
                "num_predict": 1024,
            },
        }

        logger.debug(f"Gọi Ollama model={self._model}, context_chunks={len(chunks)}")

        async with httpx.AsyncClient(timeout=120) as client:
            try:
                resp = await client.post(self._generate_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.error(f"Ollama HTTP error: {exc.response.status_code} — {exc.response.text}")
                raise
            except httpx.ConnectError:
                logger.error(f"Không kết nối được Ollama tại {self._base_url}")
                raise

        answer_text: str = data["message"]["content"]

        sources_used = [
            {
                "document_id": c.payload.get("document_id"),
                "document_name": c.payload.get("original_filename"),
                "chunk_index": c.payload.get("chunk_index"),
                "page_number": c.payload.get("page_number"),
                "content_snippet": c.content[:200],
                "relevance_score": round(c.rerank_score, 4),
            }
            for c in chunks
        ]

        logger.info(f"LLM trả lời: {len(answer_text)} ký tự, dùng {len(chunks)} nguồn.")
        return LLMAnswer(answer=answer_text, model=self._model, sources_used=sources_used)

    async def answer_stream(
        self,
        question: str,
        chunks: list[RankedChunk],
    ) -> AsyncIterator[str]:
        """
        Sinh câu trả lời dạng streaming (Server-Sent Events).
        Yield từng token/đoạn nhỏ để frontend hiển thị realtime.
        """
        if not chunks:
            yield "Tôi không tìm thấy thông tin liên quan trong tài liệu."
            return

        context = _build_context_block(chunks)
        user_msg = _build_user_message(question, context)

        payload = {
            "model": self._model,
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 1024},
        }

        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream("POST", self._generate_url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    async def health_check(self) -> bool:
        """Kiểm tra Ollama có đang chạy và model có sẵn không."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                available = any(self._model in m for m in models)
                if not available:
                    logger.warning(
                        f"Model '{self._model}' chưa được pull. "
                        f"Chạy: ollama pull {self._model}"
                    )
                return available
        except Exception as exc:
            logger.error(f"Ollama health check thất bại: {exc}")
            return False


# Singleton
llm_service = LLMService()
