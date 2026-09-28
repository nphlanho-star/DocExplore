"""
api/chat.py — Chat pipeline: query → retrieval → rerank → LLM → citation.
"""
import json
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config import get_settings
from app.database import get_db
from app.models.chat import ChatMessage, ChatSession
from app.models.permission import Permission
from app.models.document import Document
from app.models.user import User
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatSessionCreate,
    ChatSessionRead,
    CitationSource,
    QueryRequest,
    QueryResponse,
)
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.services.qdrant_service import qdrant_service
from app.services.reranker_service import reranker_service

settings = get_settings()
router = APIRouter(prefix="/api/chat", tags=["Chat"])


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_accessible_doc_ids(user: User, db: AsyncSession) -> list[uuid.UUID]:
    """Trả về danh sách document_id mà user được phép truy cập."""
    # Tài liệu do user sở hữu
    owned = await db.execute(select(Document.id).where(Document.owner_id == user.id))
    owned_ids = set(owned.scalars().all())

    # Tài liệu được chia sẻ
    shared = await db.execute(select(Permission.document_id).where(Permission.user_id == user.id))
    shared_ids = set(shared.scalars().all())

    return list(owned_ids | shared_ids)


async def _get_or_create_session(
    session_id: uuid.UUID | None,
    user: User,
    db: AsyncSession,
) -> ChatSession:
    if session_id:
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == user.id,
            )
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Phiên chat không tồn tại hoặc không có quyền truy cập.",
            )
        return session

    # Tạo session mới
    session = ChatSession(user_id=user.id, title=None)
    db.add(session)
    await db.flush()
    return session


def _sources_from_ranked(ranked: list) -> list[dict]:
    """Chuẩn hóa danh sách chunk đã rerank thành metadata nguồn (dict, JSON-safe)."""
    return [
        {
            "document_id": str(c.payload.get("document_id")),
            "document_name": c.payload.get("original_filename") or "",
            "chunk_index": c.payload.get("chunk_index") or 0,
            "page_number": c.payload.get("page_number"),
            "content_snippet": c.content[:200],
            "relevance_score": round(c.rerank_score, 4),
        }
        for c in ranked
    ]


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Luồng hỏi đáp đầy đủ:
      1. Nhận câu hỏi từ user.
      2. Tạo query embedding (BGE-M3).
      3. Tìm kiếm Qdrant với filter quyền truy cập.
      4. Rerank bằng bge-reranker-v2-m3.
      5. Gửi context cho Qwen qua Ollama.
      6. Lưu lịch sử chat + citation.
      7. Trả về câu trả lời + nguồn.
    """
    # ── Lấy danh sách document được phép ─────────────────────────────
    if body.document_ids:
        # Kiểm tra user có quyền với từng document_id được chỉ định không
        accessible = await _get_accessible_doc_ids(current_user, db)
        accessible_set = set(accessible)
        forbidden = [d for d in body.document_ids if d not in accessible_set]
        if forbidden:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Không có quyền truy cập {len(forbidden)} tài liệu.",
            )
        search_doc_ids = body.document_ids
    else:
        search_doc_ids = await _get_accessible_doc_ids(current_user, db)

    if not search_doc_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bạn chưa có tài liệu nào. Hãy upload tài liệu trước.",
        )

    # ── Embedding câu hỏi ─────────────────────────────────────────────
    query_vector = embedding_service.embed_query(body.question)

    # ── Retrieval từ Qdrant ───────────────────────────────────────────
    candidates = qdrant_service.search(
        query_vector=query_vector,
        top_k=settings.RETRIEVAL_TOP_K,
        document_ids=search_doc_ids,
        user_id=current_user.id,
    )

    # ── Reranking ─────────────────────────────────────────────────────
    ranked = reranker_service.rerank(
        query=body.question,
        candidates=candidates,
        top_k=body.top_k,
    )
    ranked = reranker_service.filter_by_threshold(ranked, threshold=0.3)

    # ── Sinh câu trả lời từ LLM ───────────────────────────────────────
    llm_answer = await llm_service.answer(body.question, ranked)

    # ── Lấy/tạo session và lưu lịch sử ──────────────────────────────
    chat_session = await _get_or_create_session(body.session_id, current_user, db)

    # Lưu tin nhắn user
    user_msg = ChatMessage(
        session_id=chat_session.id,
        role="user",
        content=body.question,
    )
    db.add(user_msg)

    # Chuẩn bị sources
    sources = [
        CitationSource(
            document_id=uuid.UUID(s["document_id"]),
            document_name=s["document_name"] or "",
            chunk_index=s["chunk_index"] or 0,
            page_number=s.get("page_number"),
            content_snippet=s["content_snippet"],
            relevance_score=s["relevance_score"],
        )
        for s in llm_answer.sources_used
    ]

    # Lưu tin nhắn assistant
    assistant_msg = ChatMessage(
        session_id=chat_session.id,
        role="assistant",
        content=llm_answer.answer,
        sources=[s.model_dump(mode="json") for s in sources],
        retrieval_metadata={
            "candidates_count": len(candidates),
            "ranked_count": len(ranked),
            "model": llm_answer.model,
        },
    )
    db.add(assistant_msg)
    await db.flush()

    # Cập nhật tiêu đề session nếu chưa có
    if not chat_session.title:
        chat_session.title = body.question[:80]

    return QueryResponse(
        session_id=chat_session.id,
        message_id=assistant_msg.id,
        answer=llm_answer.answer,
        sources=sources,
        model_used=llm_answer.model,
    )


@router.post("/query/stream")
async def query_stream(
    body: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Streaming version — trả lời qua Server-Sent Events, kèm các sự kiện trạng thái
    ("status") để frontend hiển thị đang xử lý bước nào, sự kiện "sources" (nguồn
    trích dẫn), nhiều sự kiện "token" (từng đoạn câu trả lời), và "done" khi xong.
    Mỗi dòng SSE là 1 JSON: {"type": "...", ...}
    """

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"

    async def event_stream() -> AsyncIterator[str]:
        try:
            # ── Lấy danh sách document được phép ──────────────────────
            if body.document_ids:
                accessible = await _get_accessible_doc_ids(current_user, db)
                accessible_set = set(accessible)
                forbidden = [d for d in body.document_ids if d not in accessible_set]
                if forbidden:
                    yield _sse({"type": "error", "message": f"Không có quyền truy cập {len(forbidden)} tài liệu."})
                    return
                search_doc_ids = body.document_ids
            else:
                search_doc_ids = await _get_accessible_doc_ids(current_user, db)

            if not search_doc_ids:
                yield _sse({"type": "error", "message": "Bạn chưa có tài liệu nào. Hãy upload tài liệu trước."})
                return

            # ── Retrieval ──────────────────────────────────────────────
            yield _sse({"type": "status", "stage": "embed", "message": "Đang phân tích câu hỏi…"})
            query_vector = embedding_service.embed_query(body.question)

            yield _sse({"type": "status", "stage": "retrieval", "message": "Đang tìm kiếm đoạn tài liệu liên quan…"})
            candidates = qdrant_service.search(
                query_vector=query_vector,
                top_k=settings.RETRIEVAL_TOP_K,
                document_ids=search_doc_ids,
                user_id=current_user.id,
            )

            yield _sse({
                "type": "status",
                "stage": "rerank",
                "message": f"Đang xếp hạng {len(candidates)} đoạn tìm được…",
            })
            ranked = reranker_service.rerank(body.question, candidates, top_k=body.top_k)
            ranked = reranker_service.filter_by_threshold(ranked, threshold=0.3)

            sources = _sources_from_ranked(ranked)
            yield _sse({"type": "sources", "sources": sources})

            doc_names = ", ".join(sorted({s["document_name"] for s in sources if s["document_name"]})) or "tài liệu"
            yield _sse({
                "type": "status",
                "stage": "generate",
                "message": f"Đang đọc {doc_names} và soạn câu trả lời…",
            })

            # ── Sinh câu trả lời (streaming) ──────────────────────────
            full_answer = ""
            async for token in llm_service.answer_stream(body.question, ranked):
                full_answer += token
                yield _sse({"type": "token", "content": token})

            if not full_answer:
                full_answer = "Tôi không tìm thấy thông tin liên quan trong tài liệu."

            # ── Lưu lịch sử chat ───────────────────────────────────────
            chat_session = await _get_or_create_session(body.session_id, current_user, db)
            db.add(ChatMessage(session_id=chat_session.id, role="user", content=body.question))
            assistant_msg = ChatMessage(
                session_id=chat_session.id,
                role="assistant",
                content=full_answer,
                sources=sources,
                retrieval_metadata={
                    "candidates_count": len(candidates),
                    "ranked_count": len(ranked),
                    "model": llm_service._model,
                },
            )
            db.add(assistant_msg)
            await db.flush()
            if not chat_session.title:
                chat_session.title = body.question[:80]

            yield _sse({
                "type": "done",
                "session_id": str(chat_session.id),
                "message_id": str(assistant_msg.id),
            })

        except Exception as exc:
            logger.error(f"[chat.query_stream] Lỗi: {exc}")
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Session & History ─────────────────────────────────────────────────────────

@router.get("/sessions", response_model=list[ChatSessionRead])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Lấy danh sách phiên chat của user."""
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
    )
    return result.scalars().all()


@router.post("/sessions", response_model=ChatSessionRead, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: ChatSessionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Tạo phiên chat mới."""
    session = ChatSession(user_id=current_user.id, title=body.title)
    db.add(session)
    await db.flush()
    return session


@router.get("/sessions/{session_id}", response_model=ChatHistoryResponse)
async def get_session_history(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Lấy toàn bộ lịch sử chat của một phiên."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Phiên chat không tồn tại.")

    msgs_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    messages = msgs_result.scalars().all()
    return ChatHistoryResponse(session=session, messages=messages)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Xóa phiên chat."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Phiên chat không tồn tại.")
    await db.delete(session)
