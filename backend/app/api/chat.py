"""
api/chat.py — Chat pipeline: query → retrieval → rerank → LLM → citation.
"""
import asyncio
import difflib
import functools
import json
import re
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy import case, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config import get_settings
from app.database import AsyncSessionLocal, get_db
from app.models.chat import ChatMessage, ChatSession
from app.models.permission import Permission
from app.models.document import Document, DocumentChunk
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
from app.services.injection_guard import REFUSAL_MESSAGE, is_injection_attempt
from app.services.llm_service import llm_service
from app.services.qdrant_service import qdrant_service
from app.services import tavily_service
from app.services.reranker_service import reranker_service, RankedChunk

settings = get_settings()
router = APIRouter(prefix="/api/chat", tags=["Chat"])


async def _run_sync(func, /, *args, **kwargs):
    """
    Chạy 1 hàm ĐỒNG BỘ, chặn CPU/IO nặng (load model, embed, rerank) trong
    threadpool executor thay vì gọi trực tiếp trong async def — nếu không,
    nó sẽ block toàn bộ event loop của FastAPI, khiến MỌI request khác
    (kể cả SSE heartbeat của chính request đang chạy) bị treo cứng trong
    lúc model đang tải (đặc biệt chậm khi Windows đang thiếu RAM/paging),
    dẫn đến "Lỗi kết nối" phía frontend do client tưởng server đã chết.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


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


_GREETING_KEYWORDS = (
    "chào", "hi", "hello", "hey", "alo",
    "cảm ơn", "cám ơn", "thanks", "thank you",
    "tạm biệt", "bye", "khỏe không", "khoẻ không",
)


def _is_greeting(text: str) -> bool:
    """Nhận diện câu chào hỏi/xã giao ngắn — không cần đi qua retrieval/RAG."""
    t = text.strip().lower()
    if not t or len(t) > 30:
        return False
    return any(kw in t for kw in _GREETING_KEYWORDS)


_MAX_SOURCES_SHOWN = 10


def _sources_from_ranked(ranked: list) -> list[dict]:
    """Chuan hoa danh sach chunk da rerank thanh metadata nguon (dict, JSON-safe).

    Chi lay toi da _MAX_SOURCES_SHOWN chunk lien quan nhat (da sort theo
    rerank_score giam dan tu truoc) de hien thi duoi cau tra loi, kem full
    "content" de frontend co the mo xem toan bo khi nguoi dung click vao.
    Voi van ban phap luat, bo sung article_number / article_title / chapter /
    van_ban_type de frontend hien thi citation cu the ("Dieu 8 — ...").
    """
    return [
        {
            "document_id": str(c.payload.get("document_id")),
            "document_name": c.payload.get("original_filename") or "",
            "chunk_index": c.payload.get("chunk_index") or 0,
            "page_number": c.payload.get("page_number"),
            # ── Metadata pháp luật (None nếu không phải văn bản luật) ──
            "article_number": c.payload.get("article_number"),
            "article_title": c.payload.get("article_title"),
            "chapter": c.payload.get("chapter"),
            "van_ban_type": c.payload.get("van_ban_type"),
            "van_ban_year": c.payload.get("van_ban_year"),
            # ── Nội dung ──────────────────────────────────────────────
            "content_snippet": c.content[:200],
            "content": c.content,
            "relevance_score": round(c.rerank_score, 4),
        }
        for c in ranked[:_MAX_SOURCES_SHOWN]
    ]


async def _retrieve_and_rerank(
    question: str,
    embed_text: str,
    search_doc_ids: list[uuid.UUID],
    explicit_multi_doc: bool,
    top_k: int | None,
    user_id: uuid.UUID,
) -> tuple[list, list]:
    """
    1 lượt retrieval + rerank hoàn chỉnh (dùng chung cho lượt tìm kiếm bình
    thường VÀ lượt thử lại bằng HyDE).

    Khi HYBRID_SEARCH_ENABLED=True: dùng dense + sparse (BGE-M3 lexical) kết
    hợp bằng RRF trong Qdrant → cải thiện recall cho thuật ngữ pháp luật
    chính xác ("Điều 8", "hộ gia đình"…) mà dense embedding bắt kém hơn.
    Khi False: chỉ dùng dense (tương thích ngược hoàn toàn).
    """
    _t0 = time.perf_counter()
    if settings.HYBRID_SEARCH_ENABLED:
        query_vector, query_sparse = await _run_sync(
            embedding_service.embed_query_with_sparse, embed_text
        )
    else:
        query_vector = await _run_sync(embedding_service.embed_query, embed_text)
        query_sparse = None

    def _search(doc_ids):
        """Wrapper chọn dense hoặc hybrid search tuỳ cấu hình."""
        if settings.HYBRID_SEARCH_ENABLED and query_sparse:
            return qdrant_service.hybrid_search(
                query_vector=query_vector,
                query_sparse=query_sparse,
                top_k=settings.RETRIEVAL_TOP_K,
                document_ids=doc_ids,
                user_id=user_id,
            )
        return qdrant_service.search(
            query_vector=query_vector,
            top_k=settings.RETRIEVAL_TOP_K,
            document_ids=doc_ids,
            user_id=user_id,
        )

    if explicit_multi_doc:
        candidates: list = []
        ranked: list = []
        per_doc_top_k = max(2, (top_k or settings.RERANKER_TOP_K) // len(search_doc_ids) + 1)
        for doc_id in search_doc_ids:
            doc_candidates = await _run_sync(_search, [doc_id])
            candidates.extend(doc_candidates)
            doc_ranked = await _run_sync(reranker_service.rerank, question, doc_candidates, top_k=per_doc_top_k)
            ranked.extend(doc_ranked)
        ranked.sort(key=lambda c: c.rerank_score, reverse=True)
    else:
        _t1 = time.perf_counter()
        candidates = await _run_sync(_search, search_doc_ids)
        _t2 = time.perf_counter()
        ranked = await _run_sync(reranker_service.rerank, question, candidates, top_k=top_k)
        _t3 = time.perf_counter()
        logger.info(
            f"[timing] hybrid={settings.HYBRID_SEARCH_ENABLED} embed={_t1 - _t0:.1f}s "
            f"qdrant={_t2 - _t1:.2f}s rerank({len(candidates)} chunk)={_t3 - _t2:.1f}s"
        )

    return candidates, ranked


async def _detect_named_document(
    question: str,
    doc_ids: list[uuid.UUID],
    db: AsyncSession,
) -> uuid.UUID | None:
    """
    Case #08: nếu câu hỏi nhắc rõ tên/mã số của đúng 1 tài liệu trong danh sách
    được phép tìm kiếm → tự động giới hạn phạm vi tìm kiếm vào tài liệu đó,
    tránh bị lẫn kết quả từ tài liệu khác (VD: hỏi về "hợp đồng 2024" nhưng
    lại bị trả lời bằng nội dung "hợp đồng 2022").
    """
    if len(doc_ids) <= 1:
        return None

    result = await db.execute(
        select(Document.id, Document.original_filename).where(Document.id.in_(doc_ids))
    )
    rows = result.all()

    q_lower = question.lower()
    matches: set[uuid.UUID] = set()
    for doc_id, filename in rows:
        stem = Path(filename or "").stem.lower()
        # Tên file ngắn như "P1", "P2" cũng phải nhận ra: token ≥4 ký tự khớp theo chuỗi con,
        # token ngắn hơn (≥2) phải khớp NGUYÊN TỪ (tránh "p1" khớp nhầm trong "p10").
        # Nhắc nguyên tên file (kể cả có khoảng trắng/ký tự đặc biệt) cũng khớp.
        tokens = [t for t in re.split(r"[\s_\-.]+", stem) if len(t) >= 2]
        hit = bool(stem) and re.search(r"(?<!\w)" + re.escape(stem) + r"(?!\w)", q_lower) is not None
        if not hit:
            for t in tokens:
                if len(t) >= 4:
                    if t in q_lower:
                        hit = True
                        break
                elif re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", q_lower):
                    hit = True
                    break
        if hit:
            matches.add(doc_id)

    if len(matches) == 1:
        return next(iter(matches))
    return None


_MAX_CONTEXT_CHARS = 6000
# Ước tính 1 token ≈ 3 ký tự tiếng Việt (có dấu, nhiều byte).
# Qwen 0.8B context window ≈ 8192 token; chừa ~2000 token cho system prompt +
# câu trả lời → budget tối đa cho context ≈ 6000 token ≈ 18 000 ký tự.
# Nhưng để an toàn, vẫn cắt tại 6000 ký tự/chunk tổng thể (≈ 2000 token) để
# tránh attention dilution khi có nhiều chunk cùng lúc.
_MAX_SINGLE_CHUNK_CHARS = 2000   # 1 chunk đơn lẻ không được vượt quá mức này


def _truncate_chunk_content(content: str, max_chars: int = _MAX_SINGLE_CHUNK_CHARS) -> str:
    """
    Case #06: nếu 1 chunk riêng lẻ quá dài (VD: Điều có 30 khoản), cắt tại
    ranh giới câu/khoản thay vì cắt giữa chừng — giữ nội dung còn đọc được.
    """
    if len(content) <= max_chars:
        return content
    cut = content[:max_chars]
    for sep in ("\n", ". ", "; ", ", "):
        pos = cut.rfind(sep)
        if pos > max_chars // 2:
            return cut[:pos + len(sep)].rstrip() + "\n[…đã rút gọn do vượt giới hạn context]"
    return cut.rstrip() + "\n[…đã rút gọn]"


_HEADER_RE = re.compile(r"^(\[[^\]]{1,120}\]|Điều\s+\d+)")
_UNIT_SPLIT = re.compile(r"\n+|(?<=[.;:])\s+(?=\S)")


def _with_content(c, content: str):
    import dataclasses
    try:
        return dataclasses.replace(c, content=content)
    except TypeError:
        return type("_TC", (), {**c.__dict__, "content": content})()


def _compress_sync(chunks: list, question: str, max_total: int, max_single: int) -> list:
    """
    Nén context THEO CÂU (extractive): chỉ khi tổng context vượt ngân sách hoặc 1 chunk quá dài.
    - Ngân sách từng chunk chia theo điểm rerank (chunk liên quan hơn được giữ nhiều hơn).
    - Với chunk bị nén: chấm điểm từng câu với câu hỏi bằng reranker, giữ các câu điểm cao NGUYÊN VĂN
      (luôn giữ dòng tiêu đề "[Chương…] Điều N — …"), giữ nguyên thứ tự gốc, chỗ bị bỏ ghi "[…]".
    Không sinh thêm chữ nào → không thể bịa; lỗi bất kỳ → trả chunk gốc để cách cắt cũ xử lý.
    """
    if not chunks:
        return chunks
    lens = [len(c.content) for c in chunks]
    if sum(lens) <= max_total and max(lens) <= max_single:
        return chunks

    weights = [max(float(getattr(c, "rerank_score", 0.0) or 0.0), 0.05) for c in chunks]
    budgets = [None] * len(chunks)
    remaining, active = max_total, set(range(len(chunks)))
    while active:                       # chunk ngắn hơn phần được chia thì giữ nguyên, trả phần dư cho chunk khác
        wsum = sum(weights[i] for i in active)
        shrink = [i for i in active if lens[i] <= min(max_single, remaining * weights[i] / wsum)]
        if not shrink:
            for i in active:
                budgets[i] = int(max(500, min(max_single, remaining * weights[i] / wsum)))
            break
        for i in shrink:
            budgets[i] = lens[i]
            remaining -= lens[i]
            active.discard(i)

    # Tách câu cho các chunk cần nén, chấm điểm 1 lượt duy nhất
    todo = [i for i in range(len(chunks)) if lens[i] > budgets[i]]
    units: dict[int, list[str]] = {}
    flat: list[tuple[int, int]] = []
    for i in todo:
        us = [u.strip() for u in _UNIT_SPLIT.split(chunks[i].content) if u and u.strip()]
        units[i] = us
        flat += [(i, k) for k, u in enumerate(us) if not (k == 0 and _HEADER_RE.match(u))]
    if len(flat) > 120:                 # chặn chi phí rerank trên CPU
        flat = flat[:120]
    scores = reranker_service.score_texts(question, [units[i][k] for i, k in flat])
    score_of = {key: sc for key, sc in zip(flat, scores)}

    out = list(chunks)
    for i in todo:
        us, budget = units[i], budgets[i]
        keep, used = set(), 0
        if us and _HEADER_RE.match(us[0]):
            keep.add(0)
            used += len(us[0]) + 1
        for k in sorted((k for (ii, k) in score_of if ii == i), key=lambda k: score_of[(i, k)], reverse=True):
            if used + len(us[k]) + 1 > budget:
                continue
            keep.add(k)
            used += len(us[k]) + 1
        if len(keep) <= (1 if us and 0 in keep else 0):   # không chọn được câu nào → để cách cắt cũ
            continue
        parts, prev = [], -1
        for k in sorted(keep):
            if prev != -1 and k != prev + 1:
                parts.append("[…]")
            parts.append(us[k])
            prev = k
        if prev != len(us) - 1:
            parts.append("[…]")
        new_content = "\n".join(parts)
        logger.info(f"[compress] chunk {i}: {lens[i]} → {len(new_content)} ký tự ({len(keep)}/{len(us)} câu)")
        out[i] = _with_content(chunks[i], new_content)
    return out


async def _compress_context(chunks: list, question: str) -> tuple[list, dict]:
    """Nén theo câu (nếu bật) rồi áp ngân sách cứng như cũ làm lớp an toàn cuối.
    Trả về (chunks, stats) — stats dùng cho biểu tượng mức context trên giao diện."""
    orig_chars = sum(len(c.content) for c in chunks)
    if settings.CONTEXT_COMPRESS_ENABLED and chunks:
        try:
            chunks = await _run_sync(_compress_sync, chunks, question, _MAX_CONTEXT_CHARS, _MAX_SINGLE_CHUNK_CHARS)
        except Exception as exc:
            logger.warning(f"[compress] bỏ qua nén (lỗi: {exc}) → dùng cách cắt cũ")
    final = _apply_context_budget(chunks)
    used = sum(len(c.content) for c in final)
    stats = {
        "type": "context",
        "percent": round(orig_chars * 100 / _MAX_CONTEXT_CHARS) if orig_chars else 0,  # >100 = đã vượt ngưỡng nén
        "orig_chars": orig_chars,
        "used_chars": used,
        "budget_chars": _MAX_CONTEXT_CHARS,
        "compressed": used < orig_chars,
    }
    return final, stats


def _apply_context_budget(chunks: list, max_chars: int = _MAX_CONTEXT_CHARS) -> list:
    """
    Case #06: giới hạn tổng độ dài context đưa vào LLM — tránh vượt quá context
    window hoặc làm loãng sự chú ý của model khi ghép quá nhiều chunk lại.

    Nâng cấp so với v1:
    - Truncate từng chunk quá dài TRƯỚC khi đếm tổng (thay vì drop cả chunk).
    - Log warning khi ngân sách bị kích hoạt.
    - Luôn trả về ít nhất 1 chunk (dù chunk đó bị truncate).
    """
    import dataclasses

    result = []
    total = 0
    clipped = 0
    for c in chunks:
        # Truncate từng chunk nếu quá dài
        trimmed_content = _truncate_chunk_content(c.content)
        if trimmed_content != c.content:
            clipped += 1
            # Tạo bản copy với content đã rút gọn (RankedChunk là dataclass)
            try:
                c = dataclasses.replace(c, content=trimmed_content)
            except TypeError:
                # Không phải dataclass → patch trực tiếp (an toàn vì chỉ dùng nội bộ)
                c = type("_TC", (), {**c.__dict__, "content": trimmed_content})()

        length = len(c.content)
        if result and total + length > max_chars:
            logger.warning(
                f"[context_budget] Đã đạt ngân sách {max_chars} ký tự sau {len(result)} chunk "
                f"(bỏ {len(chunks) - len(result)} chunk còn lại). "
                f"Gợi ý: giảm top_k hoặc tăng _MAX_CONTEXT_CHARS."
            )
            break
        result.append(c)
        total += length

    if clipped:
        logger.warning(f"[context_budget] Đã rút gọn {clipped} chunk vượt {_MAX_SINGLE_CHUNK_CHARS} ký tự.")
    return result or chunks[:1]


# ── Case #07: Phát hiện ngôn ngữ câu hỏi ────────────────────────────────────

_VI_DIACRITICS_RE = re.compile(
    r"[àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    r"ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]",
    re.UNICODE,
)
_EN_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")


def _detect_language(text: str) -> str:
    """
    Case #07: phát hiện ngôn ngữ câu hỏi.
    Trả về "vi" | "en" | "mixed".
    Heuristic: đếm ký tự có dấu tiếng Việt vs từ ASCII thuần.
    """
    vi_hits = len(_VI_DIACRITICS_RE.findall(text))
    en_words = len(_EN_WORD_RE.findall(text))
    total_words = len(text.split())

    if vi_hits >= 2:
        return "vi" if en_words < 5 else "mixed"

    # Câu ngắn mà không có dấu tiếng Việt → ưu tiên English
    # VD: "apartment ?", "what is this", "housing law"
    if vi_hits == 0 and en_words >= 1:
        return "en"

    # Câu dài hơn nhưng vẫn đủ từ tiếng Anh
    if en_words >= 2 and en_words >= total_words // 2:
        return "en"

    # Không phân loại được (VD: số, ký hiệu, từ viết tắt…) → mặc định tiếng Việt
    return "vi"


# ── Case #05: Phát hiện câu hỏi mơ hồ ───────────────────────────────────────

_AMBIGUOUS_STOPWORDS = {
    "là", "gì", "và", "của", "có", "không", "được", "cho", "các", "những",
    "này", "đó", "thì", "sao", "với", "trong", "về", "như", "nào", "hãy",
    "tôi", "bạn", "thế", "vậy", "hơn", "còn", "một", "ạ", "ơi", "nhé",
    "nha", "đi", "ư", "ừ", "à", "á", "nhỉ", "hả", "ha", "hah", "the",
    "what", "and", "for", "is", "are", "how", "why", "who",
}
_SPECIFIC_CONTENT_RE = re.compile(
    r"điều\s*\d+|khoản\s*\d+|chương\s*[IVXivx\d]+|"
    r"quyền|nghĩa\s*vụ|điều\s*kiện|thủ\s*tục|hợp\s*đồng|"
    r"phạt|bồi\s*thường|sở\s*hữu|thuê|mua|bán|cấp\s*phép|"
    r"article|section|clause|right|obligation|condition",
    re.IGNORECASE,
)
_AMBIGUOUS_CLARIFY_MSG_VI = (
    "Câu hỏi của bạn chưa đủ rõ để tìm kiếm trong tài liệu. 🤔\n\n"
    "Bạn muốn hỏi về nội dung gì? Ví dụ:\n"
    "- \"Quyền của chủ sở hữu nhà ở là gì?\"\n"
    "- \"Điều kiện để được cấp Giấy chứng nhận?\"\n"
    "- \"Khoản 2 Điều 10 quy định gì?\""
)
_AMBIGUOUS_CLARIFY_MSG_EN = (
    "Your question is too vague to search in the document. 🤔\n\n"
    "What would you like to know? For example:\n"
    "- \"What are the rights of a homeowner?\"\n"
    "- \"What are the conditions for obtaining a Certificate?\"\n"
    "- \"What does Article 10, Clause 2 state?\""
)


def _is_ambiguous_query(question: str, has_history: bool) -> bool:
    """
    Case #05: câu hỏi mơ hồ khi:
    1. Không có lịch sử hội thoại (nếu có lịch sử thì _resolve_followup sẽ mở rộng)
    2. Quá ít từ có nghĩa (< 3 sau khi bỏ stopwords)
    3. Không chứa bất kỳ tín hiệu nội dung cụ thể nào (Điều X, Khoản Y, từ khóa pháp lý)
    """
    if has_history:
        return False   # followup context → _resolve_followup lo
    if _SPECIFIC_CONTENT_RE.search(question):
        return False   # có keyword pháp lý cụ thể → không mơ hồ
    words = [w for w in re.split(r"[\s\W]+", question.lower()) if w and w not in _AMBIGUOUS_STOPWORDS]
    return len(words) < 3


_META_DOC_KEYWORDS = (
    "nói về gì", "nói về cái gì", "về luật gì", "luật gì vậy", "là luật gì",
    "về gì vậy", "nội dung chính", "nội dung gì", "tóm tắt nội dung", "về vấn đề gì",
)
_DOC_REFERENCE_WORDS = ("tài liệu", "văn bản", "bài", "file")


def _is_meta_document_question(question: str) -> bool:
    """
    Câu hỏi khái quát HỎI VỀ CHÍNH TÀI LIỆU (VD: "tài liệu này nói về gì",
    "bài của mình là về luật gì vậy") — bản thân câu hỏi ít tín hiệu ngữ nghĩa
    để similarity-search so khớp, nên dù điểm rerank thấp vẫn nên cho phép
    dùng top chunk để LLM trả lời, KHÁC với 1 câu hỏi về chủ đề pháp lý cụ thể
    nhưng không có trong tài liệu (VD: "luật hôn nhân là gì" hỏi trên tài liệu
    Luật Nhà ở) — trường hợp đó phải bị từ chối, không được LLM tự suy diễn.
    """
    q = question.lower()
    return any(w in q for w in _DOC_REFERENCE_WORDS) and any(k in q for k in _META_DOC_KEYWORDS)


_SUMMARY_KEYWORDS = (
    "tóm tắt", "tom tat", "khái quát", "khai quat",
    "tổng quan", "tong quan", "tổng kết", "tong ket", "summary", "summarize",
    # Cụm từ tự nhiên thêm vào — người dùng hay nói theo kiểu này
    "tóm lại", "tom lai", "tóm gọn", "tom gon",
    "nội dung chính", "noi dung chinh", "ý chính", "y chinh",
    "điểm chính", "diem chinh", "ý chính", "points chính",
    "nói ngắn", "noi ngan", "ngắn gọn lại", "rút gọn",
    "overview", "nêu sơ qua", "sơ lược",
)
_SPECIFIC_REF_PATTERN = re.compile(
    r"điều\s*\d+"           # "Điều 8", "điều8"
    r"|\d+\s*điều"          # "10 điều", "5điều"  ← số trước tên
    r"|chương\s*\d+"        # "Chương II", "chương 2"
    r"|\d+\s*chương"        # "2 chương"
    r"|khoản\s*\d+",        # "Khoản 3"
    re.IGNORECASE,
)


def _is_full_summary_request(question: str) -> bool:
    """
    Phát hiện yêu cầu tóm tắt tài liệu theo Điều (toàn bộ hoặc một phần —
    VD: "tóm tắt file", "tóm tắt 10 điều đầu", "tóm tắt Chương II").
    Kích hoạt pipeline lấy chunk theo Điều từ DB thay vì similarity search.
    Nếu câu hỏi nhắc rõ 1 Điều/Khoản CỤ THỂ bằng số ("Điều 8 quy định gì")
    thì vẫn coi là câu hỏi thường và dùng RAG bình thường.

    Phân biệt "tóm tắt toàn bộ / một phần" sẽ do extract_article_range()
    (LLM) xử lý ở bước sau — hàm này chỉ cần quyết định có kích hoạt
    pipeline tóm tắt theo Điều hay không.
    """
    q = question.lower()
    if not any(k in q for k in _SUMMARY_KEYWORDS):
        return False
    # Chỉ đi RAG bình thường khi nhắc rõ 1 Điều/Khoản duy nhất bằng số cụ thể
    # và KHÔNG kèm từ chỉ số lượng ("10 điều đầu" → vẫn là summary pipeline,
    # nhưng "giải thích Điều 8" → RAG bình thường).
    ref_matches = _SPECIFIC_REF_PATTERN.findall(q)
    # Nếu chỉ có đúng 1 kết quả match và đó là dạng "điều\s*\d+" (không phải
    # "\d+\s*điều") → câu hỏi về 1 điều cụ thể → RAG thường.
    if len(ref_matches) == 1:
        single = ref_matches[0].strip().lower()
        if re.match(r"^điều\s*\d+$", single, re.IGNORECASE):
            return False
    return True


def _apply_article_range(
    chunks: list,
    range_info: dict,
) -> list:
    """
    Lọc/sắp xếp danh sách chunk theo kết quả extract_article_range() của LLM.

    Parameters
    ----------
    chunks     : list[RankedChunk] — đã gom theo Điều, sắp xếp theo thứ tự tài liệu.
    range_info : dict              — output từ llm_service.extract_article_range().

    Returns
    -------
    list[RankedChunk] đã lọc theo range.
    """
    rtype = range_info.get("type", "all")

    if rtype == "all":
        return chunks

    def _num(c):
        m = re.match(r"\d+", str(c.payload.get("article_number") or "").strip())
        return int(m.group()) if m else None

    if rtype in ("first_n", "last_n"):
        n = int(range_info.get("n", len(chunks)))
        # "N điều đầu/cuối" theo SỐ ĐIỀU của từng tài liệu (Điều 1..N), không theo thứ tự chunk
        # (file có thể bắt đầu bằng chương khác, vd Điều 98 nằm trước Điều 1). Các nhóm không có
        # số Điều (tiêu đề luật, "Phần 1/2"…) không phải Điều nên không đưa vào bản tóm tắt.
        by_doc: dict = {}
        for c in chunks:
            if _num(c) is not None:
                by_doc.setdefault(c.payload.get("document_id"), []).append(c)
        keep: list = []
        for lst in by_doc.values():
            nums = sorted({_num(c) for c in lst})
            chosen = set(nums[:n] if rtype == "first_n" else nums[-n:])
            keep.extend(c for c in lst if _num(c) in chosen)
        keep.sort(key=lambda c: (str(c.payload.get("document_id")), _num(c), c.payload.get("chunk_index") or 0))
        return keep

    if rtype == "range":
        from_no = int(range_info.get("from", 1))
        to_no = int(range_info.get("to", 9999))
        return [
            c for c in chunks
            if from_no <= int(c.payload.get("article_number") or 0) <= to_no
        ]

    if rtype == "list":
        article_list = {int(x) for x in range_info.get("list", [])}
        return [
            c for c in chunks
            if int(c.payload.get("article_number") or 0) in article_list
        ]

    if rtype == "chapter":
        chapter_target = str(range_info.get("chapter", "")).strip().lower()
        return [
            c for c in chunks
            if chapter_target in str(c.payload.get("chapter") or "").lower()
        ]

    return chunks


_SENT_SPLIT_RE = re.compile(r"(?<![0-9])(?<=[.!?])\s+(?=[^\d\s])")
_HEADING_PREFIX_RE = re.compile(r"^điều\s*\d+[a-zA-Z]?\s*[.:]?\s*", re.IGNORECASE)


def _clean_article_title(title: str) -> str:
    '''"Điều 1. Phạm vi điều chỉnh" -> "Phạm vi điều chỉnh" (nhãn đã có "Điều 1").'''
    return _HEADING_PREFIX_RE.sub("", " ".join((title or "").split())).strip()


def _strip_heading(content: str, title: str) -> str:
    '''Bỏ phần tiêu đề lặp ở đầu nội dung ("Điều 1. Phạm vi điều chỉnh Luật này…" -> "Luật này…").'''
    text = " ".join(content.split())
    # Chunker gắn header "[Chương VII] Điều 98 — Tên điều" vào ĐẦU MỖI chunk (kể cả các chunk nối tiếp
    # của cùng 1 Điều) → xoá mọi lần xuất hiện của header đó trước khi bỏ tiêu đề lặp.
    _ct = _clean_article_title(title)
    if _ct:
        text = re.sub(
            r"\[[^\]]{1,80}\]\s*Điều\s*\d+[a-zA-Z]?\s*—\s*" + re.escape(_ct) + r"\s*", "", text, flags=re.IGNORECASE
        ).strip()
        text = re.sub(
            r"(?<!\w)Điều\s*\d+[a-zA-Z]?\s*—\s*" + re.escape(_ct) + r"\s*", "", text, flags=re.IGNORECASE
        ).strip()
    t = " ".join((title or "").split())
    if t and text.lower().startswith(t.lower()):
        return text[len(t):].strip()
    return _HEADING_PREFIX_RE.sub("", text, count=1).strip() if not t else text


def _complete_snippet(content: str, max_chars: int, hard_cap: int) -> str:
    '''
    Rút gọn theo CÂU TRỌN VẸN, không bao giờ cắt giữa câu:
      - gom các câu liên tiếp từ đầu chừng nào tổng ≤ max_chars;
      - câu đầu dài hơn max_chars nhưng ≤ hard_cap thì vẫn lấy nguyên câu đó;
      - câu đầu quá dài hoặc kết thúc bằng ":" (dẫn vào 1 danh sách) → trả "" để
        chỉ hiển thị tiêu đề Điều (đủ ý, không dở dang).
    Vì trích nguyên văn nên tuyệt đối không bịa.
    '''
    text = " ".join(content.split())
    if not text:
        return ""
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    picked: list[str] = []
    total = 0
    for s in sentences:
        if s.endswith(":"):
            break
        extra = len(s) + (1 if picked else 0)
        if picked and total + extra > max_chars:
            break
        if not picked and len(s) > hard_cap:
            break
        picked.append(s)
        total += extra
    return " ".join(picked)


_SUBITEM_SPLIT_RE = re.compile(r'(?<!\w)([a-zđ])\)\s+', re.IGNORECASE)


def _cut_at_boundary(text: str, max_chars: int) -> str:
    """Cắt text tại ranh giới câu/mệnh đề gần nhất trong max_chars."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in (". ", "; ", ", "):
        pos = cut.rfind(sep)
        if pos > max_chars // 3:
            return cut[: pos + 1].rstrip()
    pos = cut.rfind(" ")
    return (cut[:pos] if pos > 0 else cut) + "…"


def _koan_snippet(text: str, max_chars: int) -> str:
    """
    Trích nội dung 1 khoản — tóm gọn, không copy nguyên văn toàn bộ.

    Nếu khoản có sub-items a) b) c)...: tách từng sub-item, mỗi cái cắt
    ngắn tại ranh giới câu (~100 ký tự), liệt kê đủ tất cả sub-item trên
    từng dòng — đảm bảo không bỏ sót item nào dù vẫn ngắn gọn.

    Nếu không có sub-items: cắt tại ranh giới câu trong max_chars.
    """
    t = " ".join(text.split())
    if not t:
        return ""

    # Tách sub-items a) b) c)...
    parts = _SUBITEM_SPLIT_RE.split(t)
    # parts = [intro, 'a', content_a, 'b', content_b, ...]
    # parts[0] là phần intro trước a), sau đó cứ 2 phần tử là (nhãn, nội dung)
    if len(parts) >= 3:
        intro = parts[0].strip().rstrip(":")
        lines = [intro] if intro else []
        i = 1
        while i + 1 < len(parts):
            label = parts[i].strip()          # "a", "b", "c"...
            content = parts[i + 1].strip()    # nội dung sub-item
            short = _cut_at_boundary(content, 120)
            lines.append(f"  {label}) {short}")
            i += 2
        return "\n".join(lines)

    # Không có sub-items → cắt tại ranh giới câu
    return _cut_at_boundary(t, max_chars)


def _summary_snippet(text: str, level: str) -> str:
    """
    Trích nội dung tóm tắt cho 1 Điều trong pipeline full_summary.

    Khác `_complete_snippet` ở chỗ:
    - Nhận diện cấu trúc khoản (dòng bắt đầu bằng số như "1.", "2.", "a)", "b)")
      và liệt kê từng khoản ngắn gọn thay vì cắt giữa chừng.
    - Không bao giờ để lại số khoản lơ lửng ("2.") ở cuối.
    - level "brief"  → chỉ lấy câu dẫn đầu + đếm số khoản (VD: "... (3 khoản)")
    - level "normal" → lấy câu dẫn đầu + tóm tắt 1 dòng mỗi khoản, tối đa 3 khoản
    - level "detail" → lấy câu dẫn đầu + tất cả các khoản, mỗi khoản 1 dòng
    """
    import re as _re
    text = " ".join(text.split())
    if not text:
        return ""

    # Tách phần dẫn đầu (trước khoản 1) và các khoản
    # Pattern: dòng bắt đầu bằng "1." hoặc "1 " (sau normalize whitespace)
    # Vì text đã được join thành 1 dòng, tìm pattern " 1. " hoặc bắt đầu "1. "
    koan_pattern = _re.compile(r'(?:^|\s)(\d+)\.\s')
    matches = list(koan_pattern.finditer(text))

    # Không có cấu trúc khoản → dùng logic cũ
    if not matches or matches[0].group(1) != "1":
        return _complete_snippet(text, *{"brief": (170, 170), "normal": (260, 450), "detail": (650, 900)}[level])

    # Có cấu trúc khoản — tách phần dẫn và các khoản
    first_match = matches[0]
    intro = text[:first_match.start()].strip().rstrip(":")

    # Tách nội dung từng khoản
    koan_texts: list[str] = []
    for idx, m in enumerate(matches):
        start = m.start() + len(m.group(0)) - len(m.group(1) + ". ")  # vị trí sau số khoản
        # Điều chỉnh: lấy từ sau "N. "
        content_start = m.end()
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        koan_content = text[content_start:content_end].strip()
        koan_texts.append(koan_content)

    total_koan = len(koan_texts)

    if level == "brief":
        # Chỉ câu dẫn + số khoản
        intro_short = _complete_snippet(intro, 150, 150) if intro else ""
        suffix = f"({total_koan} khoản)" if total_koan > 0 else ""
        return f"{intro_short} {suffix}".strip() if intro_short else suffix

    if level == "normal":
        # Câu dẫn + TẤT CẢ khoản, mỗi khoản 1 dòng riêng — không bỏ khoản nào
        # (văn bản pháp luật: bỏ khoản = mất nội dung quan trọng)
        lines = [intro] if intro else []
        for k_idx, kt in enumerate(koan_texts, start=1):
            lines.append(f"{k_idx}. {_koan_snippet(kt, 250)}")
        return "\n".join(lines)

    # level == "detail" — trích đầy đủ hơn
    lines = [intro] if intro else []
    for k_idx, kt in enumerate(koan_texts, start=1):
        lines.append(f"{k_idx}. {_koan_snippet(kt, 500)}")
    return "\n".join(lines)


def _extractive_snippet(text: str, max_chars: int = 220) -> str:
    """
    Rút gọn 1 đoạn văn bản mà KHÔNG qua LLM — cắt tại ranh giới câu gần nhất
    trong giới hạn max_chars. Vì là trích nguyên văn từ tài liệu nên tuyệt đối
    không có rủi ro bịa đặt, và cực nhanh (không tốn 1 lần gọi model nào).
    """
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    truncated = normalized[:max_chars]
    last_dot = truncated.rfind(". ")
    if last_dot > 40:
        return truncated[: last_dot + 1]
    last_space = truncated.rfind(" ")
    head = truncated[:last_space] if last_space > 40 else truncated
    return head + "…"


async def _group_chunks_by_article(
    doc_ids: list[uuid.UUID],
    db: AsyncSession,
) -> list[RankedChunk]:
    """
    Tóm tắt ĐẦY ĐỦ toàn bộ tài liệu theo từng Điều — dùng chunk_metadata
    (chapter/article_number/article_title) đã trích xuất sẵn lúc chunking để
    GỘP các chunk thuộc CÙNG 1 Điều lại thành 1 nhóm (1 điều có thể trải dài
    qua nhiều chunk), trả về đúng 1 nhóm cho MỖI điều thực sự có trong tài
    liệu — không bỏ sót điều nào, khác hẳn kiểu lấy mẫu rải đều trước đây
    (dễ nhảy cóc bỏ sót hàng chục điều giữa các điểm lấy mẫu).
    """
    result = await db.execute(
        select(DocumentChunk, Document.original_filename)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.document_id.in_(doc_ids))
        .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
    )
    rows = result.all()

    groups: list[RankedChunk] = []
    current_key = None
    current_parts: list[str] = []
    current_meta: dict = {}

    def _flush():
        if current_parts:
            groups.append(
                RankedChunk(
                    original_id=str(current_meta.get("first_chunk_id", "")),
                    content="\n".join(current_parts),
                    rerank_score=1.0,
                    payload=dict(current_meta),
                )
            )

    for chunk, filename in rows:
        meta = chunk.chunk_metadata or {}
        chapter = str(meta.get("chapter") or "")
        article_number = str(meta.get("article_number") or "")
        article_title = str(meta.get("article_title") or "")
        key = (chunk.document_id, article_number, chapter)

        if key != current_key:
            _flush()
            current_key = key
            current_parts = [chunk.content]
            current_meta = {
                "document_id": str(chunk.document_id),
                "original_filename": filename,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "chapter": chapter,
                "article_number": article_number,
                "article_title": article_title,
                "first_chunk_id": chunk.id,
            }
        else:
            current_parts.append(chunk.content)

    _flush()
    return groups


_SINGLE_ARTICLE_RE = re.compile(r"(?i)điều\s*(\d+[a-zA-Z]?)")


async def _fetch_article_chunks_from_db(
    article_number: str,
    doc_ids: list[uuid.UUID],
    db: AsyncSession,
) -> list[RankedChunk]:
    """
    Lấy TẤT CẢ chunk của 1 Điều cụ thể từ DB (theo article_number trong
    chunk_metadata), gộp thành 1 RankedChunk đầy đủ mỗi document — không
    qua similarity search, không bị cắt bởi reranker top_k.

    Dùng cho câu hỏi "Điều X quy định gì?" để đảm bảo không bỏ sót khoản
    nào, thay vì phụ thuộc vào similarity search chỉ lấy top-K chunk.
    """
    result = await db.execute(
        select(DocumentChunk, Document.original_filename)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.document_id.in_(doc_ids))
        .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
    )
    rows = result.all()

    groups: list[RankedChunk] = []
    current_parts: list[str] = []
    current_meta: dict = {}
    current_doc: uuid.UUID | None = None

    def _flush_group():
        if current_parts:
            groups.append(RankedChunk(
                original_id=str(current_meta.get("first_chunk_id", "")),
                content="\n".join(current_parts),
                rerank_score=1.0,
                payload=dict(current_meta),
            ))

    for chunk, filename in rows:
        meta = chunk.chunk_metadata or {}
        art = str(meta.get("article_number") or "").strip()
        if art.lower() != article_number.lower():
            continue

        if current_doc != chunk.document_id and current_parts:
            _flush_group()
            current_parts = []

        if not current_parts:
            current_meta = {
                "document_id": str(chunk.document_id),
                "original_filename": filename,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "chapter": str(meta.get("chapter") or ""),
                "article_number": art,
                "article_title": str(meta.get("article_title") or ""),
                "first_chunk_id": chunk.id,
            }
            current_doc = chunk.document_id

        current_parts.append(chunk.content)

    _flush_group()
    return groups


def _is_single_article_question(question: str) -> bool:
    """
    True khi câu hỏi hỏi VỀ NỘI DUNG của đúng 1 Điều cụ thể (ví dụ:
    "Điều 10 quy định gì?", "Điều 8 nói về gì", "nội dung Điều 5 là gì")
    VÀ không phải yêu cầu tóm tắt (đã được _is_full_summary_request xử lý).
    Điều kiện: có đúng 1 match điều/khoản, match đó là "điều N".
    """
    q = question.lower()
    ref_matches = _SPECIFIC_REF_PATTERN.findall(q)
    if len(ref_matches) != 1:
        return False
    single = ref_matches[0].strip().lower()
    return bool(re.match(r"^điều\s*\d+[a-zA-Z]?$", single, re.IGNORECASE))


_BROKEN_COMPARISON_MARKERS = (
    "###", "**điểm", "điểm giống nhau", "điểm khác nhau", "dưới đây là",
    "quy tắc", "1. **", "2. **", "- [", "\n1.", "\n2.",
    "no enough information", "not enough information", "cannot compare",
    "insufficient information",
)
_CJK_PATTERN = re.compile(r"[一-鿿]")
_CONTENT_SIMILARITY_THRESHOLD = 0.55


def _content_similarity(text_a: str, text_b: str) -> float:
    """
    Đo độ giống nhau về NỘI DUNG THẬT (không phải tiêu đề Điều) giữa 2 đoạn
    văn bản, dùng để quyết định có CẦN gọi LLM so sánh hay không. 2 tài liệu
    trong tính năng này có thể thực chất là cùng 1 nguồn (VD: 1 bản là bản
    tóm tắt của bản kia) — khi đó nội dung mỗi Điều gần như giống hệt nhau,
    và việc bắt model 0.5B "bịa" ra 1 câu khác biệt không có thật là nguyên
    nhân chính gây ra output vô nghĩa/lạc đề. Cắt bớt độ dài trước khi so
    khớp (quick_ratio) để tránh tốn thời gian với các Điều dài.
    """
    norm_a = " ".join(text_a.lower().split())[:2000]
    norm_b = " ".join(text_b.lower().split())[:2000]
    if not norm_a or not norm_b:
        return 0.0
    return difflib.SequenceMatcher(None, norm_a, norm_b).quick_ratio()


def _sanitize_comparison(text: str, max_chars: int = 280) -> str:
    """Chỉ giữ lại ĐÚNG 1 câu văn sạch từ output so sánh của LLM; nếu output
    "vỡ format" (markdown nhiều đoạn, danh sách, lặp lại đề bài, lẫn tiếng
    Anh/chữ Hán...) thì coi như thất bại — trả về chuỗi rỗng để nơi gọi dùng
    câu fallback an toàn thay vì hiển thị nguyên văn rác cho người dùng."""
    if not text:
        return ""
    first_line = text.strip().split("\n")[0].strip()
    first_line = re.sub(r"^[#>\-\*\d\.\)\s]+", "", first_line).strip()
    lowered = first_line.lower()
    if not first_line or any(m in lowered for m in _BROKEN_COMPARISON_MARKERS):
        return ""
    if any(m in text.lower() for m in ("###", "\n1.", "\n2.", "\n- ")):
        return ""
    if _CJK_PATTERN.search(first_line):
        return ""
    snippet = _extractive_snippet(first_line, max_chars=max_chars)
    # Nếu bị cắt cụt giữa câu (không kết thúc bằng dấu câu) → đánh dấu rõ
    # bằng "…" thay vì để trông như câu bị lỗi.
    if snippet and snippet[-1] not in ".!?…)":
        snippet = snippet.rstrip(",;: ") + "…"
    return snippet


_COMPARE_KEYWORDS = (
    "so sánh", "so sanh", "khác nhau", "khac nhau", "giống nhau", "giong nhau",
    "khác biệt", "khac biet", "điểm khác", "diem khac", "điểm giống", "diem giong",
    "compare",
)


_NON_CONTENT_COMPARE_KEYWORDS = (
    "văn phong", "van phong", "phong cách", "phong cach", "cách viết", "cach viet",
    "mức độ cụ thể", "muc do cu the", "chi tiết hơn", "chi tiet hon",
    "cái nào kỹ hơn", "cai nao ky hon", "dài hơn", "dai hon", "ngắn hơn", "ngan hon",
    "bố cục", "bo cuc", "trình bày", "trinh bay", "hình thức", "hinh thuc",
)


_DIFF_ONLY_KEYWORDS = (
    "khác nhau", "khac nhau", "khác biệt", "khac biet", "điểm khác", "diem khac",
)
_SAME_ONLY_KEYWORDS = (
    "giống nhau", "giong nhau", "điểm giống", "diem giong", "tương đồng", "tuong dong",
)


def _compare_mode(question: str) -> str:
    """
    Người dùng có thể chỉ muốn xem 1 LOẠI thông tin khi so sánh:
    - "diff": chỉ hỏi khác nhau ("tìm điểm khác nhau") → CHỈ liệt kê các Điều
      có khác biệt, ẩn hẳn các Điều giống nhau (trước đây luôn hiện cả 2 loại
      dù người dùng chỉ hỏi 1 loại, gây khó đọc/không đúng ý).
    - "same": chỉ hỏi giống nhau → CHỈ liệt kê các Điều giống nhau.
    - "both": hỏi so sánh chung chung, hoặc hỏi cả giống lẫn khác → hiện cả 2.
    """
    q = question.lower()
    wants_diff = any(k in q for k in _DIFF_ONLY_KEYWORDS)
    wants_same = any(k in q for k in _SAME_ONLY_KEYWORDS)
    if wants_diff and not wants_same:
        return "diff"
    if wants_same and not wants_diff:
        return "same"
    return "both"


def _compare_keywords_match(question: str) -> bool:
    """Pre-filter nhanh bằng keyword — True nếu câu CÓ THỂ là so sánh."""
    q = question.lower()
    return any(k in q for k in _COMPARE_KEYWORDS)


def _is_compare_request(question: str) -> bool:
    """True nếu câu hỏi là yêu cầu so sánh NỘI DUNG pháp lý giữa 2 tài liệu."""
    q = question.lower()
    if not any(k in q for k in _COMPARE_KEYWORDS):
        return False
    # Câu hỏi về hình thức/văn phong → không phải so sánh nội dung pháp lý
    if any(k in q for k in _NON_CONTENT_COMPARE_KEYWORDS):
        return False
    return True


def _article_label(chunk) -> str:
    article_number = chunk.payload.get("article_number") or ""
    article_title = chunk.payload.get("article_title") or ""
    label = f"Điều {article_number}" if article_number else "Phần không có số"
    if article_title:
        label += f" ({article_title})"
    return label


def _pair_articles(groups_a: list, groups_b: list, threshold: float = 0.45) -> list[tuple]:
    # Ghep cap cac Dieu "tuong ung" giua 2 tai lieu KHAC NHAU bang CODE xac
    # dinh (khong dung LLM) - so khop article_title bang do tuong dong chuoi
    # (difflib), ghep greedy theo diem cao nhat truoc de tranh 1 dieu bi ghep
    # nham voi nhieu dieu khac. Cung so thu tu Dieu giua 2 luat KHAC NHAU
    # khong co nghia la cung noi dung, nen KHONG dung article_number de ghep,
    # chi dung lam nhan hien thi. Dieu nao khong tim duoc cap tuong ung ben
    # kia thi van giu lai rieng (chi co o 1 tai lieu) - khong am tham bo qua.
    scored: list[tuple[float, int, int]] = []
    for i, ca in enumerate(groups_a):
        title_a = (ca.payload.get("article_title") or "").strip().lower()
        if not title_a:
            continue
        for j, cb in enumerate(groups_b):
            title_b = (cb.payload.get("article_title") or "").strip().lower()
            if not title_b:
                continue
            score = difflib.SequenceMatcher(None, title_a, title_b).ratio()
            if score >= threshold:
                scored.append((score, i, j))

    scored.sort(key=lambda t: t[0], reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: dict[int, tuple] = {}
    for score, i, j in scored:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched[i] = (groups_a[i], groups_b[j])

    result: list[tuple] = []
    for i, ca in enumerate(groups_a):
        result.append(matched[i] if i in matched else (ca, None))
    for j, cb in enumerate(groups_b):
        if j not in used_b:
            result.append((None, cb))
    return result


# ── Case #10: Conversational RAG ─────────────────────────────────────────────

_FOLLOWUP_MARKERS = (
    "chi tiết hơn", "cụ thể hơn", "rõ hơn", "giải thích thêm", "nói thêm", "thêm nữa",
    "còn gì", "còn nữa", "tiếp đi", "tiếp tục", "thế còn", "vậy còn", "còn điều", "còn khoản",
    "điều đó", "điều này", "cái đó", "cái này", "ý trên", "phần trên", "vừa rồi", "vừa nói",
    "như trên", "ở trên", "tại sao", "vì sao", "thì sao", "trường hợp đó", "trường hợp này",
    "nhất có thể", "ngắn nhất", "hơn đi",
    "ngắn gọn", "ngắn hơn", "dài hơn", "hơn nữa", "súc tích", "rút gọn", "viết lại",
    "bằng bảng", "gạch đầu dòng", "bằng tiếng anh", "bằng tiếng việt",
    "more detail", "what about", "explain more", "why is that", "shorter", "in english",
)
_FOLLOWUP_MAX_WORDS = 3
_HISTORY_MAX_MESSAGES = 6          # 2 lượt hỏi-đáp gần nhất là đủ để hiểu câu nối tiếp
_STOPWORDS = {
    "là", "gì", "và", "của", "có", "không", "được", "cho", "các", "những", "này", "đó",
    "thì", "sao", "với", "trong", "về", "như", "nào", "hãy", "tôi", "bạn", "thế", "vậy",
    "hơn", "còn", "một", "the", "what", "and", "for",
}


_NEG_WORDS = r"(?:không|khong|ko|đừng|dung|chớ|chẳng|khỏi|bỏ\s+qua)"
_NEG_FILLER = r"(?:cần|can|muốn|muon|phải|phai|hãy|hay|có|co)"
_NEG_TRAIL_WORDS = (
    "lại", "lai", "giúp", "giup", "cho", "tôi", "toi", "mình", "minh", "file", "này", "nay",
    "tài", "liệu", "văn", "bản", "nó", "các", "những", "hết", "toàn", "bộ", "đâu", "nhé", "nha",
)
_NEG_INTENT_KEYWORDS = tuple(_SUMMARY_KEYWORDS) + ("so sánh", "so sanh", "compare")
_NEG_INTENT_RE = re.compile(
    rf"{_NEG_WORDS}(?:\s+{_NEG_FILLER})?\s+(?:{'|'.join(map(re.escape, _NEG_INTENT_KEYWORDS))})"
    rf"(?:\s+(?:{'|'.join(_NEG_TRAIL_WORDS)}))*",
    re.IGNORECASE,
)
_DOC_FILLER_WORDS = {"file", "tài", "liệu", "văn", "bản", "này", "nay", "cho", "tôi", "mình", "nhé", "nha", "đâu"}
_NEGATION_GUIDE = (
    "Mình hiểu là bạn KHÔNG muốn tóm tắt/so sánh. Bạn muốn hỏi nội dung cụ thể nào "
    "trong tài liệu? (VD: \"Điều 5 quy định gì?\", \"Quyền của người thuê nhà là gì?\")"
)


def _strip_negated_intents(question: str) -> tuple[str, bool]:
    """
    Phát hiện phủ định đi kèm từ khóa tóm tắt/so sánh ("không tóm tắt file này",
    "đừng so sánh"). Trả về (câu hỏi đã tước phủ định, có_phủ_định).
    Dùng regex thuần — không gọi LLM để tránh model nhỏ classify sai.
    """
    cleaned = _NEG_INTENT_RE.sub(" ", question)
    if cleaned == question:
        return question, False
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-—?!")
    return cleaned, True


_CONSTRAINT_SPLIT_RE = re.compile(
    r"\s*(?:,|;|-|—)?\s*\b(?:mà\s+không|nhưng\s+không|và\s+không|không\s+cần|không\s+được|"
    r"đừng|chớ|ngoại\s+trừ|trừ|bỏ\s+qua|loại\s+bỏ|loại\s+trừ|không\s+bao\s+gồm|bỏ)\b",
    re.IGNORECASE,
)
_BRIEF_KEYWORDS = ("ngắn gọn", "ngắn hơn", "súc tích", "vắn tắt", "rút gọn", "ngắn nhất", "nhất có thể")
_EXTREME_BRIEF_KEYWORDS = ("ngắn nhất", "nhất có thể", "cực ngắn")
_HANDLED_CONSTRAINT_WORDS = {
    "không", "cần", "được", "đề", "cập", "tới", "đến", "nhắc", "mà", "đâu", "nhé", "nha",
    "hơn", "nữa", "ngắn", "gọn", "trừ", "ngoại", "bỏ", "qua", "đừng", "chớ", "và", "nhưng",
    "loại", "bao", "gồm", "đi", "nhất", "thể",
}


def _split_constraints(question: str) -> tuple[str, str]:
    '''
    Tách "nhiệm vụ" khỏi "ràng buộc" khi câu là yêu cầu tóm tắt/so sánh kèm điều
    kiện phủ định (VD: "tóm tắt file này mà không cần đề cập điều 5"). Nếu không
    tách, số "điều 5" trong ràng buộc bị hiểu nhầm là câu hỏi về Điều 5.
    Chỉ tách khi phần TRƯỚC có từ khóa tóm tắt/so sánh (tránh cắt nhầm câu hỏi
    thường như "người thuê nhà không được làm gì").
    '''
    # Phần sau " — " là câu nối tiếp được ghép thêm (concat) → không thuộc ràng buộc
    head = question.split(" — ")[0]
    m = _CONSTRAINT_SPLIT_RE.search(head)
    if not m or m.start() == 0:
        return question, ""
    task, constraint = head[: m.start()].strip(), head[m.start():].strip(" ,;-—")
    low = task.lower()
    if any(k in low for k in _NEG_INTENT_KEYWORDS) and _has_real_content(task):
        return task, constraint
    return question, ""


_NUM_OR_RANGE = r"\d+(?:\s*(?:-|đến|tới)\s*\d+)?"
# "điều 6", "điều 6 và 7", "điều 6, 7 và điều 9", "điều 6-8", "điều 6 đến 8"
_ARTICLE_LIST_RE = re.compile(
    rf"điều\s*(?:số\s*)?{_NUM_OR_RANGE}(?:\s*(?:,|;|và|với|hoặc)\s*(?:điều\s*(?:số\s*)?)?{_NUM_OR_RANGE})*",
    re.IGNORECASE,
)


def _excluded_articles(constraint: str) -> set[str]:
    '''Các số Điều mà ràng buộc yêu cầu BỎ QUA (VD: "không cần đề cập điều 6 và 7").'''
    result: set[str] = set()
    for m in _ARTICLE_LIST_RE.finditer(constraint.lower()):
        text = re.sub(r"^điều\s*(?:số\s*)?", "", m.group(0))
        for a, b in re.findall(r"(\d+)(?:\s*(?:-|đến|tới)\s*(\d+))?", text):
            lo = int(a)
            hi = int(b) if b else lo
            if lo <= hi <= lo + 300:
                result.update(str(n) for n in range(lo, hi + 1))
    return result


def _unhandled_constraint(constraint: str) -> str:
    '''Phần ràng buộc hệ thống KHÔNG thực hiện được (VD: định dạng "không dùng dấu chấm").'''
    rest = _ARTICLE_LIST_RE.sub(" ", constraint.lower())
    words = [w for w in re.findall(r"\w+", rest) if w not in _HANDLED_CONSTRAINT_WORDS]
    return constraint if len(words) >= 1 else ""


def _has_real_content(text: str) -> bool:
    return bool(_content_words(text) - _DOC_FILLER_WORDS)


_UNAMBIGUOUS_SUMMARY_RE = re.compile(
    # Dạng 1: câu BẮT ĐẦU bằng lệnh tóm tắt (động từ)
    # VD: "tóm tắt file này", "hãy tóm tắt chương II", "cho tôi tóm tắt",
    #     "giúp tóm tắt" (không từ trung gian), "giúp tôi tóm tắt" (có từ trung gian)
    # giúp\s+(?:\w+\s+)? → từ trung gian OPTIONAL để khớp cả 2 dạng
    r"^(?:hãy\s+|hay\s+|cho\s+\w+\s+|giúp\s+(?:\w+\s+)?|làm\s+ơn\s+)?"
    r"(?:tóm\s*tắt|tom\s*tat|khái\s*quát|khai\s*quat|tổng\s*quan|tong\s*quan|tổng\s*kết|tong\s*ket|summary|summarize)"
    r"|"
    # Dạng 2: câu lịch sự "bạn/anh/chị có thể tóm tắt ... không"
    # VD: "bạn có thể tóm tắt file này không", "có thể tóm tắt nội dung chính không"
    r"^(?:\w+\s+)?có\s+thể\s+"
    r"(?:tóm\s*tắt|tom\s*tat|khái\s*quát|tổng\s*quan|tổng\s*kết|summary|summarize)",
    re.IGNORECASE,
)


async def _confirm_summary_intent(question: str) -> bool:
    '''
    Xác nhận câu có từ khóa tóm tắt có thực sự là yêu cầu tóm tắt không.
    Không đếm chữ — dùng regex cho dạng câu rõ ràng, dùng LLM cho câu mơ hồ.
    LLM đóng vai trò hiểu ngôn ngữ tự nhiên, không làm gatekeeper cứng.
    Fallback khi LLM không parse được: coi là SUMMARY (đã qua keyword gate rồi).
    '''
    # Câu lệnh hoặc câu lịch sự rõ ràng → không cần LLM
    if _UNAMBIGUOUS_SUMMARY_RE.match(question.strip()):
        logger.info(f"_confirm_summary_intent: unambiguous → True ({question[:60]!r})")
        return True
    # Câu mơ hồ → nhờ LLM phân tích ngữ nghĩa
    label = await llm_service.classify_summary_intent(question)
    logger.info(f"Intent LLM cho {question[:60]!r}: {label or 'n/a'}")
    # LLM không parse được ("") → fallback True (đã qua _is_full_summary_request rồi)
    return label != "OTHER"


def _find_articles_with_topic(groups: list, topic: str, also: list[str] | None = None) -> list[tuple]:
    '''
    Quét CODE (không bỏ sót) các Điều có chứa cụm chủ đề hoặc các từ liên quan do LLM gợi ý.
    Trả [(group, tổng số lần nhắc)]; group.payload["matched"] ghi các cụm đã khớp.
    '''
    import unicodedata
    norm = lambda s: " ".join(unicodedata.normalize("NFC", s).lower().split())
    terms = [norm(topic)] + [norm(x) for x in (also or [])]
    hits = []
    for g in groups:
        if not str(g.payload.get("article_number") or "").strip():
            continue   # chỉ liệt kê các Điều thật sự
        text = norm(g.content)
        counts = {t: text.count(t) for t in terms}
        matched = [t for t, c in counts.items() if c]
        if matched:
            g.payload = {**g.payload, "matched": matched}
            hits.append((g, sum(counts.values())))
    return hits


def _topic_sentence(content: str, topic: str, max_chars: int = 220) -> str:
    '''Câu NGUYÊN VĂN đầu tiên chứa chủ đề (chỉ lấy khi câu trọn vẹn, không cắt giữa chừng).'''
    import unicodedata
    t = unicodedata.normalize("NFC", topic).lower()
    text = " ".join(unicodedata.normalize("NFC", content).split())
    for s in _SENT_SPLIT_RE.split(text):
        if t in s.lower() and len(s) <= max_chars:
            return s.strip()
    return ""


_PLAN_LENGTHS = {"titles", "brief", "normal", "detail"}
_NEGATION_WORD_RE = re.compile(r"\b(?:không|khong|ko|đừng|chớ|chẳng|khỏi|bỏ\s+qua)\b", re.IGNORECASE)


def _validate_plan(plan: dict | None, user_texts: list[str], question: str) -> dict | None:
    '''
    Kiểm tra kết quả của planner LLM — model nhỏ có thể bịa số/nhãn, nên:
      - action/length phải thuộc tập nhãn hợp lệ;
      - MỌI số (first_n, from, to, only, exclude…) phải xuất hiện trong chữ số của
        câu hỏi hoặc các câu hỏi trước (không cho LLM tự nghĩ ra số Điều);
      - "summary" phải có tín hiệu tóm tắt trong câu mới hoặc ở lượt trước;
      - "clarify" chỉ hợp lệ khi câu mới thật sự có từ phủ định;
      - câu viết lại (qa) không được dài, lẫn chữ Hán, hay là câu "trả lời".
    Không đạt → None (dùng logic dự phòng).
    '''
    if not isinstance(plan, dict):
        return None
    action = str(plan.get("action", "")).lower()
    if action not in {"summary", "qa", "clarify", "find"}:
        return None
    allowed = {int(n) for t in user_texts for n in re.findall(r"\d+", t)}
    # Khoảng "4-9", "4 đến 9", "4 tới 9" → cho phép mọi số nằm trong khoảng
    for t in user_texts:
        for a, b in re.findall(r"(\d+)\s*(?:-|–|đến|tới)\s*(\d+)", t.lower()):
            lo, hi = int(a), int(b)
            if lo <= hi <= lo + 300:
                allowed.update(range(lo, hi + 1))

    def _ints(v) -> list[int] | None:
        if v is None:
            return []
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = [v]
        if not isinstance(v, list):
            return None
        out = []
        for x in v:
            try:
                n = int(x)
            except (TypeError, ValueError):
                return None
            if n not in allowed or not (0 < n < 1000):
                return None
            out.append(n)
        return out

    q_low = question.lower()
    summary_signal = any(k in q_low for k in _SUMMARY_KEYWORDS) or any(
        any(k in t.lower() for k in _SUMMARY_KEYWORDS) for t in user_texts[:-1]
    )
    # "liệt kê/nêu 12 điều cuối": không có từ "tóm tắt" nhưng chính câu mới nêu 1 phạm vi Điều bằng SỐ
    # có thật trong câu → đủ bằng chứng là yêu cầu theo phạm vi Điều (không cần thêm từ khoá cứng).
    _q_ints = {int(n) for n in re.findall(r"\d+", question)}
    _range_vals = []
    for _k in ("first_n", "last_n", "from", "to", "only"):
        _v = plan.get(_k)
        if _v is None:
            continue
        _range_vals.extend(_v if isinstance(_v, list) else [_v])
    own_range = False
    try:
        own_range = bool(_range_vals) and all(int(x) in _q_ints for x in _range_vals)
    except (TypeError, ValueError):
        own_range = False
    if own_range:
        summary_signal = True
    out: dict = {"action": action, "exclude": [], "length": None, "range": {"type": "all"}, "question": "", "topic": "", "scope": None}
    # Phạm vi tài liệu do LLM đề xuất: "all" | "same" | tên file. Tên file phải có thật trong hội thoại
    # (không cho LLM tự nghĩ ra); kiểm tra tiếp với danh sách tài liệu ở bước áp dụng.
    _sc = " ".join(str(plan.get("scope") or "").lower().split())
    if _sc in ("all", "same"):
        out["scope"] = _sc
    elif _sc:
        _sc_stem = Path(_sc).stem.strip()
        # Tên file phải do CHÍNH câu mới nhắc tới; nếu chỉ lặp lại file ở "Đang làm việc trên"
        # (LLM nhỏ hay chép) thì bỏ → quy tắc dự phòng quyết định (chủ đề mới = tìm trên tất cả file).
        _ctx_all = " ".join(question.lower().split())
        if len(_sc_stem) >= 2 and re.search(r"(?<!\w)" + re.escape(_sc_stem) + r"(?!\w)", _ctx_all):
            out["scope"] = _sc

    if action == "clarify":
        return out if _NEGATION_WORD_RE.search(question) else None

    if action == "find":
        topic = " ".join(str(plan.get("topic") or "").lower().split())
        # Chủ đề phải CHÉP NGUYÊN từ câu hỏi (không cho LLM tự nghĩ ra) và câu phải có dấu hiệu "tìm/liệt kê"
        # Chủ đề có thể nằm ở câu hiện tại HOẶC các câu hỏi trước (câu nối tiếp "liệt kê đầy đủ đi")
        _ctx = " ".join(" ".join(user_texts).lower().split())
        # LLM hay chuẩn hoá cụm chủ đề ("mua , bán nhà" → "mua nhà"): chấp nhận khi MỌI từ của chủ đề
        # đều có trong hội thoại (vẫn không cho LLM tự nghĩ ra từ mới).
        def _words_in(text: str) -> bool:
            _w = set(re.findall(r"\w+", text))
            return all(x in _w for x in re.findall(r"\w+", topic))
        if not (2 <= len(topic) <= 40) or not _words_in(_ctx):
            return None
        # Chủ đề chỉ được mượn từ lượt trước khi câu mới là câu nối tiếp thuần tuý;
        # câu có yêu cầu tóm tắt riêng ("tóm tắt 10 điều đầu…") là yêu cầu mới, không bị chủ đề cũ chi phối.
        if not _words_in(" ".join(q_low.split())):
            # Câu nối tiếp = LLM đánh dấu "continue": true (câu nhắc lại nội dung lượt trước) hoặc
            # dạng câu cụt đã nhận ra từ trước; câu có yêu cầu tóm tắt riêng thì không bao giờ mượn chủ đề.
            _continue = plan.get("continue") is True or _looks_like_followup(question)
            # Từ "tóm gọn/tóm tắt" chỉ chặn việc mượn chủ đề khi câu có SỐ riêng ("tóm tắt 10 điều đầu");
            # "nói tóm gọn thôi" (không số) sau 1 lượt liệt kê = yêu cầu rút gọn danh sách đó.
            if (any(k in q_low for k in _SUMMARY_KEYWORDS) and re.search(r"\d", question)) or not _continue:
                return None
        # Từ đồng nghĩa/liên quan do LLM gợi ý (tuỳ chọn) — chỉ dùng để quét thêm, hiển thị rõ từ khớp
        also = plan.get("also")
        terms = []
        if isinstance(also, list):
            for x in also[:3]:
                x = " ".join(str(x).lower().split())
                if 2 <= len(x) <= 40 and x != topic and x not in terms:
                    terms.append(x)
        # LLM hay giữ chủ đề cũ và đẩy chủ đề MỚI của câu hiện tại vào "also" ("về mua bán nhà thì sao"
        # sau lượt "chung cư") → nếu chủ đề không có trong câu mới mà một từ "also" lại có, đó mới là chủ đề thật.
        _q_norm = " ".join(q_low.split())
        if not _words_in(_q_norm):
            _new = [t for t in terms if t in _q_norm]
            if _new:
                terms = [t for t in [topic] + terms if t != _new[0]][:3]
                topic = _new[0]
        out["topic"] = topic
        out["also"] = terms
        if str(plan.get("length") or "") in _PLAN_LENGTHS:
            out["length"] = str(plan["length"])
        return out

    exclude = _ints(plan.get("exclude"))
    if exclude is None:
        return None
    # LLM thường chỉ trả 2 đầu mút của khoảng ("bỏ điều 3-9" → [3, 9]) → tự lấp đầy bằng code
    exclude_set = set(exclude)
    for a, b in re.findall(r"(\d+)\s*(?:-|–|đến|tới)\s*(\d+)", question.lower()):
        lo, hi = int(a), int(b)
        if lo in exclude_set and hi in exclude_set and lo <= hi <= lo + 300:
            exclude_set.update(range(lo, hi + 1))
    out["exclude"] = [str(n) for n in sorted(exclude_set)]
    length = plan.get("length")
    if length is not None:
        if str(length) not in _PLAN_LENGTHS:
            return None
        out["length"] = str(length)

    if action == "qa":
        rq = _clean_rewrite(str(plan.get("question") or ""))
        if rq:
            if (
                len(rq) > 300
                or _CJK_PATTERN.search(rq)
                or rq.lower().startswith(("bạn", "tôi", "xin", "dạ", "mình"))
            ):
                return None
            # Câu viết lại phải mang thêm từ khóa từ hội thoại trước (không tự bịa chủ đề)
            ctx_words = _content_words(" ".join(user_texts[:-1]))
            if not ((_content_words(rq) - _content_words(question)) & ctx_words):
                return None
            out["question"] = rq
        return out

    # action == "summary"
    if not summary_signal:
        return None
    # Câu hiện tại KHÔNG tự nhắc tóm tắt → chỉ hợp lệ khi nó là câu chỉnh sửa/nối tiếp
    # của lượt tóm tắt trước ("bỏ điều 3", "ngắn hơn", "chi tiết hơn"). Một câu hỏi
    # mới hoàn chỉnh về chủ đề khác ("có điều nào liên quan tới chung cư không")
    # tuyệt đối không được bị kéo vào nhánh tóm tắt chỉ vì lịch sử có tóm tắt.
    if not any(k in q_low for k in _SUMMARY_KEYWORDS) and not own_range and not _looks_like_followup(question):
        return None
    rng: dict = {"type": "all"}
    first_n, last_n = _ints(plan.get("first_n")), _ints(plan.get("last_n"))
    frm, to = _ints(plan.get("from")), _ints(plan.get("to"))
    only = _ints(plan.get("only"))
    if None in (first_n, last_n, frm, to, only):
        return None
    if first_n:
        rng = {"type": "first_n", "n": first_n[0]}
    elif last_n:
        rng = {"type": "last_n", "n": last_n[0]}
    elif frm and to and frm[0] <= to[0]:
        rng = {"type": "range", "from": frm[0], "to": to[0]}
    elif only:
        rng = {"type": "list", "list": only}
    elif plan.get("chapter"):
        ch = str(plan["chapter"]).strip().upper()
        if not re.fullmatch(r"[IVXLCDM\d]{1,6}", ch) or ch.lower() not in " ".join(user_texts).lower():
            return None
        rng = {"type": "chapter", "chapter": ch}
    out["range"] = rng
    return out


_EXCLUSION_RE = re.compile(
    r"^\s*(?:và\s+|thêm\s+|hãy\s+)?(?:bỏ(?:\s+qua)?|loại(?:\s+bỏ|\s+trừ)?|trừ|ngoại\s+trừ|"
    r"không\s+(?:cần|lấy|tính|bao\s+gồm)|đừng)\b",
    re.IGNORECASE,
)


def _is_exclusion_followup(text: str) -> bool:
    '''Câu kiểu "bỏ điều 3", "loại điều 2 và 4 ra" — chỉnh lại kết quả trước, không đứng độc lập.'''
    return bool(_EXCLUSION_RE.search(text)) and bool(_ARTICLE_LIST_RE.search(text.lower()))


def _looks_like_followup(question: str) -> bool:
    """
    Câu hỏi có phụ thuộc vào ngữ cảnh trước không? Chỉ khi có dấu hiệu rõ (từ
    tham chiếu kiểu "chi tiết hơn", "thì sao"... hoặc cực ngắn) mới xử lý nối
    tiếp — câu hỏi đã đầy đủ ý thì giữ nguyên, KHÔNG gọi LLM viết lại vô ích
    (vừa tốn thời gian vừa có rủi ro model nhỏ làm méo câu hỏi gốc).
    """
    q = f" {question.lower().strip()} "
    return (
        any(m in q for m in _FOLLOWUP_MARKERS)
        or len(question.split()) <= _FOLLOWUP_MAX_WORDS
        or _is_exclusion_followup(question)
    )


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) >= 2 and w not in _STOPWORDS}


async def _load_recent_history(
    session_id: uuid.UUID | None,
    user: User,
    db: AsyncSession,
) -> list[tuple[str, str]]:
    """Lấy vài tin nhắn gần nhất của đúng phiên chat (và đúng chủ sở hữu phiên)."""
    if not session_id:
        return []
    result = await db.execute(
        select(ChatMessage.role, ChatMessage.content)
        .join(ChatSession, ChatSession.id == ChatMessage.session_id)
        .where(ChatMessage.session_id == session_id, ChatSession.user_id == user.id)
        .order_by(
            ChatMessage.created_at.desc(),
            # created_at = now() của transaction -> tin user & assistant cùng lượt
            # trùng timestamp; assistant phải được coi là MỚI hơn user.
            case((ChatMessage.role == "assistant", 1), else_=0).desc(),
        )
        .limit(_HISTORY_MAX_MESSAGES)
    )
    rows = list(reversed(result.all()))
    history: list[tuple[str, str]] = []
    for role, content in rows:
        limit = 300 if role == "user" else 500   # rút gọn để không làm phình prompt
        history.append((role, (content or "")[:limit]))
    return history


async def _load_last_memory(session_id: uuid.UUID | None, user: User, db: AsyncSession) -> dict:
    """Trạng thái bộ nhớ hội thoại ({used_chars, summary, compactions}) của lượt trả lời gần nhất."""
    if not session_id:
        return {"used_chars": 0, "summary": "", "compactions": 0}
    result = await db.execute(
        select(ChatMessage.retrieval_metadata)
        .join(ChatSession, ChatSession.id == ChatMessage.session_id)
        .where(
            ChatMessage.session_id == session_id,
            ChatSession.user_id == user.id,
            ChatMessage.role == "assistant",
        )
        .order_by(ChatMessage.created_at.desc())
        .limit(4)
    )
    for (meta,) in result.all():
        if isinstance(meta, dict) and isinstance(meta.get("memory"), dict):
            m = meta["memory"]
            return {
                "used_chars": int(m.get("used_chars", 0) or 0),
                "summary": str(m.get("summary", "") or ""),
                "compactions": int(m.get("compactions", 0) or 0),
            }
    return {"used_chars": 0, "summary": "", "compactions": 0}


def _digest_turns(turns: list[dict], total_chars: int) -> str:
    """
    Nén các lượt hội thoại cũ thành ghi nhớ NGUYÊN VĂN (extractive): với mỗi lượt giữ câu hỏi + các câu của câu trả lời
    liên quan nhất tới câu hỏi (chấm bằng reranker; câu có "Điều N" được ưu tiên), kèm Điều/văn bản đã dẫn.
    Không dùng LLM viết lại → không thể bịa; lượt càng nhiều thì mỗi lượt được giữ càng ít.
    """
    if not turns:
        return ""
    per_turn = max(160, total_chars // len(turns))
    flat: list[tuple[int, str]] = []
    for i, tr in enumerate(turns):
        text = re.sub(r"^[#>*\-•\s]+", "", tr["answer"], flags=re.M)
        for s in _UNIT_SPLIT.split(text):
            s = " ".join((s or "").split())
            if len(s) >= 25 and not s.startswith("Nguồn"):
                flat.append((i, s))
    flat = flat[:150]
    try:
        scores = [0.0] * len(flat)
        # chấm theo từng câu hỏi (các câu của cùng lượt dùng chung câu hỏi của lượt đó)
        for i in range(len(turns)):
            idx = [k for k, (ti, _) in enumerate(flat) if ti == i]
            if idx:
                sc = reranker_service.score_texts(turns[i]["question"], [flat[k][1] for k in idx])
                for k, s_ in zip(idx, sc):
                    scores[k] = s_
    except Exception as exc:
        logger.warning(f"[memory] không chấm điểm được ({exc}) → lấy các câu đầu")
        scores = [1.0 - k * 0.001 for k in range(len(flat))]
    out: list[str] = []
    for i, tr in enumerate(turns):
        idx = [k for k, (ti, _) in enumerate(flat) if ti == i]
        order = sorted(idx, key=lambda k: scores[k] + (0.3 if re.search(r"Điều\s+\d+", flat[k][1]) else 0.0), reverse=True)
        budget = per_turn - min(len(tr["question"]), 100) - len(tr.get("basis", "")) - 20
        keep, used = [], 0
        for k in order:
            if used + len(flat[k][1]) + 1 > budget:
                continue
            keep.append(k)
            used += len(flat[k][1]) + 1
        if not keep and order:          # câu quan trọng nhất dài hơn phần được chia → vẫn giữ TRỌN câu, không cắt cụt
            keep.append(order[0])
        ans = " ".join(flat[k][1] for k in sorted(keep)) if keep else ""
        line = f"• Hỏi: {tr['question'][:100]}"
        if ans:
            line += f"\n  Đáp: {ans}"
        if tr.get("basis"):
            line += f"\n  {tr['basis']}"
        out.append(line)
    while len(out) > 1 and sum(len(x) + 1 for x in out) > total_chars + 400:
        out.pop(0)                      # quá dài → bỏ nguyên cả lượt cũ nhất, không bao giờ cắt giữa câu
    return "\n".join(out)


async def _advance_memory(
    session_id: uuid.UUID | None, user: User, db: AsyncSession,
    prev: dict, question: str, answer: str, new_topic: bool = False,
) -> tuple[dict, bool]:
    """
    Cộng dồn mức dùng bộ nhớ sau 1 lượt. Nén ở chỗ hết ý (xem MEMORY_COMPACT_MIN/MAX) → NÉN: các lượt cũ được cô đọng thành ghi nhớ trích nguyên văn
    (câu hỏi + câu trả lời quan trọng nhất + Điều đã dẫn), chỉ giữ nguyên văn MEMORY_KEEP_TURNS lượt gần nhất.
    Trả về (state_mới, vừa_nén?).
    """
    budget = max(1000, settings.MEMORY_BUDGET_CHARS)
    used = prev["used_chars"] + len(question) + len(answer)
    state = {"used_chars": used, "summary": prev["summary"], "compactions": prev["compactions"]}
    p = used / budget
    lo, hi = settings.MEMORY_COMPACT_MIN, max(settings.MEMORY_COMPACT_MAX, settings.MEMORY_COMPACT_MIN)
    if p >= hi:
        keep = max(1, settings.MEMORY_KEEP_TURNS)        # buộc nén (đầy): giữ nguyên văn vài lượt gần nhất
    elif p >= lo and new_topic:
        keep = 1                                          # chủ đề cũ đã trọn → nén hết, chỉ giữ lượt mở đầu chủ đề mới
    else:
        if p >= 0.7:
            logger.info(f"[memory] {p:.0%} — đang cùng chủ đề, chờ trọn ý rồi mới nén (tối đa {hi:.0%})")
        return state, False
    older: list[dict] = []
    kept_chars = len(question) + len(answer)
    if session_id:
        rows = await db.execute(
            select(ChatMessage.role, ChatMessage.content, ChatMessage.sources)
            .join(ChatSession, ChatSession.id == ChatMessage.session_id)
            .where(ChatMessage.session_id == session_id, ChatSession.user_id == user.id)
            .order_by(ChatMessage.created_at.asc(), case((ChatMessage.role == "assistant", 1), else_=0).asc())
        )
        msgs = rows.all()
        turns: list[dict] = []
        for role, content, srcs in msgs:
            if role == "user":
                turns.append({"question": " ".join((content or "").split()), "answer": "", "basis": "", "chars": len(content or "")})
            elif turns:
                turns[-1]["answer"] = content or ""
                turns[-1]["chars"] += len(content or "")
                arts: dict[str, list[str]] = {}
                for s in (srcs or []):
                    if isinstance(s, dict) and s.get("article_number"):
                        arts.setdefault(s.get("document_name") or "", [])
                        lab = f"Điều {s['article_number']}"
                        if lab not in arts[s.get("document_name") or ""]:
                            arts[s.get("document_name") or ""].append(lab)
                if arts:
                    turns[-1]["basis"] = "Căn cứ: " + "; ".join(
                        f"{d + ' — ' if d else ''}{', '.join(v[:6])}" for d, v in list(arts.items())[:3]
                    )
        keep_prev = keep - 1                      # lượt hiện tại (chưa lưu) đã tính là 1 lượt giữ lại
        split = len(turns) - keep_prev if keep_prev > 0 else len(turns)
        split = max(0, split)
        older, kept = turns[:split], turns[split:]
        kept_chars += sum(t_["chars"] for t_ in kept)
    summary = await _run_sync(_digest_turns, older, settings.MEMORY_SUMMARY_CHARS) if older else prev["summary"]
    state = {"used_chars": len(summary) + kept_chars, "summary": summary, "compactions": prev["compactions"] + 1}
    logger.info(f"[memory] nén bộ nhớ: {used} → {state['used_chars']} ký tự (lần {state['compactions']}, {len(older)} lượt cũ)")
    return state, True


async def _load_last_task_state(
    session_id: uuid.UUID | None,
    user: User,
    db: AsyncSession,
) -> dict:
    """
    Trạng thái nhiệm vụ của lượt trả lời gần nhất trong phiên (file đang làm việc…),
    lưu trong retrieval_metadata["task_state"] — đọc thẳng thay vì đoán lại từ chữ trong lịch sử.
    """
    if not session_id:
        return {}
    result = await db.execute(
        select(ChatMessage.retrieval_metadata)
        .join(ChatSession, ChatSession.id == ChatMessage.session_id)
        .where(
            ChatMessage.session_id == session_id,
            ChatSession.user_id == user.id,
            ChatMessage.role == "assistant",
        )
        .order_by(ChatMessage.created_at.desc())
        .limit(4)
    )
    for (meta,) in result.all():
        if isinstance(meta, dict) and isinstance(meta.get("task_state"), dict):
            return meta["task_state"]
    return {}


def _clean_rewrite(text: str) -> str:
    first = (text or "").strip().split("\n")[0].strip().strip('"“”\'')
    first = re.sub(r"^(câu hỏi( độc lập)?|question)\s*:\s*", "", first, flags=re.IGNORECASE)
    return first.strip().strip('"“”\'').strip()


async def _resolve_followup(question: str, history: list[tuple[str, str]]) -> tuple[str, str]:
    """
    Trả về (câu hỏi độc lập, cách xử lý) với cách xử lý ∈ {"none", "llm", "concat"}.

    - Nếu lượt trước là yêu cầu ĐẶC BIỆT (tóm tắt toàn bộ / so sánh 2 tài liệu)
      → ghép chuỗi xác định "câu trước — câu hiện tại" để giữ nguyên ý định
      (VD: "tóm tắt file" + "chi tiết hơn" → vẫn vào nhánh tóm tắt, bản chi
      tiết). LLM viết lại dễ làm rơi mất các từ khóa kích hoạt nhánh này.
    - Ngược lại → nhờ LLM viết lại, rồi KIỂM TRA: câu viết lại phải mang thêm
      ngữ cảnh từ hội thoại (có từ khóa của lượt trước), không quá dài, không
      lẫn chữ Hán. Không đạt → quay về ghép chuỗi (an toàn, không bịa).
    """
    user_qs = [c for r, c in history if r == "user"]
    if not user_qs or not _looks_like_followup(question):
        return question, "none"
    # Neo vào câu hỏi GỐC gần nhất (bỏ qua các câu nối tiếp như "ngắn gọn hơn")
    base_idx = next(
        (i for i in range(len(user_qs) - 1, -1, -1) if not _looks_like_followup(user_qs[i])),
        len(user_qs) - 1,
    )
    last_user_q = user_qs[base_idx]
    # Gộp cả các câu chỉnh sửa nối tiếp giữa câu gốc và câu hiện tại
    # (VD: "tóm tắt 5 điều đầu" → "bỏ điều 3" → "ngắn hơn": giữ nguyên "bỏ điều 3")
    concat = " — ".join([last_user_q, *user_qs[base_idx + 1:], question])
    base_task, _ = _split_constraints(last_user_q)   # bỏ ràng buộc ("bỏ qua điều 6") trước khi xét ý định
    if _is_full_summary_request(base_task) or _is_compare_request(base_task):
        return concat, "concat"

    rewritten = _clean_rewrite(await llm_service.rewrite_followup_question(history, question))
    history_words = _content_words(" ".join(c for _, c in history))
    brought_context = (_content_words(rewritten) - _content_words(question)) & history_words
    if (
        rewritten
        and len(rewritten) <= 300
        and not _CJK_PATTERN.search(rewritten)
        and not rewritten.lower().startswith(("bạn", "tôi", "xin", "dạ", "mình"))   # model trả lời thay vì viết lại
        and brought_context
    ):
        return rewritten, "llm"
    return concat, "concat"


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
    query_vector = await _run_sync(embedding_service.embed_query, body.question)

    # ── Retrieval từ Qdrant ───────────────────────────────────────────
    candidates = await _run_sync(
        qdrant_service.search,
        query_vector=query_vector,
        top_k=settings.RETRIEVAL_TOP_K,
        document_ids=search_doc_ids,
        user_id=current_user.id,
    )

    # ── Reranking ─────────────────────────────────────────────────────
    ranked = await _run_sync(
        reranker_service.rerank,
        query=body.question,
        candidates=candidates,
        top_k=body.top_k,
    )
    ranked = reranker_service.filter_by_threshold(ranked, threshold=settings.RERANK_MIN_SCORE)

    # ── Sinh câu trả lời từ LLM ───────────────────────────────────────
    _q_lang = _detect_language(body.question)   # Case #07
    llm_answer = await llm_service.answer(body.question, ranked, lang=_q_lang)

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
        # QUAN TRỌNG: với StreamingResponse, FastAPI (0.115) đóng/commit session
        # của Depends(get_db) NGAY khi route trả về StreamingResponse — TRƯỚC khi
        # generator này chạy. Dùng lại session đó thì mọi thứ ghi trong stream
        # (phiên chat, tin nhắn) không bao giờ được commit → mất sạch, và phiên
        # chat vừa "tạo" không tồn tại ở request kế tiếp (lỗi 404 "Phiên chat
        # không tồn tại"). Nên stream tự mở session riêng và tự commit.
        db = AsyncSessionLocal()
        try:
            # ── Case #14: câu hỏi có dấu hiệu prompt injection → từ chối ngay,
            # KHÔNG đưa vào LLM (không cho model cơ hội "nghe theo") ─────────
            if is_injection_attempt(body.question):
                yield _sse({"type": "sources", "sources": []})
                yield _sse({"type": "token", "content": REFUSAL_MESSAGE})
                chat_session = await _get_or_create_session(body.session_id, current_user, db)
                db.add(ChatMessage(session_id=chat_session.id, role="user", content=body.question))
                assistant_msg = ChatMessage(
                    session_id=chat_session.id,
                    role="assistant",
                    content=REFUSAL_MESSAGE,
                    sources=[],
                    retrieval_metadata={"injection_blocked": True},
                )
                db.add(assistant_msg)
                await db.flush()
                if not chat_session.title:
                    chat_session.title = body.question[:80]
                await db.commit()
                yield _sse({
                    "type": "done",
                    "session_id": str(chat_session.id),
                    "message_id": str(assistant_msg.id),
                })
                return

            # ── Câu chào hỏi/xã giao — trả lời trực tiếp, bỏ qua RAG ────
            if _is_greeting(body.question):
                yield _sse({"type": "status", "stage": "generate", "message": "Đang trả lời…"})
                full_answer = ""
                async for token in llm_service.answer_chitchat_stream(body.question):
                    full_answer += token
                    yield _sse({"type": "token", "content": token})

                chat_session = await _get_or_create_session(body.session_id, current_user, db)
                db.add(ChatMessage(session_id=chat_session.id, role="user", content=body.question))
                assistant_msg = ChatMessage(
                    session_id=chat_session.id,
                    role="assistant",
                    content=full_answer,
                    sources=[],
                    retrieval_metadata={"chitchat": True},
                )
                db.add(assistant_msg)
                await db.flush()
                if not chat_session.title:
                    chat_session.title = body.question[:80]
                await db.commit()

                yield _sse({
                    "type": "done",
                    "session_id": str(chat_session.id),
                    "message_id": str(assistant_msg.id),
                })
                return

            # ── Lấy danh sách document được phép ──────────────────────
            explicit_multi_doc = False
            if body.document_ids:
                accessible = await _get_accessible_doc_ids(current_user, db)
                accessible_set = set(accessible)
                forbidden = [d for d in body.document_ids if d not in accessible_set]
                if forbidden:
                    yield _sse({"type": "error", "message": f"Không có quyền truy cập {len(forbidden)} tài liệu."})
                    return
                search_doc_ids = body.document_ids
                explicit_multi_doc = len(body.document_ids) > 1
            else:
                search_doc_ids = await _get_accessible_doc_ids(current_user, db)

            if not search_doc_ids:
                yield _sse({"type": "error", "message": "Bạn chưa có tài liệu nào. Hãy upload tài liệu trước."})
                return

            # ── Case #10: câu hỏi nối tiếp → viết lại thành câu hỏi độc lập dựa
            # trên lịch sử phiên chat. Từ đây trở xuống toàn bộ pipeline dùng
            # `question` (đã hiểu theo ngữ cảnh); câu gốc body.question chỉ dùng
            # để lưu lịch sử hiển thị cho người dùng ──────────────────────
            history = await _load_recent_history(body.session_id, current_user, db)
            _mem_prev = await _load_last_memory(body.session_id, current_user, db)
            logger.info(f"[history] session={body.session_id} loaded={len(history)} msgs")

            # ── Bộ hiểu yêu cầu (LLM + kiểm tra chặt): đọc câu mới cùng vài câu hỏi
            # trước để biết: tóm tắt hay hỏi đáp, phạm vi, Điều cần bỏ, độ dài, câu
            # nối tiếp… Không cần liệt kê từ khóa. Sai/không hợp lệ → plan=None và
            # logic từ khóa phía dưới đảm nhiệm như cũ. Chỉ gọi khi có lịch sử hoặc
            # câu có dấu hiệu tóm tắt (câu hỏi luật độc lập đầu tiên không tốn lượt LLM).
            plan = None
            _prev_user_qs = [c for r, c in history if r == "user"]
            if True:   # luôn để LLM đọc ý định (~1s); không còn bộ lọc từ khóa gõ cứng
                last_state = await _load_last_task_state(body.session_id, current_user, db)
                _name_rows = await db.execute(
                    select(Document.original_filename).where(Document.id.in_(search_doc_ids)).limit(8)
                )
                _doc_names = [n for (n,) in _name_rows.all() if n]
                _raw_plan = await llm_service.analyze_request(
                    _prev_user_qs, body.question,
                    doc_names=_doc_names, last_scope=last_state.get("scope_name") or None,
                )
                plan = _validate_plan(_raw_plan, _prev_user_qs + [body.question], body.question)
                logger.info(f"[planner] raw={_raw_plan} → valid={plan}")
                # Planner lớn hay xếp nhầm câu xin tư vấn ("mua nhà nên chú ý gì") vào "find". Khi chủ đề
                # nằm ngay trong câu mới (không phải câu nối tiếp), hỏi thêm 1 câu phân loại nhỏ để xác nhận.
                if plan and plan["action"] == "find":
                    _qw = set(re.findall(r"\w+", body.question.lower()))
                    if all(w in _qw for w in re.findall(r"\w+", plan["topic"])):
                        _wants_list = await llm_service.confirm_list_request(body.question)
                        logger.info(f"[planner] xác nhận yêu cầu danh sách → {_wants_list}")
                        if _wants_list is False:
                            plan.update({"action": "qa", "topic": "", "also": [], "question": ""})

            # Phủ định ("không tóm tắt file này") -> bỏ phần bị phủ định
            # Tách ràng buộc TRƯỚC ("...mà không cần tóm tắt điều 5") để cụm
            # "không cần tóm tắt" trong ràng buộc không bị hiểu là phủ định cả câu.
            pre_task, pre_constraint = _split_constraints(body.question)
            effective_q, negated = _strip_negated_intents(pre_task)
            if plan:
                negated, pre_constraint, effective_q = False, "", body.question
            if (plan and plan["action"] == "clarify") or (not plan and negated and not _has_real_content(effective_q)):
                yield _sse({"type": "sources", "sources": []})
                yield _sse({"type": "token", "content": _NEGATION_GUIDE})
                chat_session = await _get_or_create_session(body.session_id, current_user, db)
                db.add(ChatMessage(session_id=chat_session.id, role="user", content=body.question))
                assistant_msg = ChatMessage(
                    session_id=chat_session.id, role="assistant", content=_NEGATION_GUIDE,
                    sources=[], retrieval_metadata={"negated_intent": True},
                )
                db.add(assistant_msg)
                await db.flush()
                if not chat_session.title:
                    chat_session.title = body.question[:80]
                await db.commit()
                yield _sse({"type": "done", "session_id": str(chat_session.id), "message_id": str(assistant_msg.id)})
                return

            # ── Case #05: Câu hỏi mơ hồ → yêu cầu làm rõ TRƯỚC khi vào pipeline ──
            # Chỉ kích hoạt khi không có lịch sử hội thoại (câu mới hoàn toàn)
            # và câu quá ngắn/thiếu từ khóa cụ thể. Nếu có lịch sử, _resolve_followup
            # sẽ tự bổ sung ngữ cảnh → không cần hỏi lại.
            if not plan and _is_ambiguous_query(effective_q, has_history=bool(history)):
                _lang = _detect_language(effective_q)
                _clarify_msg = _AMBIGUOUS_CLARIFY_MSG_EN if _lang == "en" else _AMBIGUOUS_CLARIFY_MSG_VI
                yield _sse({"type": "sources", "sources": []})
                yield _sse({"type": "token", "content": _clarify_msg})
                chat_session = await _get_or_create_session(body.session_id, current_user, db)
                db.add(ChatMessage(session_id=chat_session.id, role="user", content=body.question))
                _ambig_msg = ChatMessage(
                    session_id=chat_session.id,
                    role="assistant",
                    content=_clarify_msg,
                    sources=[],
                    retrieval_metadata={"ambiguous_query": True, "detected_lang": _lang},
                )
                db.add(_ambig_msg)
                await db.flush()
                if not chat_session.title:
                    chat_session.title = body.question[:80]
                await db.commit()
                yield _sse({
                    "type": "done",
                    "session_id": str(chat_session.id),
                    "message_id": str(_ambig_msg.id),
                })
                return

            if history and _looks_like_followup(effective_q):
                yield _sse({"type": "status", "stage": "embed", "message": "Đang hiểu câu hỏi theo ngữ cảnh hội thoại trước…"})
            if plan:
                # Planner đã hiểu ngữ cảnh: câu hỏi độc lập lấy từ plan (qa) hoặc giữ nguyên (summary)
                question = plan["question"] if (plan["action"] == "qa" and plan["question"]) else body.question
                followup_mode = "planner" if question != body.question else "none"
            else:
                question, followup_mode = await _resolve_followup(effective_q, history)
            logger.info(f"[history] followup_mode={followup_mode} question={question[:100]!r}")
            # Ràng buộc kèm theo ("mà không cần đề cập điều 5", "ngắn gọn hơn")
            wants_brief = any(k in f"{body.question} {question}".lower() for k in _BRIEF_KEYWORDS)
            question, later_constraint = _split_constraints(question)
            resolved_full = question   # bản đầy đủ (kể cả phần nối tiếp) để đọc cờ chi tiết/ngắn
            tail_constraint = ""
            if followup_mode == "concat" and " — " in question:
                parts = question.split(" — ")
                # Câu chỉnh sửa nối tiếp ("bỏ điều 3") là RÀNG BUỘC, không phải nhiệm vụ
                tail_constraint = " ".join(p for p in parts[1:] if _EXCLUSION_RE.search(p))
                question = parts[0]
            constraint_text = " ".join(x for x in (pre_constraint, later_constraint, tail_constraint) if x)
            excluded_articles = _excluded_articles(constraint_text)
            unhandled_note = _unhandled_constraint(constraint_text)
            if constraint_text:
                logger.info(f"[constraint] task={question!r} excl={excluded_articles} unhandled={unhandled_note!r}")
            summary_intent = (not negated) and _is_full_summary_request(question)
            if summary_intent and not plan:
                summary_intent = await _confirm_summary_intent(question)
            if plan:
                summary_intent = plan["action"] == "summary"
                excluded_articles = set(plan["exclude"])
                unhandled_note = ""
            # ── LLM fallback router: chỉ khi câu KHÔNG khớp keyword nào rõ ràng
            # (không phải RAG về Điều cụ thể, không phải compare) VÀ ngắn/có tín
            # hiệu mơ hồ → gọi LLM phân loại. Tránh gọi cho mọi câu RAG thông
            # thường (tốn 2-5s/lần không cần thiết khi câu đã rõ ý định).
            _maybe_summary_signal = any(
                w in question.lower() for w in (
                    "tóm", "tom", "ý chính", "nội dung chính", "overview",
                    "điểm chính", "tổng", "sơ lược", "rút gọn", "ngắn gọn",
                )
            )
            if (
                not negated
                and not plan
                and not summary_intent
                and not _is_single_article_question(question)
                and not _compare_keywords_match(question)
                and _maybe_summary_signal
            ):
                _llm_intent = await llm_service.classify_query_intent(question)
                if _llm_intent == "SUMMARY":
                    logger.info(f"[intent-llm] classify_query_intent=SUMMARY cho {question[:60]!r}")
                    summary_intent = True
            if followup_mode != "none":
                yield _sse({"type": "status", "stage": "embed", "message": f"Hiểu câu hỏi là: {resolved_full[:150]}"})

            # ── "Chốt chặn" cho yêu cầu so sánh 2 tài liệu: tính năng này CHỈ
            # hoạt động khi người dùng đã chọn ĐÚNG 2 tài liệu cụ thể (qua bộ
            # lọc tài liệu ở sidebar). Trước đây nếu thiếu điều kiện này, hệ
            # thống ÂM THẦM rơi vào RAG tìm-kiếm-toàn-bộ — với 1 câu hỏi kiểu
            # "so sánh 2 file" gần như không có tín hiệu ngữ nghĩa cụ thể để
            # similarity-search bắt trúng gì, nên luôn ra "không tìm thấy",
            # khiến người dùng không hiểu vì sao. Giờ báo rõ lý do thay vì im
            # lặng trả lời sai/không liên quan.
            if _is_compare_request(question) and not (explicit_multi_doc and len(search_doc_ids) == 2):
                selected_count = len(body.document_ids) if body.document_ids else 0
                guide_msg = (
                    "Để so sánh, bạn cần chọn ĐÚNG 2 tài liệu ở mục \"Tài liệu\" bên "
                    f"sidebar trước khi hỏi (hiện đang chọn {selected_count} tài liệu)."
                )
                yield _sse({"type": "sources", "sources": []})
                yield _sse({"type": "token", "content": guide_msg})

                chat_session = await _get_or_create_session(body.session_id, current_user, db)
                db.add(ChatMessage(session_id=chat_session.id, role="user", content=body.question))
                assistant_msg = ChatMessage(
                    session_id=chat_session.id,
                    role="assistant",
                    content=guide_msg,
                    sources=[],
                    retrieval_metadata={"compare_guard": True},
                )
                db.add(assistant_msg)
                await db.flush()
                if not chat_session.title:
                    chat_session.title = body.question[:80]
                await db.commit()

                yield _sse({
                    "type": "done",
                    "session_id": str(chat_session.id),
                    "message_id": str(assistant_msg.id),
                })
                return

            # ── Case #08: câu hỏi nhắc rõ 1 tài liệu cụ thể → tự thu hẹp phạm vi ──
            scope_doc_id = None
            scope_name = ""
            if not body.document_ids:
                # Phạm vi file do LLM (planner) quyết định: "same" = tiếp tục file của lượt trước (đọc từ
                # task_state đã lưu), tên file = file được nhắc, "all" = tìm trên tất cả tài liệu.
                # File được nhắc rõ ràng trong câu vừa gõ luôn được ưu tiên. Code chỉ kiểm tra tính hợp lệ.
                _last_id = None
                try:
                    _last_id = uuid.UUID(str(last_state.get("scope_doc_id"))) if last_state.get("scope_doc_id") else None
                except (ValueError, TypeError):
                    _last_id = None
                if _last_id not in search_doc_ids:
                    _last_id = None
                _sc = plan.get("scope") if plan else None
                detected_id = await _detect_named_document(body.question, search_doc_ids, db)
                if not detected_id:
                    if _sc == "same":
                        detected_id = _last_id
                    elif _sc == "all":
                        detected_id = None
                    elif _sc:
                        detected_id = await _detect_named_document(_sc, search_doc_ids, db)
                    elif _last_id and (
                        _looks_like_followup(body.question)
                        or (bool(plan) and plan["action"] == "summary"
                            and any(k in (_prev_user_qs[-1] if _prev_user_qs else "").lower() for k in _SUMMARY_KEYWORDS))
                    ):
                        # LLM không nêu scope → quy tắc dự phòng: chỉ tiếp tục file cũ khi là nhiệm vụ nối tiếp
                        detected_id = _last_id
                if detected_id:
                    search_doc_ids = [detected_id]
                    scope_doc_id = str(detected_id)
                    _nm = await db.execute(select(Document.original_filename).where(Document.id == detected_id))
                    scope_name = _nm.scalar_one_or_none() or ""
                    logger.info(f"[scope] thu hẹp vào {scope_name} (planner scope={_sc!r})")

            # Giá trị mặc định — các nhánh không dùng RAG bình thường bên dưới
            # không tự set hết các biến này, nhưng retrieval_metadata ở cuối hàm
            # cần đọc chúng.
            candidates: list = []
            ranked: list = []
            embed_text = question
            full_summary_used = False
            _cstat = None   # thống kê mức context (chỉ có ở nhánh RAG thường)
            article_lookup_used = False   # Case mới: tra cứu trực tiếp 1 Điều
            find_topic = plan["topic"] if (plan and plan["action"] == "find") else ""
            find_used = False
            find_hits: list = []
            compare_used = False
            compare_pairs: list = []
            compare_doc_names: tuple[str, str] = ("Tài liệu 1", "Tài liệu 2")
            compare_mode = "both"

            # ── Yêu cầu SO SÁNH 2 tài liệu → ghép cặp Điều tương ứng bằng CODE
            # (theo độ tương đồng tiêu đề điều), sau đó dùng LLM cho MỖI CẶP để
            # viết 1 câu so sánh ngắn — phạm vi rất nhỏ (2 đoạn/lần) nên vẫn giữ
            # được độ chính xác cao, khác hẳn việc bắt LLM so sánh nguyên 2 tài
            # liệu cùng lúc (dễ lẫn lộn, dễ bịa) ─────────────────────────────
            if explicit_multi_doc and len(search_doc_ids) == 2 and _is_compare_request(question):
                yield _sse({
                    "type": "status",
                    "stage": "retrieval",
                    "message": "Đang ghép các Điều tương ứng giữa 2 tài liệu…",
                })
                doc_rows = await db.execute(
                    select(Document.id, Document.original_filename).where(Document.id.in_(search_doc_ids))
                )
                name_map = {str(doc_id): (name or "") for doc_id, name in doc_rows.all()}
                doc_a_id, doc_b_id = str(search_doc_ids[0]), str(search_doc_ids[1])
                doc_name_a = name_map.get(doc_a_id) or "Tài liệu 1"
                doc_name_b = name_map.get(doc_b_id) or "Tài liệu 2"
                compare_doc_names = (doc_name_a, doc_name_b)

                all_groups = await _group_chunks_by_article(search_doc_ids, db)
                groups_by_doc: dict[str, list] = {}
                for g in all_groups:
                    groups_by_doc.setdefault(g.payload.get("document_id"), []).append(g)
                groups_a = [g for g in groups_by_doc.get(doc_a_id, []) if (g.payload.get("article_number") or "").strip()]
                groups_b = [g for g in groups_by_doc.get(doc_b_id, []) if (g.payload.get("article_number") or "").strip()]

                compare_pairs = _pair_articles(groups_a, groups_b)
                context_chunks = [c for pair in compare_pairs for c in pair if c is not None]
                candidates = context_chunks
                ranked = context_chunks
                compare_used = True
                compare_mode = _compare_mode(question)

            # ── Yêu cầu tóm tắt TOÀN BỘ tài liệu → nhóm chunk theo từng Điều
            # (dùng metadata có sẵn), KHÔNG bỏ sót điều nào, và tóm bằng trích
            # xuất trực tiếp — không cần gọi LLM cho việc chỉ đơn thuần rút
            # gọn nội dung, vừa nhanh vừa tuyệt đối chính xác (nguyên văn) ──
            # ── "Có Điều nào đề cập tới X không / liệt kê hết": quét TOÀN BỘ tài liệu
            # bằng code (không phụ thuộc điểm rerank, không bỏ sót), kết quả là danh sách
            # Điều + câu nguyên văn chứa chủ đề — không qua LLM nên không thể bịa ──
            elif find_topic:
                yield _sse({
                    "type": "status",
                    "stage": "retrieval",
                    "message": f"Đang rà soát toàn bộ tài liệu tìm các Điều nhắc tới “{find_topic}”…",
                })
                _all_groups = await _group_chunks_by_article(search_doc_ids, db)
                find_hits = _find_articles_with_topic(_all_groups, find_topic, plan.get("also"))
                context_chunks = [g for g, _c in sorted(find_hits, key=lambda x: -x[1])[:10]]
                candidates = context_chunks
                ranked = context_chunks
                find_used = True

            elif summary_intent:
                # Dùng LLM để hiểu phạm vi tóm tắt: "10 điều đầu", "từ Điều 3 đến 8",
                # "Chương II", "toàn bộ"… thay vì regex cứng chỉ match "Điều X".
                article_range = plan["range"] if plan else await llm_service.extract_article_range(question)
                range_type = article_range.get("type", "all")
                logger.info(f"[summary] extract_article_range: {article_range}")

                range_desc = {
                    "all":       "toàn bộ tài liệu",
                    "first_n":   f"{article_range.get('n', '?')} Điều đầu tiên",
                    "last_n":    f"{article_range.get('n', '?')} Điều cuối",
                    "range":     f"Điều {article_range.get('from','?')} đến Điều {article_range.get('to','?')}",
                    "list":      f"các Điều {article_range.get('list', [])}",
                    "chapter":   f"Chương {article_range.get('chapter', '?')}",
                }.get(range_type, "tài liệu")

                yield _sse({
                    "type": "status",
                    "stage": "retrieval",
                    "message": f"Đang tổng hợp nội dung {range_desc} theo từng Điều…",
                })
                context_chunks = await _group_chunks_by_article(search_doc_ids, db)

                # Áp dụng range filter từ LLM trước khi loại excluded_articles
                context_chunks = _apply_article_range(context_chunks, article_range)

                if excluded_articles:
                    context_chunks = [
                        c for c in context_chunks
                        if str(c.payload.get("article_number") or "") not in excluded_articles
                    ]
                candidates = context_chunks
                ranked = context_chunks
                full_summary_used = True

            # ── Câu hỏi về NỘI DUNG của 1 Điều cụ thể ("Điều 10 quy định gì?") ──
            # Khác với summary pipeline (trích xuất ngắn từng Điều để liệt kê),
            # nhánh này lấy TOÀN BỘ nội dung của Điều đó từ DB rồi cho LLM trả
            # lời — đảm bảo không bỏ sót khoản nào khi Điều trải dài nhiều chunk.
            # Đặt TRƯỚC else (RAG thường) nhưng SAU summary để không tranh chấp
            # với "tóm tắt Điều 8" (đã được _is_full_summary_request bắt trước).
            elif not summary_intent and _is_single_article_question(question):
                m_art = _SINGLE_ARTICLE_RE.search(question)
                art_no = m_art.group(1) if m_art else None
                if art_no:
                    yield _sse({
                        "type": "status",
                        "stage": "retrieval",
                        "message": f"Đang tra cứu toàn bộ nội dung Điều {art_no} từ tài liệu…",
                    })
                    db_chunks = await _fetch_article_chunks_from_db(art_no, search_doc_ids, db)
                    if db_chunks:
                        # Dùng DB chunks là primary (đầy đủ nhất).
                        # Bổ sung thêm kết quả similarity search để LLM có thêm
                        # ngữ cảnh nếu câu hỏi liên quan đến nhiều điều.
                        supp_candidates, supp_ranked = await _retrieve_and_rerank(
                            question, question, search_doc_ids, explicit_multi_doc, body.top_k, current_user.id,
                        )
                        supp_confident = reranker_service.filter_by_threshold(supp_ranked, threshold=settings.RERANK_MIN_SCORE)
                        # Loại bỏ chunk của Điều đang tra cứu khỏi phần bổ sung
                        # (đã có đầy đủ từ DB rồi, không cần thêm bản thứ 2).
                        other_chunks = [
                            c for c in (supp_confident or [])
                            if str(c.payload.get("article_number") or "") != art_no
                        ]
                        context_chunks, _cstat = await _compress_context(db_chunks + other_chunks[:2], question)
                        candidates = supp_candidates
                        ranked = supp_ranked
                        article_lookup_used = True
                    else:
                        # Không tìm thấy Điều này trong DB (chưa có metadata) →
                        # fall through sang RAG thường
                        art_no = None

                if not art_no or not article_lookup_used:
                    # Fallback: RAG thường
                    yield _sse({"type": "status", "stage": "embed", "message": "Đang phân tích câu hỏi…"})
                    yield _sse({"type": "status", "stage": "retrieval", "message": "Đang tìm kiếm đoạn tài liệu liên quan…"})
                    candidates, ranked = await _retrieve_and_rerank(
                        question, question, search_doc_ids, explicit_multi_doc, body.top_k, current_user.id,
                    )
                    ranked_confident = reranker_service.filter_by_threshold(ranked, threshold=settings.RERANK_MIN_SCORE)
                    context_chunks = ranked_confident or (ranked[:3] if _is_meta_document_question(question) else [])
                    context_chunks, _cstat = await _compress_context(context_chunks, question)

            else:
                # ── Retrieval — lượt 1: dùng THẲNG câu hỏi gốc để tìm kiếm ───
                yield _sse({"type": "status", "stage": "embed", "message": "Đang phân tích câu hỏi…"})
                yield _sse({"type": "status", "stage": "retrieval", "message": "Đang tìm kiếm đoạn tài liệu liên quan…"})

                candidates, ranked = await _retrieve_and_rerank(
                    question, question, search_doc_ids, explicit_multi_doc, body.top_k, current_user.id,
                )
                ranked_confident = reranker_service.filter_by_threshold(ranked, threshold=settings.RERANK_MIN_SCORE)

                # ── Case #05 (HyDE), áp dụng THEO ĐỘ TIN CẬY thay vì đếm số từ:
                # không cần đoán trước câu nào "ngắn/mơ hồ" — cứ tìm bình thường
                # trước; CHỈ KHI kết quả không đủ tin cậy (và không phải câu hỏi
                # khái quát về tài liệu, vốn vẫn xử lý riêng bên dưới) mới nhờ LLM
                # viết 1 đoạn văn giả định rồi thử tìm lại lần 2 bằng đoạn đó —
                # vì lúc này nhiều khả năng do câu hỏi quá ngắn/thiếu ngữ cảnh chứ
                # không phải do tài liệu thực sự không có thông tin liên quan.
                _top_score = ranked[0].rerank_score if ranked else 0.0
                if (
                    not ranked_confident
                    and not _is_meta_document_question(question)
                    and _top_score < settings.HYDE_MIN_TOP_SCORE
                ):
                    logger.info(f"[hyde] bỏ qua: điểm cao nhất {_top_score:.4f} quá thấp → ngoài tài liệu")
                if (
                    not ranked_confident
                    and not _is_meta_document_question(question)
                    and _top_score >= settings.HYDE_MIN_TOP_SCORE
                ):
                    yield _sse({
                        "type": "status",
                        "stage": "hyde",
                        "message": "Chưa tìm thấy đoạn nào đủ liên quan — đang diễn giải lại câu hỏi để tìm chính xác hơn…",
                    })
                    hypo = await llm_service.generate_hypothetical_answer(question)
                    if hypo:
                        candidates2, ranked2 = await _retrieve_and_rerank(
                            question, hypo, search_doc_ids, explicit_multi_doc, body.top_k, current_user.id,
                        )
                        ranked2_confident = reranker_service.filter_by_threshold(ranked2, threshold=settings.RERANK_MIN_SCORE)
                        if ranked2_confident:
                            candidates, ranked, ranked_confident = candidates2, ranked2, ranked2_confident
                            embed_text = hypo

                yield _sse({
                    "type": "status",
                    "stage": "rerank",
                    "message": f"Đang xếp hạng {len(candidates)} đoạn tìm được…",
                })

                if ranked_confident:
                    context_chunks = ranked_confident
                elif ranked and _is_meta_document_question(question):
                    # Câu hỏi khái quát về CHÍNH tài liệu (không phải 1 chủ đề pháp lý
                    # cụ thể) → vẫn cho LLM xem top chunk dù điểm rerank thấp, vì bản
                    # thân câu hỏi ít tín hiệu để similarity-search so khớp chính xác.
                    context_chunks = ranked[:3]
                else:
                    # Không đủ tin cậy VÀ không phải câu hỏi khái quát về tài liệu →
                    # nhiều khả năng câu hỏi thuộc chủ đề/lĩnh vực khác không có trong
                    # tài liệu (VD: hỏi luật hôn nhân trên tài liệu luật nhà ở) →
                    # KHÔNG dùng RAG, tránh để LLM tự suy diễn/bịa nội dung.
                    context_chunks = []
                # Case #06: giới hạn ngân sách context trước khi đưa vào LLM.
                context_chunks, _cstat = await _compress_context(context_chunks, question)

            # ── Web search (Tavily): CHỈ khi tài liệu không có gì tin cậy VÀ câu hỏi là
            # câu hỏi PHÁP LUẬT (LLM phân loại). Câu hỏi khác (code, thời tiết…) → không search.
            if _cstat:
                logger.info(f"[context] {_cstat['orig_chars']} ký tự tìm được / ngưỡng {_cstat['budget_chars']} → dùng {_cstat['used_chars']}")
            web_results: list = []
            if (
                not context_chunks
                and tavily_service.is_enabled()
                and not _is_meta_document_question(question)
            ):
                if await llm_service.is_legal_question(question):
                    yield _sse({"type": "status", "stage": "web", "message": "Tài liệu không có thông tin — đang tra cứu nguồn pháp luật trên web…"})
                    web_results = await tavily_service.search_legal(question)
                    if web_results:
                        # Lọc LẠC ĐỀ theo từng CÂU: chấm điểm câu web với câu hỏi bằng reranker, chỉ giữ câu liên quan;
                        # nguồn không có câu nào đủ điểm bị loại; tối đa 3 nguồn tốt nhất.
                        import dataclasses as _dc
                        _units = [(i, ln) for i, r in enumerate(web_results) for ln in r.content.splitlines() if ln.strip()][:80]
                        _sc = await _run_sync(reranker_service.score_texts, question, [ln for _, ln in _units])
                        _per: dict[int, list] = {}
                        for (i, ln), s_ in zip(_units, _sc):
                            _per.setdefault(i, []).append((s_, ln))
                        _kept = []
                        for i, r in enumerate(web_results):
                            items = _per.get(i, [])
                            best = max((s_ for s_, _ in items), default=0.0)
                            logger.info(f"[web_search] nguồn {i + 1}: điểm câu tốt nhất {best:.2f}")
                            if best < settings.WEB_MIN_RELEVANCE:
                                continue
                            top = sorted((x for x in items if x[0] >= settings.WEB_SENTENCE_MIN), key=lambda x: x[0], reverse=True)[:5]
                            keep_set = {ln for _, ln in top}
                            lines = [ln for ln in r.content.splitlines() if ln in keep_set]   # giữ thứ tự gốc
                            if lines:
                                _kept.append((best, _dc.replace(r, content="\n".join(lines))))
                        _kept.sort(key=lambda x: x[0], reverse=True)
                        web_results = [r for _, r in _kept[:3]]
                else:
                    logger.info("[web_search] bỏ qua: không phải câu hỏi pháp luật")

            sources = _sources_from_ranked(context_chunks)
            # Nguồn web chỉ gửi qua SSE (không lưu vào DB vì schema CitationSource yêu cầu document_id UUID);
            # link nguồn đã nằm trong nội dung câu trả lời.
            _sse_sources = sources if not web_results else [
                {"document_id": "", "document_name": f"Nguồn web: {r.title}", "content": r.content[:300]}
                for r in web_results
            ]
            yield _sse({"type": "sources", "sources": _sse_sources})

            doc_names = ", ".join(sorted({s["document_name"] for s in sources if s["document_name"]})) or "tài liệu"
            yield _sse({
                "type": "status",
                "stage": "generate",
                "message": f"Đang đọc {doc_names} và soạn câu trả lời…",
            })

            if find_used:
                if not find_hits:
                    full_answer = f"Tôi không tìm thấy Điều nào nhắc tới “{find_topic}” trong tài liệu."
                    yield _sse({"type": "token", "content": full_answer})
                else:
                    _by_doc: dict[str, list] = {}
                    for _g, _c in find_hits:
                        _by_doc.setdefault(str(_g.payload.get("original_filename") or "tài liệu"), []).append((_g, _c))
                    _parts: list[str] = []
                    _find_len = (plan or {}).get("length")
                    _brief_find = _find_len in ("brief", "titles")
                    for _fname, _items in _by_doc.items():
                        _total_items = len(_items)
                        if _brief_find and _total_items > 8:
                            # Rút gọn: giữ 8 Điều khớp nhiều nhất (theo điểm), xếp lại theo số Điều
                            _items = sorted(sorted(_items, key=lambda x: -x[1])[:8],
                                            key=lambda x: int(re.match(r"\d+", str(x[0].payload.get("article_number") or "0")).group() if re.match(r"\d+", str(x[0].payload.get("article_number") or "0")) else 0))
                        # Dòng trống giữa các tài liệu — nếu không, markdown dính tiêu đề file sau vào dòng cuối của file trước
                        _head = ("\n" if _parts else "") + f"**{_fname}** — {_total_items} Điều có nhắc tới “{find_topic}”" + (f" (nêu {len(_items)} Điều chính)" if len(_items) < _total_items else "") + ":\n\n"
                        _parts.append(_head)
                        yield _sse({"type": "token", "content": _head})
                        for _g, _c in _items:
                            _no = _g.payload.get("article_number")
                            _ttl = _clean_article_title(str(_g.payload.get("article_title") or ""))
                            _extra = [m for m in (_g.payload.get("matched") or []) if m != find_topic]
                            _line = f"- Điều {_no}" + (f". {_ttl}" if _ttl else "")
                            if _extra:
                                _line += f" (khớp thêm: {', '.join(_extra)})"
                            _line += "\n"
                            _parts.append(_line)
                            yield _sse({"type": "token", "content": _line})
                        if len(_items) < _total_items:
                            _more = f"- … và {_total_items - len(_items)} Điều khác (yêu cầu “liệt kê đầy đủ” để xem hết)\n"
                            _parts.append(_more)
                            yield _sse({"type": "token", "content": _more})
                    full_answer = "".join(_parts)

            elif full_summary_used:
                # ── Tóm tắt EXTRACTIVE (trích xuất trực tiếp, KHÔNG gọi LLM) ────
                # Chỉ đơn thuần rút gọn nội dung mỗi Điều xuống còn 1-2 câu đầu —
                # không cần "suy nghĩ" gì thêm, nên không tốn thời gian gọi model,
                # và vì là nguyên văn nên tuyệt đối không có rủi ro bịa đặt.
                bullets: list[str] = []
                total = len(context_chunks)
                last_chapter = None
                # Người dùng muốn bản chi tiết hơn (thường là câu nối tiếp sau 1
                # lần tóm tắt) → trích mỗi Điều dài hơn, vẫn nguyên văn, không LLM.
                wants_detail = any(k in f"{body.question} {resolved_full}".lower() for k in ("chi tiết", "cụ thể", "đầy đủ hơn", "kỹ hơn"))
                wants_min = any(k in f"{body.question} {resolved_full}".lower() for k in _EXTREME_BRIEF_KEYWORDS)
                # Mức chi tiết (đều trích NGUYÊN VĂN, cắt theo câu trọn vẹn):
                #   titles (ngắn nhất) → chỉ tiêu đề Điều; brief → tối đa 1 câu ngắn;
                #   normal → vài câu ~220 ký tự; detail → ~600 ký tự.
                level = "detail" if wants_detail else ("titles" if wants_min else ("brief" if wants_brief else "normal"))
                if plan and plan["length"]:
                    level = plan["length"]
                _LEVEL_LIMITS = {"titles": (0, 0), "brief": (170, 170), "normal": (260, 450), "detail": (650, 900)}
                snippet_chars = _LEVEL_LIMITS[level][0]
                for i, chunk in enumerate(context_chunks, start=1):
                    content = chunk.content.strip()
                    if not content:
                        continue

                    chapter = chunk.payload.get("chapter") or ""
                    article_number = chunk.payload.get("article_number") or ""
                    article_title = chunk.payload.get("article_title") or ""

                    if chapter and chapter != last_chapter:
                        header = f"\n**{chapter}**\n"
                        bullets.append(header)
                        yield _sse({"type": "token", "content": header})
                        last_chapter = chapter

                    # Nội dung quá ngắn (VD: chỉ là tiêu đề chương/điều, chưa có
                    # nội dung cụ thể) → hiển thị nguyên văn, không cần rút gọn.
                    clean_title = _clean_article_title(article_title)
                    if article_number:
                        body_text = _strip_heading(content, article_title)
                        if level == "titles":
                            snippet = ""
                        else:
                            # Trích NGUYÊN VĂN (extractive) — nhận diện cấu trúc
                            # khoản (1. 2. 3.) để liệt kê gọn từng khoản thay vì
                            # cắt giữa chừng và để lại "2." lơ lửng ở cuối.
                            snippet = _summary_snippet(body_text, level)
                    else:
                        # Đoạn không thuộc Điều nào (phần mở đầu…): trích nguyên văn
                        snippet = "" if level == "titles" else _extractive_snippet(content, max_chars=max(snippet_chars, 220))

                    label = f"Điều {article_number}" if article_number else f"Phần {i}"
                    if clean_title:
                        label += f" — {clean_title}"
                    # Nếu snippet nhiều dòng (có khoản) → tiêu đề Điều trên dòng
                    # riêng, các khoản thụt vào bên dưới. Giúp dễ đọc hơn.
                    if snippet and "\n" in snippet:
                        line = f"**{label}**\n{snippet}\n\n"
                    elif snippet:
                        line = f"**{label}**: {snippet}\n\n"
                    else:
                        line = f"**{label}**\n\n"
                    bullets.append(line)
                    yield _sse({"type": "token", "content": line})

                    if i % 10 == 0 or i == total:
                        yield _sse({
                            "type": "status",
                            "stage": "generate",
                            "message": f"Đang tổng hợp… ({i}/{total})",
                        })

                notes = []
                if excluded_articles:
                    notes.append("đã bỏ qua Điều " + ", ".join(sorted(excluded_articles, key=int)))
                if unhandled_note:
                    notes.append(f"không áp dụng được yêu cầu \"{unhandled_note[:80]}\" vì bản tóm tắt là trích nguyên văn từng Điều")
                if notes:
                    note_line = "\n*(Lưu ý: " + "; ".join(notes) + ".)*\n"
                    bullets.append(note_line)
                    yield _sse({"type": "token", "content": note_line})

                full_answer = "".join(bullets) or "Tôi không tìm thấy thông tin liên quan trong tài liệu."
                if not bullets:
                    yield _sse({"type": "token", "content": full_answer})

            elif compare_used:
                # ── So sánh từng cặp Điều — CHỈ hiện đúng loại người dùng hỏi
                # (chỉ khác nhau / chỉ giống nhau / cả hai), thay vì luôn liệt
                # kê tất cả — và bỏ qua các cặp không thuộc loại được hỏi mà
                # KHÔNG cần gọi LLM cho chúng (tiết kiệm thời gian) ──────────
                doc_name_a, doc_name_b = compare_doc_names
                bullets = []
                total = len(compare_pairs)
                for i, (chunk_a, chunk_b) in enumerate(compare_pairs, start=1):
                    line = ""
                    if chunk_a is not None and chunk_b is not None:
                        label_a = _article_label(chunk_a)
                        label_b = _article_label(chunk_b)
                        similarity = _content_similarity(chunk_a.content, chunk_b.content)
                        is_same = similarity >= _CONTENT_SIMILARITY_THRESHOLD

                        if is_same:
                            if compare_mode in ("same", "both"):
                                # Nội dung 2 bên đã rất giống nhau (thường do 2
                                # tài liệu cùng 1 nguồn) → không cần AI "bịa" ra
                                # khác biệt không có thật, ghi thẳng bằng code.
                                line = f"**{label_a}** ↔ **{label_b}**: Nội dung tương tự nhau, không có khác biệt đáng kể.\n"
                            # is_same nhưng người dùng chỉ hỏi "khác nhau" → bỏ qua hẳn, không gọi LLM.
                        else:
                            if compare_mode in ("diff", "both"):
                                raw_comparison = await llm_service.compare_articles(
                                    f"{doc_name_a} - {label_a}", chunk_a.content.strip(),
                                    f"{doc_name_b} - {label_b}", chunk_b.content.strip(),
                                )
                                comparison = _sanitize_comparison(raw_comparison) or (
                                    "Nội dung có khác biệt — xem chi tiết ở nguồn trích dẫn bên dưới."
                                )
                                line = f"**{label_a}** ↔ **{label_b}**: {comparison}\n"
                            # khác nhau nhưng người dùng chỉ hỏi "giống nhau" → bỏ qua, không gọi LLM.
                    elif compare_mode in ("diff", "both"):
                        # Điều chỉ có ở 1 trong 2 tài liệu — bản thân đây CŨNG
                        # LÀ 1 điểm khác biệt (khác biệt về cấu trúc), nên chỉ
                        # hiện khi người dùng có hỏi tới khía cạnh khác nhau.
                        only_chunk = chunk_a if chunk_a is not None else chunk_b
                        only_doc = doc_name_a if chunk_a is not None else doc_name_b
                        line = f"{_article_label(only_chunk)}: chỉ có trong *{only_doc}*.\n"

                    if line:
                        line = f"- {line}"
                        bullets.append(line)
                        yield _sse({"type": "token", "content": line})

                    if i % 5 == 0 or i == total:
                        yield _sse({
                            "type": "status",
                            "stage": "generate",
                            "message": f"Đang so sánh… ({i}/{total})",
                        })

                if not bullets:
                    fallback_label = {
                        "diff": "khác biệt đáng kể nào",
                        "same": "điểm giống nhau đáng kể nào",
                        "both": "nội dung nào để so sánh",
                    }[compare_mode]
                    full_answer = f"Không tìm thấy {fallback_label} giữa 2 tài liệu."
                    yield _sse({"type": "token", "content": full_answer})
                else:
                    full_answer = "".join(bullets)

            else:
                # ── Sinh câu trả lời (streaming) — luôn dùng câu hỏi GỐC của người
                # dùng (không dùng đoạn văn HyDE giả định) để câu trả lời bám sát
                # đúng ý người hỏi, HyDE chỉ dùng để tìm kiếm chính xác hơn ─────
                full_answer = ""
                _q_lang = _detect_language(body.question)   # Case #07
                if web_results:
                    # KHÔNG dùng LLM viết lại nội dung web (model nhỏ dễ trộn số liệu / bịa nguồn):
                    # hiển thị NGUYÊN VĂN đoạn trích của từng trang kèm link để người dùng tự đối chiếu.
                    _parts = [
                        "⚠️ Tài liệu bạn tải lên không có thông tin này. Dưới đây là **nguyên văn đoạn trích** "
                        "từ các trang web pháp luật tìm được (chưa qua tóm tắt, cần đối chiếu văn bản gốc; "
                        "không phải tư vấn pháp lý):\n"
                    ]
                    for _i, _r in enumerate(web_results, 1):
                        _host = re.sub(r"^https?://(www\.)?", "", _r.url).split("/")[0]
                        _bul = "\n".join(f"- {ln}" for ln in _r.content.splitlines() if ln.strip())
                        _ttl = _r.title.replace("[", "(").replace("]", ")")
                        _parts.append(f"\n**Nguồn {_i} — {_host}**\n\n[{_ttl}]({_r.url})\n\n{_bul}\n")
                    _web_text = "".join(_parts)
                    full_answer += _web_text
                    yield _sse({"type": "token", "content": _web_text})
                else:
                    async for token in llm_service.answer_stream(
                        question, context_chunks, lang=_q_lang,
                        history=history, original_question=body.question,
                        memory_summary=_mem_prev["summary"],
                    ):
                        full_answer += token
                        yield _sse({"type": "token", "content": token})

                if not full_answer:
                    full_answer = "Tôi không tìm thấy thông tin liên quan trong tài liệu."

            # ── Lưu lịch sử chat ───────────────────────────────────────
            chat_session = await _get_or_create_session(body.session_id, current_user, db)
            _mem_new, _mem_compacted = await _advance_memory(
                chat_session.id if body.session_id else None, current_user, db,
                _mem_prev, body.question, full_answer, new_topic=(followup_mode == "none"),
            )
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
                    "hyde_used": embed_text != question,
                    "multi_doc": explicit_multi_doc,
                    "full_summary": full_summary_used,
                    "article_lookup": article_lookup_used,
                    "compare_used": compare_used,
                    "followup_mode": followup_mode,
                    "resolved_question": question if followup_mode != "none" else None,
                    "memory": _mem_new,
                    "task_state": {
                        "scope_doc_id": scope_doc_id,
                        "scope_name": scope_name,
                        "action": (plan or {}).get("action") if plan else None,
                    },
                },
            )
            db.add(assistant_msg)
            await db.flush()
            if not chat_session.title:
                chat_session.title = body.question[:80]
            await db.commit()

            yield _sse({
                "type": "memory",
                "percent": min(100, round(_mem_new["used_chars"] * 100 / max(1000, settings.MEMORY_BUDGET_CHARS))),
                "compacted": _mem_compacted,
                "compactions": _mem_new["compactions"],
                "used_chars": _mem_new["used_chars"],
                "budget_chars": settings.MEMORY_BUDGET_CHARS,
            })
            yield _sse({
                "type": "done",
                "session_id": str(chat_session.id),
                "message_id": str(assistant_msg.id),
            })

        except Exception as exc:
            logger.error(f"[chat.query_stream] Lỗi: {exc}")
            await db.rollback()
            yield _sse({"type": "error", "message": str(exc)})
        finally:
            await db.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Session & History ─────────────────────────────────────────────────────────

@router.get("/sessions", response_model=list[ChatSessionRead])
async def list_sessions(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Lấy danh sách phiên chat của user (mới nhất trước, mặc định 50)."""
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .limit(max(1, min(limit, 200)))
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
    _mem = await _load_last_memory(session_id, current_user, db)
    _pct = min(100, round(_mem["used_chars"] * 100 / max(1000, settings.MEMORY_BUDGET_CHARS)))
    return ChatHistoryResponse(session=session, messages=messages, memory_percent=_pct)


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
