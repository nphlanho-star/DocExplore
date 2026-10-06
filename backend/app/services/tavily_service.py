"""Tìm kiếm web bổ sung qua Tavily — CHỈ dùng cho câu hỏi về pháp luật mà tài liệu
đã tải lên không có thông tin. Mọi lỗi đều được nuốt (trả [] ) để không phá luồng RAG."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import httpx
from loguru import logger

from app.config import get_settings

_URL = "https://api.tavily.com/search"
_INJECT_HINT = re.compile(
    r"(ignore (?:(?:all|any|the|previous|prior|above|your)\s+)*instructions|disregard .{0,30}instructions|bỏ qua (?:(?:mọi|các|tất cả|toàn bộ)\s+)?(?:hướng dẫn|chỉ dẫn|lệnh)|system prompt|you are now)", re.I
)


@dataclass
class WebResult:
    title: str
    url: str
    content: str


# Câu "rác" của trang web (đăng nhập, liên hệ, chú thích ảnh…) — không phải nội dung pháp luật.
_JUNK = re.compile(
    r"(vui lòng đăng nhập|đăng nhập để|đăng ký làm thành viên|đăng ký để sử dụng|tiện ích này|"
    r"điện thoại di động|số điện thoại|chịu trách nhiệm chính|nhập thêm số|hình từ internet|ảnh minh họa|"
    r"xem thêm|bài viết liên quan|theo dõi chúng tôi|chia sẻ bài viết)",
    re.I,
)
_SENT_SPLIT = re.compile(r"(?<=[.;:!?])\s+(?=[A-ZÀ-Ỵ0-9(“\"\-+•])|\n+|\s*\[\.\.\.\]\s*")


# ── Sửa lỗi mất chữ "â" do nguồn/Tavily (VD "hôn nhn", "quan tm", "xy dựng", "đy") ──────────────
# Chỉ sửa các từ KHÔNG THỂ là âm tiết tiếng Việt hợp lệ (không có nguyên âm, hoặc đ/x/c/d/g + "y"),
# nên không thể làm sai từ đúng. Các trường hợp mơ hồ (quân→quan, cân→can…) không đoán, giữ nguyên.
_INITIALS = ["ngh", "ng", "nh", "ch", "gh", "kh", "ph", "th", "tr", "qu", "gi",
             "đ", "b", "c", "d", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x"]
_FINALS = {"c", "ch", "m", "n", "ng", "nh", "p", "t"}
_NOT_WORDS = {"cm", "km", "mm", "dm", "nm", "pm", "gm", "tp", "cp", "tt", "tn", "tc", "tk"}   # đơn vị / viết tắt
_TOKEN = re.compile(r"(?<![\w./@:#&-])([A-Za-zÀ-ỹĐđ]{2,})(?![\w./@&-])")


def _has_vowel(s: str) -> bool:
    base = "".join(ch for ch in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(ch) != "Mn")
    return any(ch in "aeiouy" for ch in base)


def _fix_token(m: "re.Match") -> str:
    tok = m.group(1)
    low = tok.lower()
    if (tok.isupper() and len(tok) > 1) or low in _NOT_WORDS:
        return tok
    if not _has_vowel(low):
        for ini in sorted(_INITIALS, key=len, reverse=True):
            if low.startswith(ini) and low[len(ini):] in _FINALS:
                return tok[: len(ini)] + "â" + tok[len(ini):]
        return tok
    if len(low) == 2 and low[0] in ("đ", "x", "c", "d", "g") and low[1] == "y":
        return tok[0] + "â" + tok[1]
    return tok


def _repair_missing_circumflex_a(text: str) -> str:
    return _TOKEN.sub(_fix_token, text)


def _norm_key(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def _looks_like_nav(s: str) -> bool:
    """Dòng menu/điều hướng của trang (VD 'BỘ CÔNG AN BỘ CÔNG AN BỘ CÔNG THƯƠNG…'): toàn chữ HOA hoặc lặp từ."""
    words = s.split()
    if len(words) < 8:
        return False
    if sum(w.isupper() for w in words) / len(words) > 0.6:
        return True
    return len({w.lower() for w in words}) / len(words) < 0.6


def _tidy(text: str, limit: int, title: str = "") -> str:
    """Làm sạch đoạn trích: bỏ 'Title:', ký hiệu markdown/bảng, câu rác của trang web, câu trùng lặp,
    tách thành từng câu (mỗi dòng 1 câu) và cắt ở hết câu — không để câu đứt giữa chừng."""
    text = unicodedata.normalize("NFC", text)   # gộp ký tự tổ hợp → dấu tiếng Việt hiển thị đúng
    text = _repair_missing_circumflex_a(text)
    text = re.sub(r"#{1,6}\s*", "\n", text)               # tiêu đề markdown giữa dòng → xuống dòng
    text = re.sub(r"([?!.;:])\s*\.(?=\s|$)", r"\1", text)   # "?." ".." ";." ":." → bỏ dấu chấm thừa
    text = re.sub(r"\s*…\s*$", "", text.strip())
    title_key = _norm_key(title)
    sents, seen, used = [], set(), 0
    for raw in _SENT_SPLIT.split(text):
        s = (raw or "").strip()
        s = re.sub(r"^(title:\s*)", "", s, flags=re.I)
        s = re.sub(r"^[#>*\-•+|\s]+", "", s)            # ký hiệu markdown / bảng ở đầu câu
        s = re.sub(r"[|]+", " ", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        s = re.sub(r"([.;:])\s*[.;:]+$", r"\1", s)     # ".;" "..", ":." ở cuối
        if len(s) < 25 or _JUNK.search(s):
            continue
        if _looks_like_nav(s):
            continue
        if s.endswith("?"):                      # câu hỏi = tiêu đề/FAQ của trang, không phải nội dung luật
            continue
        if s.endswith(":") and not re.search(r"(Điều|Nghị định|Luật|Bộ luật|Thông tư|khoản)", s):
            continue                              # "…như sau:" mà không dẫn chiếu văn bản → vô nghĩa khi danh sách bị cắt
        key = _norm_key(s)
        if key in seen or (title_key and (key == title_key or key in title_key)):
            continue
        seen.add(key)
        if used + len(s) > limit:
            break
        sents.append(s)
        used += len(s)
    # Tavily hay chồng đoạn: câu bị cắt dở rồi câu sau lặp lại đoạn đuôi của nó → ghép lại thành 1 câu đầy đủ
    cleaned: list[str] = []
    for s in sents:
        if cleaned and not cleaned[-1].endswith((".", ";", ":", "!", "?", "…", "”", '"')):
            prev = cleaned[-1]
            for k in range(min(len(prev), len(s)), 14, -1):
                if prev.lower().endswith(s[:k].lower()):
                    cleaned[-1] = prev + s[k:]
                    break
            else:
                cleaned.append(s)
            continue
        cleaned.append(s)
    sents = cleaned
    # Câu cuối kết thúc bằng ":" (danh sách đã bị cắt mất) → thêm dấu "…" để người đọc biết còn tiếp
    if sents and sents[-1].endswith(":"):
        sents[-1] += " (nội dung chi tiết xem ở link nguồn)"
    return "\n".join(sents)


def is_enabled() -> bool:
    s = get_settings()
    return bool(s.WEB_SEARCH_ENABLED and (s.TAVILY_API_KEY or "").strip())


async def search_legal(query: str) -> list[WebResult]:
    s = get_settings()
    if not is_enabled():
        return []
    domains = [d.strip() for d in s.WEB_SEARCH_DOMAINS.split(",") if d.strip()]
    payload = {
        "api_key": s.TAVILY_API_KEY.strip(),
        "query": query[:380],
        "search_depth": s.WEB_SEARCH_DEPTH,
        "max_results": s.WEB_SEARCH_MAX_RESULTS,
        "include_answer": False,
    }
    if domains:
        payload["include_domains"] = domains
    try:
        async with httpx.AsyncClient(timeout=s.WEB_SEARCH_TIMEOUT) as client:
            resp = await client.post(_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning(f"[web_search] Tavily lỗi: {exc}")
        return []
    out: list[WebResult] = []
    for r in data.get("results", []):
        content = (r.get("content") or "").strip()
        url = (r.get("url") or "").strip()
        if not content or not url or _INJECT_HINT.search(content):
            continue
        title = (r.get("title") or url)[:150]
        cleaned = _tidy(content, s.WEB_SEARCH_SNIPPET_CHARS, title)
        if cleaned:
            out.append(WebResult(title, url, cleaned))
    logger.info(f"[web_search] {len(out)} kết quả cho: {query[:80]!r}")
    return out
