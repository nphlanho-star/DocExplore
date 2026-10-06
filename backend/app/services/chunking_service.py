"""
services/chunking_service.py — Chia tài liệu thành các chunk.

Chiến lược (ưu tiên theo thứ tự):
  1. Cấu trúc văn bản luật (Chương → Điều → Khoản) — nếu phát hiện được
     mẫu "Điều X." trong tài liệu. Mỗi Điều là một đơn vị chunk, không
     cắt ngang ranh giới Điều. Điều quá dài mới cắt tiếp theo Khoản,
     rồi mới đến SentenceSplitter nếu Khoản vẫn quá dài.
  2. Section theo heading Markdown (nếu Docling xuất được heading `#`).
  3. Cắt toàn văn bản theo câu (SentenceSplitter, fallback cuối cùng).
"""
import re
import uuid
from dataclasses import dataclass, field

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SentenceSplitter
from loguru import logger

from app.config import get_settings

settings = get_settings()

# ── Regex nhận diện cấu trúc văn bản luật (tiếng Việt) ─────────────────────
# Khớp ở ĐẦU MỘT ĐOẠN VĂN (paragraph, sau khi tách theo dòng trống) — không
# khớp khi "Điều X" chỉ được nhắc tới giữa câu (ví dụ "...theo Điều 5 của
# luật này"), vì trường hợp đó không nằm ở đầu đoạn.
CHUONG_RE = re.compile(r"^Chương\s+([IVXLCDM\d]+)[\.:\)]?\s*(.*)$", re.IGNORECASE)
DIEU_RE = re.compile(r"^Điều\s+(\d+)[\.:]?\s*(.*)$", re.IGNORECASE)
KHOAN_RE = re.compile(r"^(\d{1,2})[\.\)]\s+(.*)$")

# ── Regex nhận diện loại văn bản từ dòng tiêu đề (preamble) ─────────────────
# Dùng để gắn metadata "van_ban_type" vào mỗi chunk — giúp filter theo loại
# khi người dùng hỏi "theo Nghị định nào" hay "Luật nào quy định".
_VAN_BAN_TYPE_RE = re.compile(
    r"\b(Luật|Bộ\s+luật|Nghị\s+định|Thông\s+tư|Quyết\s+định|Pháp\s+lệnh"
    r"|Nghị\s+quyết|Chỉ\s+thị|Công\s+văn|Thông\s+báo)\b",
    re.IGNORECASE,
)
# Năm ban hành từ dòng tiêu đề (VD: "năm 2023", "/2023/", "2023")
_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")


def _detect_van_ban_type(preamble_text: str) -> str | None:
    """Trích loại văn bản từ đoạn mở đầu (tối đa 500 ký tự đầu)."""
    sample = preamble_text[:500]
    m = _VAN_BAN_TYPE_RE.search(sample)
    if m:
        return " ".join(m.group(0).split())  # chuẩn hóa khoảng trắng
    return None


def _detect_year(preamble_text: str) -> str | None:
    """Trích năm ban hành từ đoạn mở đầu."""
    sample = preamble_text[:500]
    m = _YEAR_RE.search(sample)
    return m.group(0) if m else None


def _build_chunk_header(
    chapter: str | None,
    article_number: int | None,
    article_title: str,
) -> str:
    """
    Tạo header ngắn gọn gắn vào đầu mỗi chunk để embedding "biết" chunk này
    thuộc Điều nào, Chương nào — giúp retrieval tốt hơn khi câu hỏi nhắc tên
    Điều/Chương mà không có trong nội dung câu đó.

    Ví dụ: "[Chương II] Điều 8 — Điều kiện được công nhận quyền sở hữu nhà ở"
    """
    parts: list[str] = []
    if chapter:
        # Lấy tên/số chương, bỏ phần mô tả dài
        chapter_short = chapter.split("\n")[0].strip()
        parts.append(f"[{chapter_short}]")
    if article_number is not None:
        # Bỏ phần "Điều X." hoặc "Điều X:" khỏi article_title để không lặp
        clean_title = re.sub(
            r"^Điều\s*\d+[a-zA-Z]?\s*[\.:\(]?\s*", "", article_title, flags=re.IGNORECASE
        ).strip()
        # Bỏ phần "(Tên điều)" nếu có dạng "Điều 8 (Tiêu đề):"
        clean_title = re.sub(r"^[\(\[](.*?)[\)\]]$", r"\1", clean_title).strip()
        if clean_title:
            parts.append(f"Điều {article_number} — {clean_title}")
        else:
            parts.append(f"Điều {article_number}")
    return " ".join(parts)


@dataclass
class ChunkData:
    chunk_index: int
    content: str
    page_number: int | None = None
    metadata: dict = field(default_factory=dict)


class ChunkingService:
    def __init__(self) -> None:
        self._splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            paragraph_separator="\n\n",
        )

    # ── Entry point chính — dùng trong pipeline ─────────────────────────

    def chunk_document(
        self,
        markdown_text: str,
        plain_text: str,
        document_id: uuid.UUID,
        extra_metadata: dict | None = None,
    ) -> list[ChunkData]:
        """
        Chọn chiến lược chunk phù hợp nhất, theo thứ tự ưu tiên đã mô tả
        ở đầu file. Luôn trả về danh sách chunk (có thể rỗng nếu cả
        markdown_text lẫn plain_text đều trống).
        """
        source_text = markdown_text or plain_text

        legal_chunks = self.chunk_by_legal_structure(source_text, document_id, extra_metadata)
        if legal_chunks:
            return legal_chunks

        if markdown_text:
            md_chunks = self.chunk_by_markdown_sections(markdown_text, document_id, extra_metadata)
            if md_chunks:
                return md_chunks

        return self.chunk_text(plain_text, document_id, extra_metadata)

    # ── Chiến lược 1: cấu trúc luật (Chương/Điều/Khoản) ─────────────────

    def chunk_by_legal_structure(
        self,
        text: str,
        document_id: uuid.UUID,
        extra_metadata: dict | None = None,
    ) -> list[ChunkData] | None:
        """
        Chunk theo cấu trúc văn bản luật.

        Trả về None nếu phát hiện ít hơn 2 "Điều" trong tài liệu — khi đó
        không đủ tin cậy để coi là văn bản luật có cấu trúc rõ ràng, caller
        nên fallback sang chiến lược khác.
        """
        if not text or not text.strip():
            return None

        paragraphs = self._split_paragraphs(text)

        # ── Gom nhóm theo Điều ───────────────────────────────────────
        articles: list[dict] = []
        preamble: list[str] = []
        current_chapter: str | None = None
        current_article: dict | None = None

        for para in paragraphs:
            # Chỉ khớp regex trên DÒNG ĐẦU TIÊN của đoạn — nếu khớp cả đoạn
            # nhiều dòng, dấu $ sẽ không dừng đúng chỗ khi có nội dung theo
            # sau trên dòng thứ 2 trở đi (ví dụ "Điều 1. Tiêu đề\nNội dung...").
            first_line = para.split("\n", 1)[0].strip()

            if CHUONG_RE.match(first_line):
                current_chapter = para.strip()
                continue

            dieu_match = DIEU_RE.match(first_line)
            if dieu_match:
                if current_article:
                    articles.append(current_article)
                current_article = {
                    "chapter": current_chapter,
                    "article_number": int(dieu_match.group(1)),
                    "article_title": first_line,
                    "paragraphs": [para],
                }
                continue

            if current_article:
                current_article["paragraphs"].append(para)
            else:
                preamble.append(para)

        if current_article:
            articles.append(current_article)

        if len(articles) < 2:
            # Không đủ dấu hiệu là văn bản luật có cấu trúc Điều rõ ràng.
            return None

        logger.info(f"Document {document_id}: phát hiện cấu trúc luật — {len(articles)} Điều.")

        # ── Detect loại văn bản + năm từ preamble ───────────────────
        preamble_text = "\n\n".join(preamble)
        van_ban_type = _detect_van_ban_type(preamble_text)
        van_ban_year = _detect_year(preamble_text)

        chunks: list[ChunkData] = []
        global_idx = 0
        threshold_chars = settings.CHUNK_SIZE * 4  # ước lượng ~4 ký tự / token

        # Đoạn mở đầu (căn cứ pháp lý, tiêu đề…) trước Điều 1 — vẫn giữ để tìm kiếm được.
        if preamble:
            meta: dict = {"chapter": None, "section": "preamble"}
            if van_ban_type:
                meta["van_ban_type"] = van_ban_type
            if van_ban_year:
                meta["van_ban_year"] = van_ban_year
            if extra_metadata:
                meta.update(extra_metadata)
            chunks.append(
                ChunkData(chunk_index=global_idx, content=preamble_text, metadata=meta)
            )
            global_idx += 1

        for art in articles:
            full_text = "\n\n".join(art["paragraphs"])
            base_meta: dict = {
                "chapter": art["chapter"],
                "article_number": art["article_number"],
                "article_title": art["article_title"],
            }
            if van_ban_type:
                base_meta["van_ban_type"] = van_ban_type
            if van_ban_year:
                base_meta["van_ban_year"] = van_ban_year
            if extra_metadata:
                base_meta.update(extra_metadata)

            # ── Tạo context header để inject vào đầu chunk ──────────
            # Header giúp embedding model "biết" chunk thuộc Điều/Chương nào,
            # tăng khả năng match khi câu hỏi nhắc tên Điều mà không có trong nội dung.
            header = _build_chunk_header(
                chapter=art["chapter"],
                article_number=art["article_number"],
                article_title=art["article_title"],
            )

            def _with_header(text: str) -> str:
                return f"{header}\n{text}" if header else text

            if len(full_text) <= threshold_chars:
                chunks.append(ChunkData(
                    chunk_index=global_idx,
                    content=_with_header(full_text),
                    metadata=base_meta,
                ))
                global_idx += 1
                continue

            # Điều quá dài → cắt theo Khoản trước khi cắt theo câu.
            for unit_text, khoan_no in self._split_by_khoan(art["paragraphs"]):
                unit_meta = {**base_meta, "khoan": khoan_no}
                if len(unit_text) <= threshold_chars:
                    chunks.append(ChunkData(
                        chunk_index=global_idx,
                        content=_with_header(unit_text),
                        metadata=unit_meta,
                    ))
                    global_idx += 1
                else:
                    # Khoản vẫn quá dài → cắt tiếp bằng SentenceSplitter.
                    sub_chunks = self.chunk_text(unit_text, document_id, unit_meta)
                    for j, sub in enumerate(sub_chunks):
                        sub.chunk_index = global_idx
                        # Chỉ inject header vào chunk đầu tiên của Khoản
                        if j == 0:
                            sub.content = _with_header(sub.content)
                        chunks.append(sub)
                        global_idx += 1

        return chunks

    @staticmethod
    def _split_by_khoan(paragraphs: list[str]) -> list[tuple[str, int | None]]:
        """Gom các đoạn trong một Điều theo Khoản (1., 2., 3. …)."""
        units: list[tuple[str, int | None]] = []
        current_lines: list[str] = []
        current_khoan: int | None = None

        for para in paragraphs:
            first_line = para.split("\n", 1)[0].strip()
            match = KHOAN_RE.match(first_line)
            if match:
                if current_lines:
                    units.append(("\n\n".join(current_lines), current_khoan))
                current_khoan = int(match.group(1))
                current_lines = [para]
            else:
                current_lines.append(para)

        if current_lines:
            units.append(("\n\n".join(current_lines), current_khoan))

        return units

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Tách văn bản thành các đoạn (paragraph), phân cách bởi dòng trống."""
        raw = re.split(r"\n\s*\n+", text.strip())
        return [p.strip() for p in raw if p.strip()]

    # ── Chiến lược 2: heading Markdown ───────────────────────────────────

    def chunk_by_markdown_sections(
        self,
        markdown_text: str,
        document_id: uuid.UUID,
        extra_metadata: dict | None = None,
    ) -> list[ChunkData]:
        """
        Chunk theo section Markdown (heading-aware).
        Giữ nguyên heading của mỗi đoạn làm metadata để tăng relevance khi retrieve.
        """
        sections = self._split_markdown_sections(markdown_text)
        chunks: list[ChunkData] = []
        global_idx = 0

        for heading, content in sections:
            if not content.strip():
                continue
            section_meta = {"heading": heading} if heading else {}
            if extra_metadata:
                section_meta.update(extra_metadata)

            section_chunks = self.chunk_text(content, document_id, section_meta)
            for c in section_chunks:
                c.chunk_index = global_idx
                chunks.append(c)
                global_idx += 1

        return chunks

    @staticmethod
    def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
        """Tách Markdown thành (heading, content) theo các dòng bắt đầu bằng #."""
        sections: list[tuple[str, str]] = []
        current_heading = ""
        current_lines: list[str] = []

        for line in text.splitlines():
            if line.startswith("#"):
                if current_lines:
                    sections.append((current_heading, "\n".join(current_lines)))
                current_heading = line.lstrip("#").strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "\n".join(current_lines)))

        return sections

    # ── Chiến lược 3: cắt toàn văn bản theo câu (fallback cuối cùng) ─────

    def chunk_text(
        self,
        text: str,
        document_id: uuid.UUID,
        extra_metadata: dict | None = None,
    ) -> list[ChunkData]:
        """
        Chia text thành các chunk.

        Parameters
        ----------
        text : str
            Toàn bộ text đã parse + OCR (nếu có).
        document_id : uuid.UUID
            Dùng để gắn vào metadata của mỗi chunk.
        extra_metadata : dict, optional
            Metadata bổ sung (tên file, loại file, …).

        Returns
        -------
        list[ChunkData]
        """
        if not text.strip():
            logger.warning(f"Document {document_id}: text rỗng, không có chunk nào.")
            return []

        base_meta = {"document_id": str(document_id)}
        if extra_metadata:
            base_meta.update(extra_metadata)

        llama_doc = LlamaDocument(text=text, metadata=base_meta)

        nodes = self._splitter.get_nodes_from_documents([llama_doc])

        chunks: list[ChunkData] = []
        for idx, node in enumerate(nodes):
            page_num = node.metadata.get("page_label")
            chunks.append(
                ChunkData(
                    chunk_index=idx,
                    content=node.get_content(),
                    page_number=int(page_num) if page_num else None,
                    metadata={k: v for k, v in node.metadata.items() if k != "page_label"},
                )
            )

        logger.info(
            f"Document {document_id}: chia thành {len(chunks)} chunk "
            f"(size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})."
        )
        return chunks


# Singleton
chunking_service = ChunkingService()
