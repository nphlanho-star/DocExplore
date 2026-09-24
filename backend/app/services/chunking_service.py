"""
services/chunking_service.py — Chia tài liệu thành các chunk bằng LlamaIndex.

Chiến lược:
  - Dùng SentenceSplitter của LlamaIndex để chunk có overlap.
  - Giữ metadata: document_id, page_number, chunk_index, heading.
"""
import uuid
from dataclasses import dataclass, field

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SentenceSplitter
from loguru import logger

from app.config import get_settings

settings = get_settings()


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


# Singleton
chunking_service = ChunkingService()
