"""
services/document_parser.py — Parse tài liệu bằng Docling.

Luồng:
  1. Nhận bytes + extension.
  2. Ghi tạm ra disk (Docling cần path).
  3. Docling chuyển sang Markdown / text thuần.
  4. Trả về ParseResult chứa text + metadata.
"""
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from loguru import logger


@dataclass
class ParseResult:
    text: str                          # toàn bộ nội dung text đã chuẩn hóa
    markdown: str                      # nội dung dạng Markdown (giữ cấu trúc)
    page_count: int = 0
    word_count: int = 0
    language: str | None = None
    metadata: dict = field(default_factory=dict)


class DocumentParserService:
    """Dùng Docling để extract text từ PDF, Word, Excel, PowerPoint."""

    # Các định dạng Docling hỗ trợ
    SUPPORTED_FORMATS = {
        ".pdf": InputFormat.PDF,
        ".docx": InputFormat.DOCX,
        ".doc": InputFormat.DOCX,
        ".pptx": InputFormat.PPTX,
        ".xlsx": InputFormat.XLSX,
        ".xls": InputFormat.XLSX,
        ".txt": None,   # xử lý riêng, không cần Docling
    }

    def __init__(self) -> None:
        # Cấu hình pipeline cho PDF (có thể bật OCR ở đây nếu là PDF scan)
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = True          # Docling tự OCR PDF scan
        pdf_options.do_table_structure = True

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            }
        )

    def parse(self, file_bytes: bytes, extension: str) -> ParseResult:
        """
        Parse file từ bytes.

        Parameters
        ----------
        file_bytes : bytes
            Nội dung file gốc.
        extension : str
            Ví dụ: ".pdf", ".docx"

        Returns
        -------
        ParseResult
        """
        ext = extension.lower()

        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Định dạng không được hỗ trợ: {extension}")

        # txt — không cần Docling
        if ext == ".txt":
            text = file_bytes.decode("utf-8", errors="replace")
            return ParseResult(
                text=text,
                markdown=text,
                word_count=len(text.split()),
            )

        return self._parse_with_docling(file_bytes, ext)

    def _parse_with_docling(self, file_bytes: bytes, ext: str) -> ParseResult:
        suffix = ext if ext.startswith(".") else f".{ext}"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            logger.debug(f"Docling parsing: {tmp_path}")
            result = self._converter.convert(str(tmp_path))
            doc = result.document

            # Lấy nội dung Markdown và text thuần
            markdown = doc.export_to_markdown()
            text = doc.export_to_text()

            # Metadata cơ bản
            page_count = getattr(doc, "page_count", 0) or 0
            word_count = len(text.split())

            return ParseResult(
                text=text,
                markdown=markdown,
                page_count=page_count,
                word_count=word_count,
                metadata={"source_format": ext},
            )
        except Exception as exc:
            logger.error(f"Docling parse error: {exc}")
            raise
        finally:
            tmp_path.unlink(missing_ok=True)


# Singleton
document_parser = DocumentParserService()
