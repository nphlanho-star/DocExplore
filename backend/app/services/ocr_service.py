"""
services/ocr_service.py — OCR thông minh: chỉ kích hoạt PaddleOCR
khi phát hiện Word có hình ảnh nhúng cần đọc chữ.

Chiến lược:
  ┌─────────────────────────────────────────────────────────────────┐
  │  File                  │  OCR engine                           │
  ├─────────────────────────────────────────────────────────────────┤
  │  PDF dạng scan/ảnh     │  Docling OCR (đã bật trong pipeline)  │
  │  Word (.docx) có ảnh   │  PaddleOCR chạy trên từng ảnh nhúng  │
  │  Word không có ảnh     │  KHÔNG chạy PaddleOCR                 │
  │  Excel / PPTX          │  KHÔNG chạy OCR riêng                 │
  └─────────────────────────────────────────────────────────────────┘

Quy trình cho Word:
  1. _detect_embedded_images()  → tìm ảnh nhúng trong .docx
  2. Nếu danh sách ảnh rỗng     → trả về None (bỏ qua)
  3. Nếu có ảnh                 → _run_paddle_on_images()
  4. Ghép kết quả OCR vào cuối text đã parse
"""
import io
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class OCRResult:
    """Kết quả OCR từ các ảnh nhúng trong tài liệu."""

    extracted_texts: list[str] = field(default_factory=list)  # text của từng ảnh
    image_count: int = 0

    @property
    def combined_text(self) -> str:
        """Ghép toàn bộ text OCR lại (dùng để bổ sung vào nội dung Docling)."""
        return "\n\n".join(
            f"[Hình ảnh {i + 1}]\n{t}" for i, t in enumerate(self.extracted_texts) if t.strip()
        )

    @property
    def has_content(self) -> bool:
        return any(t.strip() for t in self.extracted_texts)


class OCRService:
    """
    Dịch vụ OCR có điều kiện — chỉ chạy PaddleOCR khi cần thiết.
    PaddleOCR được khởi tạo lazy (import + init khi lần đầu dùng).
    """

    def __init__(self) -> None:
        self._paddle_ocr = None   # lazy init

    # ── Public entry point ────────────────────────────────────────────

    def process_docx(self, file_bytes: bytes) -> Optional[OCRResult]:
        """
        Xử lý file Word:
          - Bước 1: Detect ảnh nhúng
          - Bước 2: Nếu có ảnh → chạy PaddleOCR
          - Bước 3: Nếu không có ảnh → trả về None (không làm gì)

        Parameters
        ----------
        file_bytes : bytes
            Nội dung file .docx

        Returns
        -------
        OCRResult nếu có ảnh và OCR thành công, None nếu không có ảnh.
        """
        images = self._detect_embedded_images(file_bytes)

        if not images:
            logger.debug("Word không chứa ảnh nhúng — bỏ qua PaddleOCR.")
            return None

        logger.info(f"Phát hiện {len(images)} ảnh nhúng trong Word — bắt đầu PaddleOCR.")
        return self._run_paddle_on_images(images)

    # ── Step 1: Detect ────────────────────────────────────────────────

    @staticmethod
    def _detect_embedded_images(file_bytes: bytes) -> list[bytes]:
        """
        Trích xuất tất cả ảnh nhúng từ .docx (thực chất là file ZIP).

        .docx lưu ảnh trong thư mục word/media/ bên trong ZIP.
        Chỉ lấy các định dạng ảnh phổ biến (png, jpg, jpeg, bmp, tiff).

        Returns
        -------
        Danh sách bytes của từng ảnh.
        """
        IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif"}
        images: list[bytes] = []

        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                for entry in zf.namelist():
                    path = Path(entry)
                    # Ảnh nằm trong word/media/
                    if "word/media" in entry and path.suffix.lower() in IMAGE_EXTENSIONS:
                        images.append(zf.read(entry))
        except zipfile.BadZipFile:
            logger.warning("File không phải định dạng .docx hợp lệ (bad ZIP).")
        except Exception as exc:
            logger.error(f"Lỗi khi detect ảnh trong Word: {exc}")

        return images

    # ── Step 2: PaddleOCR ─────────────────────────────────────────────

    def _get_paddle(self):
        """Khởi tạo PaddleOCR một lần duy nhất (lazy)."""
        if self._paddle_ocr is None:
            try:
                from paddleocr import PaddleOCR
                # lang="vi" + "en" để hỗ trợ cả tiếng Việt lẫn tiếng Anh
                self._paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang="vi",
                    show_log=False,
                    use_gpu=False,   # đổi thành True nếu có GPU
                )
                logger.info("PaddleOCR đã được khởi tạo.")
            except ImportError:
                logger.error("PaddleOCR chưa được cài đặt. Chạy: pip install paddleocr")
                raise
        return self._paddle_ocr

    def _ocr_single_image(self, image_bytes: bytes) -> str:
        """
        Chạy PaddleOCR trên một ảnh đơn.

        Returns
        -------
        Chuỗi text đã nhận dạng (rỗng nếu không đọc được gì).
        """
        paddle = self._get_paddle()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = Path(tmp.name)

        try:
            result = paddle.ocr(str(tmp_path), cls=True)
            if not result or not result[0]:
                return ""

            lines = []
            for line in result[0]:
                # result[0] = list of [[box], (text, confidence)]
                text, confidence = line[1]
                if confidence >= 0.6:   # bỏ qua kết quả OCR kém tin cậy
                    lines.append(text)

            return "\n".join(lines)
        except Exception as exc:
            logger.error(f"PaddleOCR lỗi trên ảnh: {exc}")
            return ""
        finally:
            tmp_path.unlink(missing_ok=True)

    def _run_paddle_on_images(self, images: list[bytes]) -> OCRResult:
        """
        Chạy PaddleOCR lần lượt trên từng ảnh.
        Bỏ qua ảnh lỗi và tiếp tục với ảnh còn lại.
        """
        result = OCRResult(image_count=len(images))

        for idx, img_bytes in enumerate(images):
            try:
                text = self._ocr_single_image(img_bytes)
                result.extracted_texts.append(text)
                logger.debug(
                    f"  Ảnh {idx + 1}/{len(images)}: "
                    f"{'OK' if text.strip() else 'trống'} "
                    f"({len(text)} ký tự)"
                )
            except Exception as exc:
                logger.warning(f"  Ảnh {idx + 1} bị lỗi, bỏ qua: {exc}")
                result.extracted_texts.append("")

        if result.has_content:
            logger.info(
                f"PaddleOCR hoàn tất: {len(images)} ảnh, "
                f"{sum(bool(t.strip()) for t in result.extracted_texts)} ảnh có chữ."
            )
        else:
            logger.info("PaddleOCR: không đọc được chữ nào từ các ảnh nhúng.")

        return result

    # ── Hàm tiện ích dùng ở worker ────────────────────────────────────

    def merge_ocr_into_text(self, base_text: str, ocr_result: Optional[OCRResult]) -> str:
        """
        Ghép text OCR (từ ảnh nhúng) vào cuối text đã parse bởi Docling.
        Nếu không có OCR result hoặc OCR không có nội dung → trả nguyên base_text.
        """
        if ocr_result is None or not ocr_result.has_content:
            return base_text

        separator = "\n\n--- Nội dung từ hình ảnh (OCR) ---\n\n"
        return base_text + separator + ocr_result.combined_text


# Singleton
ocr_service = OCRService()
