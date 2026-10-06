"""
config.py — Cấu hình toàn bộ ứng dụng qua biến môi trường (.env).
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────
    APP_NAME: str = "RAG Document QA"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # ── PostgreSQL ────────────────────────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ragdb"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def SYNC_DATABASE_URL(self) -> str:
        """Dùng cho Alembic migrations (sync driver)."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── MinIO ─────────────────────────────────────────────────────────
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_ORIGINAL: str = "documents-original"
    MINIO_BUCKET_PROCESSED: str = "documents-processed"
    MINIO_BUCKET_OCR: str = "documents-ocr"

    # ── Redis ─────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Celery ────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── Qdrant ────────────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "documents"

    # ── Embedding (BGE-M3) ────────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"   # "cuda" nếu có GPU
    EMBEDDING_BATCH_SIZE: int = 32
    VECTOR_SIZE: int = 1024          # BGE-M3 output dimension

    # ── Reranker (bge-reranker-v2-m3) ────────────────────────────────
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_TOP_K: int = 5
    # Điểm rerank tối thiểu để 1 đoạn được đưa vào ngữ cảnh trả lời (thấp hơn → dễ tìm thấy hơn
    # nhưng tăng nguy cơ đoạn kém liên quan). Chỉnh qua .env: RERANK_MIN_SCORE=0.2
    RERANK_MIN_SCORE: float = 0.15
    # Tách riêng khỏi EMBEDDING_DEVICE để có thể "vừa CPU vừa GPU": ví dụ
    # embedding chạy GPU (indexing hàng loạt, lợi nhiều về tốc độ) trong khi
    # reranker chạy CPU (chỉ vài chục chunk mỗi query, không cần GPU, đỡ
    # tốn VRAM cho GPU nhỏ chạy chung với Ollama).
    RERANKER_DEVICE: str = "cpu"   # "cuda" nếu muốn reranker cũng dùng GPU

    # ── Ollama / LLM ─────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:1.5b"   # tên model trong ollama

    # ── Chunking ─────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ── Retrieval ────────────────────────────────────────────────────
    # Độ dài tối đa (token) mỗi cặp câu hỏi–đoạn khi rerank. Nhỏ hơn → nhanh hơn trên CPU.
    RERANK_MAX_LENGTH: int = 384
    # Nén context theo câu: khi tổng context vượt ngân sách (hoặc 1 chunk quá dài), giữ các CÂU liên quan
    # nhất với câu hỏi (nguyên văn) thay vì cắt đuôi / bỏ cả chunk. False → dùng cách cắt cũ.
    CONTEXT_COMPRESS_ENABLED: bool = True
    # Bộ nhớ hội thoại: mỗi lượt (câu hỏi + câu trả lời) tiêu hao một phần ngân sách này (ký tự).
    # Cộng dồn đạt 100% → nén bộ nhớ: các câu hỏi cũ gộp thành ghi chú ngắn, chỉ giữ nguyên văn MEMORY_KEEP_TURNS lượt gần nhất.
    MEMORY_BUDGET_CHARS: int = 8000
    MEMORY_KEEP_TURNS: int = 2
    # Nén ở CHỖ HẾT Ý, không cắt giữa chừng (luôn nén theo trọn lượt hỏi–đáp):
    #  • đạt MIN (60%) và người dùng vừa chuyển sang chủ đề mới → nén toàn bộ phần chủ đề cũ ngay (hạ xuống 6X%)
    #  • còn đang nói cùng chủ đề → chờ cho trọn ý, tới MAX (85%) mới buộc phải nén (rướng lên 7X–8X%)
    MEMORY_COMPACT_MIN: float = 0.60
    MEMORY_COMPACT_MAX: float = 0.85
    MEMORY_SUMMARY_CHARS: int = 1800   # kích thước tối đa phần ghi nhớ sau khi nén
    # Nếu điểm rerank CAO NHẤT lần 1 thấp hơn mức này → câu hỏi gần như chắc chắn ngoài tài liệu
    # → bỏ qua vòng HyDE (tiết kiệm 1 lượt LLM + 1 lượt rerank).
    HYDE_MIN_TOP_SCORE: float = 0.03
    RETRIEVAL_TOP_K: int = 20    # số chunk lấy từ Qdrant trước rerank

    # ── Hybrid Search (dense + sparse / BM25) ────────────────────────
    # True → BGE-M3 sparse vectors được index và search song song với dense.
    # LẦN ĐẦU BẬT: Qdrant collection sẽ bị xóa và tạo lại với sparse config
    # → cần re-upload toàn bộ tài liệu. False → chỉ dùng dense (mặc định cũ).
    HYBRID_SEARCH_ENABLED: bool = False

    # ── Web search bổ sung (Tavily) — chỉ cho câu hỏi PHÁP LUẬT ngoài tài liệu ──
    TAVILY_API_KEY: str = ""
    WEB_SEARCH_ENABLED: bool = False
    WEB_SEARCH_MAX_RESULTS: int = 4
    WEB_SEARCH_TIMEOUT: float = 15.0
    WEB_SEARCH_SNIPPET_CHARS: int = 1200
    WEB_SEARCH_DEPTH: str = "basic"   # "advanced" → trích xuất tốt hơn nhưng tốn gấp đôi credit Tavily
    # Chỉ giữ kết quả web có điểm rerank (câu hỏi vs đoạn trích) ≥ ngưỡng này — loại kết quả lạc đề.
    WEB_MIN_RELEVANCE: float = 0.15
    WEB_SENTENCE_MIN: float = 0.10     # chỉ giữ CÂU web có điểm liên quan ≥ mức này (tối đa 5 câu/nguồn)
    WEB_SEARCH_DOMAINS: str = "thuvienphapluat.vn,vbpl.vn,chinhphu.vn,moj.gov.vn,luatvietnam.vn"

    # ── Auth / JWT ────────────────────────────────────────────────────
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 ngày

    # ── Upload ────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list[str] = [
        ".pdf", ".docx", ".doc", ".pptx", ".txt",
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
