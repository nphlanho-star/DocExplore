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
    # Tách riêng khỏi EMBEDDING_DEVICE để có thể "vừa CPU vừa GPU": ví dụ
    # embedding chạy GPU (indexing hàng loạt, lợi nhiều về tốc độ) trong khi
    # reranker chạy CPU (chỉ vài chục chunk mỗi query, không cần GPU, đỡ
    # tốn VRAM cho GPU nhỏ chạy chung với Ollama).
    RERANKER_DEVICE: str = "cpu"   # "cuda" nếu muốn reranker cũng dùng GPU

    # ── Ollama / LLM ─────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:0.5b"   # tên model trong ollama

    # ── Chunking ─────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ── Retrieval ────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 20    # số chunk lấy từ Qdrant trước rerank

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
