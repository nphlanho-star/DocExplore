"""
main.py — FastAPI application entrypoint.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings
from app.database import engine, Base
from app.api.users import router as users_router
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router

settings = get_settings()


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi động: tạo bảng DB. Tắt: đóng kết nối."""
    logger.info("Khởi động ứng dụng RAG…")

    # Tạo tất cả bảng nếu chưa tồn tại (production nên dùng Alembic)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables sẵn sàng.")

    # Kiểm tra Ollama
    from app.services.llm_service import llm_service
    ok = await llm_service.health_check()
    if not ok:
        logger.warning(
            f"Ollama chưa sẵn sàng hoặc model '{settings.OLLAMA_MODEL}' chưa được pull. "
            f"Chạy: ollama pull {settings.OLLAMA_MODEL}"
        )

    yield

    logger.info("Đóng engine database…")
    await engine.dispose()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Hệ thống hỏi đáp tài liệu RAG — FastAPI + LlamaIndex + BGE-M3 + Qdrant + Qwen."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(users_router)
app.include_router(documents_router)
app.include_router(chat_router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": settings.APP_VERSION}
