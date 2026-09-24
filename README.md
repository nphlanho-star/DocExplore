# RAG Document QA System

Hệ thống hỏi đáp tài liệu dựa trên RAG — FastAPI · LlamaIndex · BGE-M3 · Qdrant · Qwen · React.

---

## Yêu cầu

| Tool | Phiên bản tối thiểu |
|------|---------------------|
| Python | 3.11+ |
| Docker + Docker Compose | 24+ |
| Git | 2.x |
| Ollama | Latest |

---

## Cài đặt lần đầu

### 1. Clone repo

```bash
git clone <repo-url>
cd <repo-name>
```

### 2. Cấu hình môi trường

```bash
cd backend
cp .env.example .env
```

Mở `.env` và chỉnh sửa ít nhất:

```env
# Tạo SECRET_KEY mới (bắt buộc khi production):
# python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY=your_random_secret_here
```

### 3. Khởi động hạ tầng (PostgreSQL · Redis · MinIO · Qdrant)

```bash
# Từ thư mục backend/
docker compose up -d postgres redis minio qdrant
```

Kiểm tra tất cả đều healthy:

```bash
docker compose ps
```

### 4. Cài Python dependencies

```bash
# Tạo virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Cài packages
pip install -r requirements.txt
```

> **Lưu ý PaddleOCR trên Windows:** nếu gặp lỗi khi cài `paddlepaddle`, dùng:
> ```bash
> pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
> pip install paddleocr
> ```

### 5. Chạy database migrations

```bash
# Từ thư mục backend/ (đã activate venv)
alembic upgrade head
```

> Lần đầu chạy sẽ tạo migration tự động từ models:
> ```bash
> alembic revision --autogenerate -m "init"
> alembic upgrade head
> ```

### 6. Pull model LLM qua Ollama

```bash
# Cài Ollama: https://ollama.com/download
ollama pull qwen2.5:0.5b
```

Model BGE-M3 và reranker sẽ tự download từ HuggingFace khi khởi động lần đầu (~2GB).

---

## Chạy ứng dụng

Mở **3 terminal** riêng biệt (đều đã `cd backend` và activate venv):

**Terminal 1 — FastAPI Backend:**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Celery Worker (xử lý tài liệu bất đồng bộ):**

```bash
celery -A app.workers.celery_app worker --loglevel=info --concurrency=2
```

**Terminal 3 — Flower (monitor Celery, tuỳ chọn):**

```bash
celery -A app.workers.celery_app flower --port=5555
```

---

## Chạy toàn bộ bằng Docker Compose (Production)

```bash
cd backend
docker compose up --build
```

Tất cả services sẽ khởi động: PostgreSQL, Redis, MinIO, Qdrant, FastAPI, Celery worker, Flower.

---

## Endpoint quan trọng

| Service | URL |
|---------|-----|
| API Docs (Swagger) | http://localhost:8000/docs |
| API Docs (ReDoc) | http://localhost:8000/redoc |
| MinIO Console | http://localhost:9001 |
| Qdrant Dashboard | http://localhost:6333/dashboard |
| Flower (Celery) | http://localhost:5555 |

---

## Sử dụng nhanh qua API

### Đăng ký & đăng nhập

```bash
# Đăng ký
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","username":"myuser","password":"password123"}'

# Đăng nhập — lấy access token
curl -X POST http://localhost:8000/api/auth/login \
  -F "username=user@example.com" \
  -F "password=password123"
```

### Upload tài liệu

```bash
TOKEN="<access_token>"

curl -X POST http://localhost:8000/api/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/document.docx"
```

### Hỏi đáp

```bash
curl -X POST http://localhost:8000/api/chat/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Tóm tắt nội dung chính của tài liệu?",
    "top_k": 5
  }'
```

---

## Cấu trúc dự án

```
backend/
├── alembic/                    # Database migrations
│   ├── env.py
│   └── versions/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py               # Cấu hình qua .env
│   ├── database.py             # SQLAlchemy async engine
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── user.py
│   │   ├── document.py
│   │   ├── chat.py
│   │   ├── permission.py
│   │   └── feedback.py
│   ├── schemas/                # Pydantic request/response
│   ├── services/               # Business logic
│   │   ├── minio_service.py    # File storage
│   │   ├── document_parser.py  # Docling
│   │   ├── ocr_service.py      # PaddleOCR (chỉ khi cần)
│   │   ├── chunking_service.py # LlamaIndex splitter
│   │   ├── embedding_service.py# BGE-M3
│   │   ├── qdrant_service.py   # Vector DB
│   │   ├── reranker_service.py # BGE Reranker
│   │   └── llm_service.py      # Ollama/Qwen
│   ├── workers/                # Celery
│   │   ├── celery_app.py
│   │   └── tasks.py            # Document processing pipeline
│   └── api/                    # FastAPI routes
│       ├── deps.py             # JWT auth dependency
│       ├── users.py
│       ├── documents.py
│       └── chat.py
├── .env.example
├── alembic.ini
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Luồng xử lý tài liệu

```
Upload file
    │
    ▼
FastAPI nhận + validate
    │ lưu MinIO (file gốc)
    │ lưu PostgreSQL (metadata)
    ▼
Celery Task (bất đồng bộ)
    │
    ├─[1] Docling parse → text + markdown
    │
    ├─[2] Detect ảnh nhúng (chỉ với .docx)
    │       ├── Không có ảnh → bỏ qua PaddleOCR
    │       └── Có ảnh       → PaddleOCR từng ảnh → merge vào text
    │
    ├─[3] Chunking (LlamaIndex SentenceSplitter)
    │
    ├─[4] BGE-M3 Embedding (batch)
    │
    └─[5] Qdrant upsert + PostgreSQL lưu chunks
```

## Luồng hỏi đáp

```
User gửi câu hỏi
    │
    ▼
BGE-M3 tạo query embedding
    │
    ▼
Qdrant search (filter theo quyền truy cập)
    │  top-20 candidates
    ▼
bge-reranker-v2-m3 rerank
    │  top-5 chunks liên quan nhất
    ▼
Qwen2.5 via Ollama (RAG prompt)
    │
    ▼
Câu trả lời + Citation (tên file, trang, đoạn trích)
```

---

## Biến môi trường quan trọng

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `SECRET_KEY` | ⚠️ Cần đổi | JWT signing key |
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | Tên model trong Ollama |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` hoặc `cuda` |
| `CHUNK_SIZE` | `512` | Số token mỗi chunk |
| `RETRIEVAL_TOP_K` | `20` | Số chunk lấy từ Qdrant trước rerank |
| `RERANKER_TOP_K` | `5` | Số chunk giữ lại sau rerank |
| `MAX_FILE_SIZE_MB` | `50` | Giới hạn dung lượng upload |
