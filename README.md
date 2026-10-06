# RAG Document QA — Hỏi đáp tài liệu pháp luật

Hệ thống hỏi đáp tài liệu (tập trung văn bản luật tiếng Việt) dựa trên RAG.
**Ưu tiên độ chính xác: chỉ trả lời từ tài liệu đã tải lên, không bịa; không có thông tin thì nói không có.**

**Công nghệ:** FastAPI · Celery · PostgreSQL · Qdrant · BGE-M3 (dense + sparse) · bge-reranker-v2-m3 · Ollama (Qwen2.5) · Reflex (giao diện) · Tavily (tra cứu web, tùy chọn)

---

## Tính năng chính

| Nhóm | Mô tả |
|------|-------|
| Tài liệu | Upload `.docx`/PDF… → Docling parse → chunk theo cấu trúc Chương/Điều → embedding → Qdrant. OCR (PaddleOCR) chỉ chạy khi file có ảnh nhúng |
| Tìm kiếm | Hybrid Search (dense + sparse BGE-M3, gộp RRF, tùy chọn) → rerank bằng bge-reranker-v2-m3 → lọc theo ngưỡng điểm |
| Hiểu yêu cầu | 1 lượt LLM phân loại: hỏi đáp / tóm tắt (N điều đầu, cuối, khoảng) / liệt kê các Điều / so sánh / làm rõ — kết quả được kiểm tra lại bằng code, sai thì quay về pipeline thường |
| Chống ảo giác | Chỉ trả lời từ `<tai_lieu>`; từ chối khi điểm rerank thấp; chặn chữ Hán; chặn prompt injection; tìm theo Điều trực tiếp từ DB |
| Nén context | Khi context vượt ngân sách, nén theo câu: giữ **nguyên văn** các câu liên quan nhất (chấm bằng reranker), không dùng LLM viết lại |
| Bộ nhớ hội thoại | Mỗi lượt tiêu hao dần bộ nhớ; tới ~60–85% (ở chỗ hết ý) nén các lượt cũ thành ghi nhớ trích nguyên văn để về sau vẫn nhớ. Có chỉ báo % trên giao diện |
| Lịch sử chat | Danh sách các cuộc trò chuyện, mở lại / xóa; nhớ file đang làm việc giữa các lượt |
| Tra cứu web (tùy chọn) | Khi tài liệu không có thông tin **và** câu hỏi là về pháp luật → Tavily tìm trên các trang luật Việt Nam, hiển thị **nguyên văn đoạn trích + link** (không để LLM tóm tắt), có lọc lạc đề theo từng câu |

---

## Yêu cầu

| Công cụ | Ghi chú |
|---------|---------|
| Python 3.10+ | Dự án đang chạy trên Python 3.10 (Windows) |
| PostgreSQL 16, Redis 7, Qdrant | Chạy bằng Docker Compose **hoặc** cài native (Memurai thay Redis trên Windows) |
| Ollama | `ollama pull qwen2.5:3b` |
| Git | |
| GPU NVIDIA (khuyến nghị) | Dùng cho reranker (`RERANKER_DEVICE=cuda`) — rerank 20 đoạn trên CPU mất ~60 s, trên GPU chỉ vài giây |

Cấu hình đã thử: laptop RTX 3050 4 GB, RAM 8 GB — xem mục [Máy yếu](#máy-yếu-8-gb-ram) bên dưới.

---

## Cài đặt lần đầu

### 1. Clone & tạo file cấu hình

```bash
git clone <repo-url>
cd RAG-2/backend
cp .env.example .env
```

Mở `backend/.env` và chỉnh ít nhất `SECRET_KEY`:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 2. Hạ tầng (PostgreSQL · Redis · Qdrant)

```bash
# từ thư mục backend/
docker compose up -d postgres redis qdrant
docker compose ps        # kiểm tra healthy
```

> Không dùng Docker: cài native PostgreSQL, Redis (Windows: Memurai) và Qdrant rồi trỏ `.env` tới đúng host/port.
> File gốc được lưu ở thư mục `backend/storage/` (hệ thống file cục bộ, không cần MinIO).

### 3. Python dependencies (backend)

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash trên Windows
# .venv\Scripts\activate             # CMD/PowerShell
# source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt
```

**Dùng GPU cho reranker:** `requirements.txt` có thể cài torch bản CPU. Kiểm tra:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu in `False` mà máy có GPU NVIDIA, cài torch bản CUDA:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Sau đó đặt `RERANKER_DEVICE=cuda` trong `.env`.

### 4. Migrations

```bash
alembic upgrade head
```

### 5. LLM

```bash
ollama pull qwen2.5:3b
```

BGE-M3 và reranker tự tải từ HuggingFace ở lần chạy đầu (~2 GB mỗi model).

### 6. Giao diện (Reflex)

```bash
cd ../frontend
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
```

---

## Chạy ứng dụng

Mở các terminal riêng (backend đã activate venv):

**Terminal 1 — Backend API** (cổng 8000):

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> ⚠️ **Không dùng `--reload` khi chạy thật**: mỗi lần lưu file sẽ nạp lại BGE-M3 + reranker, gây tăng RAM đột ngột
> (lỗi Windows `os error 1455`). Chỉ bật `--reload` khi đang lập trình và máy dư RAM.

**Terminal 2 — Celery worker** (chỉ cần khi upload/xử lý tài liệu; tắt khi không dùng để tiết kiệm RAM):

```bash
cd backend
celery -A app.workers.celery_app worker --loglevel=info --concurrency=2
```

(Windows: thêm `--pool=solo` nếu Celery báo lỗi tiến trình con.)

**Terminal 3 — Giao diện:**

```bash
cd frontend
reflex run
```

Mở http://localhost:3000. (Reflex backend nội bộ dùng cổng 8001 để không trùng FastAPI 8000.)

**Lần đầu:** vào trang Dashboard → upload tài liệu → đợi trạng thái `ready` → sang trang chat đặt câu hỏi.

### Chạy bằng Docker Compose

`backend/docker-compose.yml` hiện chỉ dựng **PostgreSQL, Redis, Qdrant**. Backend, Celery và giao diện chạy native như trên.

---

## Xác thực (chế độ dev)

Dự án đang ở **chế độ dev, không cần đăng nhập**: mọi request dùng một tài khoản `dev@local` được tạo tự động khi backend khởi động.
Các endpoint `/api/auth/*` vẫn tồn tại nhưng không bắt buộc.
**Không mở dịch vụ này ra Internet khi chưa bật xác thực thật.**

---

## Endpoint chính

| Mục | URL |
|-----|-----|
| Swagger | http://localhost:8000/docs |
| Upload tài liệu | `POST /api/documents/upload` |
| Danh sách / chunk / xóa tài liệu | `GET /api/documents/`, `GET /api/documents/{id}/chunks`, `DELETE /api/documents/{id}` |
| Hỏi đáp (streaming SSE) | `POST /api/chat/query/stream` |
| Hỏi đáp (một lần) | `POST /api/chat/query` |
| Phiên chat | `GET/POST /api/chat/sessions`, `GET/DELETE /api/chat/sessions/{id}` |
| Qdrant dashboard | http://localhost:6333/dashboard |
| Flower (tùy chọn) | `celery -A app.workers.celery_app flower --port=5555` → http://localhost:5555 |

Ví dụ hỏi đáp:

```bash
curl -N -X POST http://localhost:8000/api/chat/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Tóm tắt 10 điều đầu tiên của file P1", "top_k": 5}'
```

Các sự kiện SSE: `status`, `sources`, `token`, `memory` (mức bộ nhớ), `done`, `error`.

---

## Luồng xử lý

### Tài liệu

```
Upload → validate → lưu file (backend/storage) + metadata (PostgreSQL)
   → Celery: Docling parse → (có ảnh? PaddleOCR) → chunk theo Chương/Điều
   → BGE-M3 embedding (dense + sparse) → Qdrant + PostgreSQL
```

### Hỏi đáp

```
Câu hỏi
 ├─ chống injection / chào hỏi
 ├─ Bộ hiểu yêu cầu (1 lượt LLM, kiểm tra lại bằng code): hỏi đáp | tóm tắt | liệt kê | so sánh | làm rõ
 ├─ Phạm vi tài liệu (file nêu trong câu → file đang làm việc → tất cả)
 ├─ Tìm kiếm: BGE-M3 (hybrid) → Qdrant top-K → rerank → lọc ngưỡng (RERANK_MIN_SCORE)
 │     └─ không đủ tin cậy: HyDE thử lại (bỏ qua nếu điểm quá thấp = ngoài tài liệu)
 ├─ Nén context theo câu (nếu vượt ngân sách)
 ├─ Có context  → Qwen2.5 trả lời, kèm Điều/nguồn + ghi nhớ hội thoại
 └─ Không có    → (nếu bật) hỏi luật? → Tavily → lọc theo câu → hiển thị nguyên văn + link
                  (ngược lại) "Tôi không tìm thấy thông tin trong tài liệu"
```

---

## Biến môi trường quan trọng (`backend/.env`)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `SECRET_KEY` | ⚠️ cần đổi | Khóa ký JWT |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model trong Ollama |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` hoặc `cuda` |
| `RERANKER_DEVICE` | `cpu` | `cuda` nếu torch có CUDA (rất khuyến nghị) |
| `RERANK_MAX_LENGTH` | `384` | Số token tối đa mỗi cặp khi rerank (nhỏ hơn = nhanh hơn) |
| `RETRIEVAL_TOP_K` | `20` | Số đoạn lấy từ Qdrant trước rerank (máy yếu: `10`) |
| `RERANKER_TOP_K` | `5` | Số đoạn giữ lại sau rerank |
| `RERANK_MIN_SCORE` | `0.15` | Dưới ngưỡng này coi như không có thông tin |
| `HYBRID_SEARCH_ENABLED` | `false` | Bật dense + sparse. ⚠️ Bật lần đầu **xóa & tạo lại** collection Qdrant → phải upload lại tài liệu |
| `HYDE_MIN_TOP_SCORE` | `0.03` | Điểm cao nhất dưới mức này → bỏ vòng HyDE (câu ngoài tài liệu) |
| `CONTEXT_COMPRESS_ENABLED` | `true` | Nén context theo câu khi vượt ngân sách |
| `MEMORY_BUDGET_CHARS` | `8000` | Ngân sách bộ nhớ hội thoại (ký tự) |
| `MEMORY_COMPACT_MIN` / `MAX` | `0.60` / `0.85` | Khoảng mức đầy để nén bộ nhớ (nén ở chỗ hết ý) |
| `MEMORY_KEEP_TURNS` | `2` | Số lượt gần nhất giữ nguyên văn khi buộc nén |
| `MEMORY_SUMMARY_CHARS` | `1800` | Kích thước tối đa phần ghi nhớ sau nén |
| `TAVILY_API_KEY` | _(trống)_ | Khóa Tavily — **chỉ ghi vào `.env`, không commit** |
| `WEB_SEARCH_ENABLED` | `false` | Bật tra cứu web cho câu hỏi pháp luật ngoài tài liệu |
| `WEB_SEARCH_DOMAINS` | các trang luật VN | Danh sách trang được phép tìm (phẩy ngăn cách) |
| `WEB_SEARCH_DEPTH` | `basic` | `advanced` trích xuất tốt hơn nhưng tốn gấp đôi credit |
| `WEB_MIN_RELEVANCE` / `WEB_SENTENCE_MIN` | `0.15` / `0.10` | Ngưỡng lọc nguồn/câu web lạc đề |
| `MAX_FILE_SIZE_MB` | `50` | Giới hạn upload |

Danh sách đầy đủ và giá trị mặc định: `backend/app/config.py`.

---

## Máy yếu (8 GB RAM)

BGE-M3 và reranker mỗi cái ~2,2 GB, cộng Ollama ~2 GB nên RAM rất sát. Gợi ý:

- Chạy backend **không** `--reload`; tắt Celery khi không upload.
- `RERANKER_DEVICE=cuda` (cần torch CUDA) và `RETRIEVAL_TOP_K=10`.
- Tăng pagefile Windows lên khoảng 16 GB nếu gặp `os error 1455` ("paging file is too small") rồi khởi động lại máy.
- `ollama stop <model>` khi cần giải phóng bộ nhớ.

## Xử lý sự cố

| Triệu chứng | Cách xử lý |
|-------------|------------|
| `os error 1455` khi nạp reranker/embedding | Tăng pagefile, đóng bớt ứng dụng, bỏ `--reload`, tắt Celery |
| Trả lời rất chậm | Xem log `[timing]`; thường là rerank trên CPU → dùng GPU hoặc giảm `RETRIEVAL_TOP_K` |
| Nhiều câu hỏi bị "không tìm thấy" | Điểm rerank thấp: thử hạ `RERANK_MIN_SCORE` (VD `0.10`) |
| Đổi `HYBRID_SEARCH_ENABLED` rồi không tìm được | Collection Qdrant đã tạo lại → upload lại tài liệu |
| Web search không chạy | Cần cả `WEB_SEARCH_ENABLED=true` và `TAVILY_API_KEY`; câu hỏi phải là câu hỏi pháp luật; xem log `[web_search]` |
| Nội dung web bị mất chữ "â" | Lỗi từ nguồn/Tavily; hệ thống tự sửa phần không thể nhầm, còn lại hãy mở link nguồn để đối chiếu |

---

## Cấu trúc dự án

```
backend/
├── alembic/                     # migrations
├── app/
│   ├── main.py                  # FastAPI entrypoint (tạo dev user lúc khởi động)
│   ├── config.py                # cấu hình qua .env
│   ├── database.py              # SQLAlchemy async
│   ├── models/                  # user, document, chat, permission, feedback
│   ├── schemas/
│   ├── api/                     # deps (dev user), users, documents, chat (pipeline hỏi đáp)
│   ├── services/
│   │   ├── document_parser.py   # Docling
│   │   ├── ocr_service.py       # PaddleOCR (chỉ khi cần)
│   │   ├── chunking_service.py  # chunk theo Chương/Điều
│   │   ├── embedding_service.py # BGE-M3 (dense + sparse)
│   │   ├── qdrant_service.py    # vector DB (dense / hybrid)
│   │   ├── reranker_service.py  # bge-reranker-v2-m3 (+ chấm điểm câu cho nén/lọc web)
│   │   ├── llm_service.py       # Ollama/Qwen: bộ hiểu yêu cầu, sinh câu trả lời
│   │   ├── injection_guard.py   # chống prompt injection
│   │   ├── tavily_service.py    # tra cứu web (tùy chọn)
│   │   └── minio_service.py     # lưu file (hệ thống file cục bộ)
│   └── workers/                 # Celery: xử lý tài liệu
├── storage/                     # file gốc đã upload (không commit)
├── docker-compose.yml           # postgres · redis · qdrant
└── .env.example
frontend/                        # Reflex: dashboard + trang chat
planning/                        # tài liệu thiết kế, tech stack, edge case
```

## Lưu ý bảo mật

- `backend/.env` chứa khóa — đã được `.gitignore`; không commit, không dán lên nơi công khai. Nếu từng lộ `SECRET_KEY` hoặc API key thì tạo lại.
- Chế độ dev không có đăng nhập: chỉ chạy trong mạng cục bộ.
- Tra cứu web gửi câu hỏi của người dùng tới Tavily; kết quả web chỉ để tham khảo, **không phải tư vấn pháp lý** và cần đối chiếu văn bản gốc (văn bản có thể đã hết hiệu lực hoặc được thay thế).
