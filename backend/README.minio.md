# Chạy MinIO native trên Windows

MinIO đã không còn public image trên Docker Hub từ 2024.
Thay vào đó, chạy MinIO trực tiếp bằng binary Windows.

## 1. Tải minio.exe

Mở PowerShell hoặc Command Prompt:

```powershell
Invoke-WebRequest -Uri "https://dl.min.io/server/minio/release/windows-amd64/minio.exe" -OutFile "D:\minio\minio.exe"
```

Hoặc tải thủ công từ: https://min.io/download#/windows  
Lưu vào `D:\minio\minio.exe` (hoặc bất kỳ thư mục nào).

## 2. Chạy MinIO

Mở Git Bash / PowerShell:

```bash
mkdir -p D:/minio-data
MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin \
  D:/minio/minio.exe server D:/minio-data --console-address ":9001"
```

Trên PowerShell:

```powershell
$env:MINIO_ROOT_USER="minioadmin"
$env:MINIO_ROOT_PASSWORD="minioadmin"
D:\minio\minio.exe server D:\minio-data --console-address ":9001"
```

## 3. Kiểm tra

- API endpoint: http://localhost:9000
- Console UI:   http://localhost:9001  (login: minioadmin / minioadmin)

## 4. Thứ tự khởi động toàn bộ stack

```
1. docker-compose up -d          # postgres + redis + qdrant
2. (terminal riêng) minio.exe server D:\minio-data --console-address ":9001"
3. (terminal riêng) cd /d/RAG-2/backend && source .venv/Scripts/activate && python -m uvicorn app.main:app --reload --port 8000
4. (terminal riêng) cd /d/RAG-2/backend && source .venv/Scripts/activate && celery -A app.workers.celery_app worker --loglevel=info -P solo
5. (terminal riêng) cd /d/RAG-2/frontend && source .venv/Scripts/activate && reflex run
```
