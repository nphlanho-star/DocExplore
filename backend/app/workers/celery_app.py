"""
workers/celery_app.py — Khởi tạo Celery với Redis broker.
"""
from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "rag_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
    # Retry tự động khi broker mất kết nối
    broker_connection_retry_on_startup=True,
    # Giới hạn thời gian thực thi mỗi task
    task_time_limit=60 * 30,         # 30 phút
    task_soft_time_limit=60 * 25,    # cảnh báo sau 25 phút
    # Số lần retry mặc định khi task thất bại
    task_max_retries=3,
    task_default_retry_delay=10,     # giây
    # Theo dõi trạng thái task
    task_track_started=True,
    result_expires=60 * 60 * 24,     # kết quả lưu 1 ngày
)
