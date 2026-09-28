import reflex as rx

config = rx.Config(
    app_name="frontend",
    backend_port=8001,  # tránh xung đột với FastAPI đang chạy trên 8000
)
