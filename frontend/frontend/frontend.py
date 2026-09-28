"""
frontend/frontend.py — Entry point của Reflex app.
Reflex yêu cầu file này trùng tên với app_name trong rxconfig.py.
"""
import reflex as rx
from frontend.pages.dashboard import dashboard_page
from frontend.pages.chat import chat_page

app = rx.App(
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    style={
        "font_family": "Inter, sans-serif",
        "box_sizing": "border-box",
    },
)

app.add_page(dashboard_page, route="/", title="Dashboard — RAG QA")
app.add_page(dashboard_page, route="/dashboard", title="Dashboard — RAG QA")
app.add_page(chat_page, route="/chat", title="Chat — RAG QA")
