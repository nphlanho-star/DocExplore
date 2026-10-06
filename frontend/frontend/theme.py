"""
theme.py — Màu sắc dùng chung cho toàn bộ frontend, hỗ trợ chế độ sáng/tối.

Dùng rx.color_mode_cond() để màu tự động đổi theo color mode hiện tại — Reflex
đã quản lý sẵn trạng thái này (qua rx.color_mode.button()), không cần tự viết
thêm AppState riêng cho dark_mode.
"""
import reflex as rx

# ── Màu đổi theo theme ───────────────────────────────────────────────────────
BG = rx.color_mode_cond(light="#f3f4f8", dark="#0f1117")
CARD = rx.color_mode_cond(light="#ffffff", dark="#1a1d27")
BOT_BG = rx.color_mode_cond(light="#eef0f7", dark="#1e2235")
TEXT = rx.color_mode_cond(light="#1a1d27", dark="#e2e8f0")
MUTED = rx.color_mode_cond(light="#64748b", dark="#94a3b8")
BORDER = rx.color_mode_cond(light="rgba(15,17,23,0.10)", dark="rgba(255,255,255,0.06)")
DIVIDER = rx.color_mode_cond(light="rgba(15,17,23,0.10)", dark="rgba(255,255,255,0.08)")

# ── Màu cố định (giữ nguyên ở cả 2 theme) ────────────────────────────────────
ACCENT = "#7c6af7"
ACCENT2 = "#5eead4"
USER_BG = "#2d2460"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER = "#ef4444"
