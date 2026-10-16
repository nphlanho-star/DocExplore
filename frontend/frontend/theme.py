"""
theme.py — Design tokens dùng chung cho toàn bộ frontend.
Hỗ trợ chế độ sáng/tối qua rx.color_mode_cond().
"""
import reflex as rx

# ── Surfaces ──────────────────────────────────────────────────────────────────
BG      = rx.color_mode_cond(light="#f3f4f8",  dark="#0f1117")
CARD    = rx.color_mode_cond(light="#ffffff",   dark="#1a1d27")
BOT_BG  = rx.color_mode_cond(light="#eef0f7",  dark="#1e2235")
SIDEBAR = rx.color_mode_cond(light="#f8f9fc",  dark="#141721")

# ── Text ──────────────────────────────────────────────────────────────────────
TEXT    = rx.color_mode_cond(light="#1a1d27",  dark="#e2e8f0")
MUTED   = rx.color_mode_cond(light="#64748b",  dark="#94a3b8")
SUBTLE  = rx.color_mode_cond(light="#94a3b8",  dark="#64748b")

# ── Borders ───────────────────────────────────────────────────────────────────
BORDER  = rx.color_mode_cond(light="rgba(15,17,23,0.10)",  dark="rgba(255,255,255,0.06)")
DIVIDER = rx.color_mode_cond(light="rgba(15,17,23,0.10)",  dark="rgba(255,255,255,0.08)")

# ── Fixed accent palette ──────────────────────────────────────────────────────
ACCENT  = "#7c6af7"
ACCENT2 = "#5eead4"
USER_BG = "#2d2460"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER  = "#ef4444"

# ── Gradient shorthands ───────────────────────────────────────────────────────
ACCENT_GRAD  = "linear-gradient(135deg, #7c6af7 0%, #9d8df7 100%)"
ACCENT_GRAD2 = "linear-gradient(135deg, #7c6af7 0%, #5eead4 100%)"
USER_GRAD    = "linear-gradient(135deg, #2d2460 0%, #3b2d7e 100%)"

# ── Shadow tokens ─────────────────────────────────────────────────────────────
SHADOW_SM = rx.color_mode_cond(
    light="0 1px 3px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.04)",
    dark="0 1px 4px rgba(0,0,0,0.35), 0 1px 2px rgba(0,0,0,0.20)",
)
SHADOW_MD = rx.color_mode_cond(
    light="0 4px 16px rgba(0,0,0,0.09), 0 2px 6px rgba(0,0,0,0.05)",
    dark="0 4px 20px rgba(0,0,0,0.45), 0 2px 8px rgba(0,0,0,0.30)",
)
SHADOW_LG = "0 24px 64px rgba(0,0,0,0.48), 0 8px 20px rgba(0,0,0,0.28)"
