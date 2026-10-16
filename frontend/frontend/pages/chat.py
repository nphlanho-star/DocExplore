"""
pages/chat.py — Trang hỏi đáp RAG  (UI Pro Max).
"""
import reflex as rx
from frontend.state import AppState, Message, Source, Document, SessionItem
from frontend.theme import (
    BG, CARD, SIDEBAR,
    ACCENT, ACCENT_GRAD, ACCENT_GRAD2, USER_GRAD,
    TEXT, MUTED,
    BOT_BG, SUCCESS, WARNING, DANGER, BORDER, DIVIDER,
    SHADOW_LG,
)

CONTENT_WIDTH = "min(1100px, 94%)"


# ── Global CSS ─────────────────────────────────────────────────────────────────
def global_styles() -> rx.Component:
    return rx.html("""
<style>
/* ── Message fade-in ─────────────────────────────────────────────── */
@keyframes msgFadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
.rag-msg { animation: msgFadeUp 0.25s ease-out both; }

/* ── Bot card: copy button reveals on hover ──────────────────────── */
.rag-bot-wrap { position: relative; }
.rag-copy-btn {
  position: absolute;
  top: 8px; right: 8px;
  opacity: 0;
  transition: opacity 0.18s ease;
  z-index: 2;
}
.rag-bot-wrap:hover .rag-copy-btn { opacity: 1; }

/* ── Thinking dots ───────────────────────────────────────────────── */
@keyframes rag-dot-bounce {
  0%, 80%, 100% { transform: scale(0.6); opacity: 0.35; }
  40%           { transform: scale(1.1); opacity: 1; }
}
.rag-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: linear-gradient(135deg, #7c6af7, #5eead4);
  display: inline-block;
  animation: rag-dot-bounce 1.1s infinite ease-in-out;
}
.rag-dot:nth-child(2) { animation-delay: 0.15s; }
.rag-dot:nth-child(3) { animation-delay: 0.30s; }

/* ── Suggestion chip hover ───────────────────────────────────────── */
.rag-suggest {
  transition: all 0.18s ease !important;
  cursor: pointer !important;
}
.rag-suggest:hover {
  background: rgba(124,106,247,0.16) !important;
  border-color: #7c6af7 !important;
  transform: translateY(-1px);
}

/* ── Chat scrollbar ──────────────────────────────────────────────── */
#chat-scroll::-webkit-scrollbar       { width: 4px; }
#chat-scroll::-webkit-scrollbar-track { background: transparent; }
#chat-scroll::-webkit-scrollbar-thumb {
  background: rgba(124,106,247,0.28); border-radius: 4px;
}
#chat-scroll::-webkit-scrollbar-thumb:hover {
  background: rgba(124,106,247,0.55);
}

/* ── Sidebar outer scrollbar ────────────────────────────────────── */
.rag-sidebar::-webkit-scrollbar       { width: 3px; }
.rag-sidebar::-webkit-scrollbar-track { background: transparent; }
.rag-sidebar::-webkit-scrollbar-thumb {
  background: rgba(124,106,247,0.22); border-radius: 3px;
}

/* ── Session list inner scrollbar — hide native arrows ───────────── */
.rag-session-list { scrollbar-width: thin; scrollbar-color: rgba(124,106,247,0.28) transparent; }
.rag-session-list::-webkit-scrollbar          { width: 4px !important; }
.rag-session-list::-webkit-scrollbar-track    { background: transparent !important; }
.rag-session-list::-webkit-scrollbar-thumb    { background: rgba(124,106,247,0.35) !important; border-radius: 4px !important; }
.rag-session-list::-webkit-scrollbar-button   { display: none !important; width: 0 !important; height: 0 !important; }

/* ── Session item: left-border active indicator via CSS class ─────── */
.rag-sess   { border-left: 3px solid transparent !important; padding-left: 10px !important; }
.rag-sess-on{ border-left: 3px solid #7c6af7 !important; }

/* ── Textarea: remove default resize handle appearance ───────────── */
.rag-input textarea { scrollbar-width: thin; }

/* ── Fix #1: messages anchor to bottom of visible chat area ────────── */
#chat-scroll {
  display: flex;
  flex-direction: column;
}
.rag-messages-wrap {
  margin-top: auto;   /* pushes messages down; collapses when content overflows */
  width: 100%;
}

/* ── Tools-used badge (hover_card dùng Radix native, không cần CSS tooltip) ── */
/* Chỉ giữ lại transition nhẹ cho badge pill — _hover prop xử lý bg */

/* ── Markdown rendering bên trong bot bubble ─────────────────────── */
/* Radix UI reset xóa font-weight của strong/b, phải khôi phục lại   */
.rag-bot-wrap strong,
.rag-bot-wrap b { font-weight: 700 !important; }

.rag-bot-wrap em,
.rag-bot-wrap i { font-style: italic; }

.rag-bot-wrap h1,
.rag-bot-wrap h2,
.rag-bot-wrap h3,
.rag-bot-wrap h4 {
  font-weight: 700 !important;
  margin: 0.75em 0 0.35em;
  line-height: 1.35;
}
.rag-bot-wrap h1 { font-size: 1.25em; }
.rag-bot-wrap h2 { font-size: 1.15em; }
.rag-bot-wrap h3 { font-size: 1.05em; }

.rag-bot-wrap p { margin: 0.45em 0; }

.rag-bot-wrap ul {
  list-style-type: disc !important;
  padding-left: 1.6em;
  margin: 0.4em 0;
}
.rag-bot-wrap ol {
  list-style-type: decimal !important;
  padding-left: 1.6em;
  margin: 0.4em 0;
}
.rag-bot-wrap li { margin: 0.25em 0; }

.rag-bot-wrap code {
  background: rgba(124,106,247,0.10);
  padding: 0.12em 0.4em;
  border-radius: 4px;
  font-family: "Fira Mono", "Cascadia Code", monospace;
  font-size: 0.875em;
}
.rag-bot-wrap pre {
  background: rgba(124,106,247,0.07);
  border: 1px solid rgba(124,106,247,0.18);
  border-radius: 8px;
  padding: 12px 16px;
  overflow-x: auto;
  margin: 0.6em 0;
}
.rag-bot-wrap pre code { background: transparent; padding: 0; }

.rag-bot-wrap hr {
  border: none;
  border-top: 1px solid rgba(148,163,184,0.25);
  margin: 0.8em 0;
}

.rag-bot-wrap a { color: #7c6af7; text-decoration: underline; }
</style>
""")


# ── Tools-used badge ───────────────────────────────────────────────────────────
# Tên thân thiện cho từng tool (fallback = tên gốc)
_TOOL_LABELS: dict[str, str] = {
    "rag":            "📄 RAG – Tài liệu",
    "tilavy":         "🔍 Tra cứu pháp lý",
    "web_search":     "🌐 Tìm kiếm web",
    "calculator":     "🧮 Máy tính",
    "python":         "🐍 Chạy Python",
    "file_reader":    "📂 Đọc file",
    "database":       "🗄️ Cơ sở dữ liệu",
    "timelimit":      "⏱️ Thời hiệu",
    "jurisdiction":   "⚖️ Thẩm quyền tòa",
    "document_check": "📋 Kiểm tra hồ sơ",
}


def _tool_label(name: str) -> str:
    return _TOOL_LABELS.get(name, f"🔧 {name}")


def _tool_row(name: str) -> rx.Component:
    """Một dòng trong hover_card: icon + tên tool (name là Reflex Var[str])."""
    return rx.hstack(
        rx.box(
            width="7px",
            height="7px",
            border_radius="50%",
            background="linear-gradient(135deg,#7c6af7,#5eead4)",
            flex_shrink="0",
        ),
        rx.text(name, size="2", color="#e2e8f0"),
        spacing="2",
        align="center",
        padding_y="3px",
    )


def tools_badge(msg: Message) -> rx.Component:
    """
    Badge 'N tool đã dùng' dưới tin nhắn assistant khi agent được gọi.
    Hover → rx.hover_card (Radix native) liệt kê từng tool.
    """
    return rx.cond(
        msg.tools_used.length() > 0,
        rx.hover_card.root(
            rx.hover_card.trigger(
                # ── Badge pill ─────────────────────────────────────
                rx.hstack(
                    rx.icon("wrench", size=12, color="#7c6af7"),
                    rx.text(
                        msg.tools_used.length(),
                        " tool đã dùng",
                        size="1",
                        weight="bold",
                        color="#7c6af7",
                    ),
                    spacing="1",
                    align="center",
                    padding="3px 11px 3px 8px",
                    border_radius="20px",
                    background="rgba(124,106,247,0.10)",
                    border="1px solid rgba(124,106,247,0.22)",
                    cursor="default",
                    _hover={"background": "rgba(124,106,247,0.18)"},
                ),
                as_child=True,
            ),
            rx.hover_card.content(
                # ── Nội dung card ──────────────────────────────────
                rx.vstack(
                    rx.text(
                        "Tools đã sử dụng",
                        size="1",
                        weight="bold",
                        color="rgba(148,163,184,0.75)",
                        text_transform="uppercase",
                        letter_spacing="0.06em",
                        padding_bottom="4px",
                    ),
                    rx.divider(color_scheme="purple", size="4"),
                    rx.foreach(msg.tools_used, _tool_row),
                    spacing="1",
                    align="start",
                    width="100%",
                ),
                background="#1e1b3a",
                border="1px solid rgba(124,106,247,0.35)",
                border_radius="10px",
                padding="12px 16px",
                box_shadow="0 8px 32px rgba(0,0,0,0.45)",
                min_width="180px",
                side="top",
                align="start",
            ),
        ),
        rx.box(),
    )


# ── Thinking dots ──────────────────────────────────────────────────────────────
def thinking_dots() -> rx.Component:
    """3 chấm nhảy báo hiệu AI đang suy nghĩ."""
    return rx.hstack(
        rx.box(class_name="rag-dot"),
        rx.box(class_name="rag-dot"),
        rx.box(class_name="rag-dot"),
        spacing="1",
        align="center",
    )


def thinking_indicator() -> rx.Component:
    return rx.cond(
        AppState.is_loading,
        rx.hstack(
            rx.box(
                rx.icon("sparkles", size=14, color="white"),
                width="32px", height="32px",
                background=ACCENT_GRAD2,
                border_radius="50%",
                display="flex",
                align_items="center",
                justify_content="center",
                flex_shrink="0",
                box_shadow="0 2px 8px rgba(124,106,247,0.30)",
            ),
            rx.box(
                rx.hstack(
                    thinking_dots(),
                    rx.text(AppState.thinking_stage, size="2", color=MUTED, font_style="italic"),
                    spacing="3",
                    align="center",
                ),
                padding="16px 24px",
                background=BOT_BG,
                border_radius="4px 16px 16px 16px",
                border=f"1px solid {BORDER}",
            ),
            width="100%",
            max_width=CONTENT_WIDTH,
            margin="0 auto",
            padding_x="6",
            padding_y="2",
            align="center",
            spacing="3",
        ),
        rx.box(),
    )


# ── Source chips ───────────────────────────────────────────────────────────────
def source_chip(source: Source) -> rx.Component:
    """Chip nguồn với dot màu theo relevance score."""
    score_color = rx.cond(
        source.relevance_score >= 0.8,
        SUCCESS,
        rx.cond(source.relevance_score >= 0.6, WARNING, DANGER),
    )
    return rx.box(
        rx.hstack(
            rx.box(
                width="6px", height="6px",
                border_radius="50%",
                background=score_color,
                flex_shrink="0",
            ),
            rx.icon("file-text", size=11, color=ACCENT, flex_shrink="0"),
            rx.text(
                source.document_name,
                size="1",
                color=ACCENT,
                white_space="nowrap",
                overflow="hidden",
                text_overflow="ellipsis",
                min_width="0",
            ),
            spacing="2",
            align="center",
            flex_wrap="nowrap",
            width="100%",
        ),
        title=source.document_name,
        max_width="220px",
        min_width="0",
        flex_shrink="0",
        overflow="hidden",
        padding="5px 10px",
        background="rgba(124,106,247,0.10)",
        border="1px solid rgba(124,106,247,0.20)",
        border_radius="20px",
        cursor="pointer",
        _hover={"background": "rgba(124,106,247,0.22)", "border_color": ACCENT},
        transition="all 0.15s",
        on_click=AppState.open_source_chunk(
            source.document_name,
            source.chunk_index,
            source.page_number,
            source.content,
            source.relevance_score,
        ),
    )


# ── Source chunk modal ─────────────────────────────────────────────────────────
def source_chunk_modal() -> rx.Component:
    return rx.cond(
        AppState.show_source_chunk,
        rx.box(
            rx.box(
                # Header
                rx.box(
                    rx.hstack(
                        rx.vstack(
                            rx.hstack(
                                rx.box(
                                    rx.icon("file-text", size=14, color=ACCENT),
                                    width="28px", height="28px",
                                    background="rgba(124,106,247,0.14)",
                                    border_radius="8px",
                                    display="flex",
                                    align_items="center",
                                    justify_content="center",
                                ),
                                rx.text("Đoạn nguồn", size="4", weight="bold", color=TEXT),
                                spacing="2",
                                align="center",
                            ),
                            rx.hstack(
                                rx.text(AppState.source_chunk_doc_name, size="2", color=MUTED),
                                rx.text("·", size="2", color=MUTED),
                                rx.text("Chunk #" + AppState.source_chunk_index_str, size="2", color=MUTED),
                                rx.cond(
                                    AppState.source_chunk_page > 0,
                                    rx.hstack(
                                        rx.text("·", size="2", color=MUTED),
                                        rx.text("Trang " + AppState.source_chunk_page_str, size="2", color=MUTED),
                                        spacing="2",
                                    ),
                                    rx.box(),
                                ),
                                rx.text("·", size="2", color=MUTED),
                                rx.badge(
                                    "Liên quan: " + AppState.source_chunk_score_str,
                                    variant="soft",
                                    color_scheme="purple",
                                    radius="full",
                                    size="1",
                                ),
                                spacing="2",
                                flex_wrap="wrap",
                                align="center",
                            ),
                            spacing="1",
                            align="start",
                        ),
                        rx.spacer(),
                        rx.icon_button(
                            rx.icon("x", size=16),
                            variant="ghost",
                            cursor="pointer",
                            on_click=AppState.close_source_chunk,
                        ),
                        width="100%",
                        align="center",
                    ),
                    padding="5",
                    padding_bottom="4",
                    border_bottom=f"1px solid {DIVIDER}",
                    flex_shrink="0",
                ),
                # Body
                rx.box(
                    rx.text(
                        AppState.source_chunk_content,
                        size="2",
                        color=TEXT,
                        white_space="pre-wrap",
                        line_height="1.7",
                    ),
                    padding="5",
                    overflow_y="auto",
                    flex="1",
                ),
                background=CARD,
                border_radius="16px",
                width="min(720px, 92vw)",
                max_height="85vh",
                box_shadow=SHADOW_LG,
                border=f"1px solid {DIVIDER}",
                display="flex",
                flex_direction="column",
                overflow="hidden",
            ),
            position="fixed",
            top="0", left="0", right="0", bottom="0",
            background="rgba(0,0,0,0.65)",
            display="flex",
            align_items="center",
            justify_content="center",
            z_index="1000",
            padding="4",
        ),
        rx.box(),
    )


# ── Message bubbles ────────────────────────────────────────────────────────────
def user_bubble(msg: Message) -> rx.Component:
    return rx.hstack(
        rx.spacer(),
        rx.box(
            rx.text(msg.content, size="4", color="white", white_space="pre-wrap", line_height="1.65"),
            padding="16px 22px",
            background=USER_GRAD,
            border_radius="18px 18px 4px 18px",
            max_width="75%",
            box_shadow="0 2px 12px rgba(45,36,96,0.35)",
        ),
        width="100%",
        max_width=CONTENT_WIDTH,
        margin="0 auto",
        padding_x="6",
        padding_y="2",
        class_name="rag-msg",
    )


def bot_bubble(msg: Message) -> rx.Component:
    return rx.hstack(
        # Avatar
        rx.box(
            rx.icon("sparkles", size=14, color="white"),
            width="32px", height="32px",
            background=ACCENT_GRAD2,
            border_radius="50%",
            display="flex",
            align_items="center",
            justify_content="center",
            flex_shrink="0",
            box_shadow="0 2px 10px rgba(124,106,247,0.30)",
        ),
        # Content column
        rx.vstack(
            # Message card
            rx.box(
                # Bọc rx.markdown trong rx.box để tách style
                # (truyền color/font_size trực tiếp vào rx.markdown gây
                #  xung đột props ở nhiều phiên bản Reflex, fallback plain text)
                rx.box(
                    rx.markdown(msg.content),
                    color=TEXT,
                    font_size="16px",
                    line_height="1.65",
                    class_name="rag-md-body",
                ),
                # Copy button — revealed on hover via .rag-copy-btn CSS
                rx.box(
                    rx.icon_button(
                        rx.icon("copy", size=13),
                        variant="ghost",
                        size="1",
                        cursor="pointer",
                        color_scheme="gray",
                        border_radius="7px",
                        on_click=AppState.copy_msg(msg.content),
                        title="Sao chép",
                    ),
                    class_name="rag-copy-btn",
                ),
                class_name="rag-bot-wrap",
                position="relative",
                padding="20px 28px",
                background=BOT_BG,
                border_radius="4px 18px 18px 18px",
                max_width="100%",
                border=f"1px solid {BORDER}",
            ),
            # Sources row
            rx.cond(
                msg.sources.length() > 0,
                rx.hstack(
                    rx.text("Nguồn:", size="1", color=MUTED, weight="medium"),
                    rx.foreach(msg.sources, source_chip),
                    spacing="2",
                    flex_wrap="wrap",
                    padding_left="1",
                    align="center",
                ),
                rx.box(),
            ),
            # Tools-used badge (chỉ hiện khi agent dùng tool)
            tools_badge(msg),
            spacing="2",
            align="start",
            max_width="82%",
        ),
        width="100%",
        max_width=CONTENT_WIDTH,
        margin="0 auto",
        padding_x="6",
        padding_y="3",
        align="start",
        spacing="3",
        class_name="rag-msg",
    )


def message_bubble(msg: Message) -> rx.Component:
    return rx.cond(
        msg.role == "user",
        user_bubble(msg),
        rx.cond(msg.content == "", rx.box(), bot_bubble(msg)),
    )


# ── Sidebar helpers ────────────────────────────────────────────────────────────
def section_title(icon: str, title: str, right: rx.Component | None = None) -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.icon(icon, size=13, color=ACCENT),
            padding="5px",
            border_radius="7px",
            background="rgba(124,106,247,0.13)",
        ),
        rx.text(title, size="2", weight="bold", color=TEXT),
        rx.spacer(),
        right if right is not None else rx.fragment(),
        width="100%",
        align="center",
        spacing="2",
    )


def doc_filter_item(doc: Document) -> rx.Component:
    is_selected = AppState.selected_doc_ids.contains(doc.id)
    return rx.hstack(
        rx.cond(
            is_selected,
            rx.icon("square-check", size=15, color=ACCENT),
            rx.icon("square", size=15, color=MUTED),
        ),
        rx.icon("file-text", size=13, color=MUTED, flex_shrink="0"),
        rx.text(
            doc.original_filename,
            size="2",
            color=rx.cond(is_selected, TEXT, MUTED),
            no_wrap=True,
            overflow="hidden",
            text_overflow="ellipsis",
            flex="1",
            min_width="0",
        ),
        spacing="2",
        align="center",
        cursor="pointer",
        padding="8px 10px",
        border_radius="9px",
        border=rx.cond(is_selected, "1px solid rgba(124,106,247,0.40)", "1px solid transparent"),
        background=rx.cond(is_selected, "rgba(124,106,247,0.12)", "transparent"),
        _hover={"background": "rgba(124,106,247,0.08)"},
        width="100%",
        on_click=AppState.toggle_doc_selection(doc.id),
        transition="all 0.15s",
    )


def session_item(s: SessionItem) -> rx.Component:
    """Fix #2: single flat hstack; border_left replaces separate indicator box."""
    is_active = AppState.session_id == s.id
    return rx.hstack(
        rx.icon(
            "message-square",
            size=13,
            color=rx.cond(is_active, ACCENT, MUTED),
            flex_shrink="0",
        ),
        rx.text(
            s.title,
            size="2",
            color=rx.cond(is_active, TEXT, MUTED),
            weight=rx.cond(is_active, "medium", "regular"),
            no_wrap=True,
            overflow="hidden",
            text_overflow="ellipsis",
            flex="1",
            min_width="0",
        ),
        rx.box(
            rx.icon("trash-2", size=12),
            color=MUTED,
            padding="4px",
            border_radius="6px",
            cursor="pointer",
            flex_shrink="0",
            _hover={"color": DANGER, "background": "rgba(239,68,68,0.12)"},
            on_click=AppState.delete_session_item(s.id).stop_propagation,
        ),
        spacing="2",
        align="center",
        padding="7px 8px",
        border_radius="8px",
        cursor="pointer",
        class_name=rx.cond(is_active, "rag-sess rag-sess-on", "rag-sess"),
        background=rx.cond(is_active, "rgba(124,106,247,0.10)", "transparent"),
        _hover={"background": "rgba(124,106,247,0.08)"},
        width="100%",
        min_width="0",
        transition="all 0.15s",
        on_click=AppState.open_session(s.id),
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
def sidebar() -> rx.Component:
    return rx.vstack(
        # ── Brand header ──────────────────────────────────────────────
        rx.hstack(
            rx.link(
                rx.box(
                    rx.icon("arrow-left", size=15, color=MUTED),
                    padding="7px",
                    border_radius="8px",
                    cursor="pointer",
                    _hover={"background": "rgba(124,106,247,0.11)", "color": ACCENT},
                    transition="all 0.15s",
                    display="flex",
                    align_items="center",
                ),
                href="/dashboard",
            ),
            rx.hstack(
                rx.box(
                    rx.icon("sparkles", size=14, color="white"),
                    width="30px", height="30px",
                    background=ACCENT_GRAD2,
                    border_radius="9px",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    box_shadow="0 2px 8px rgba(124,106,247,0.30)",
                ),
                rx.text("RAG QA", weight="bold", size="3", color=TEXT, letter_spacing="-0.2px"),
                spacing="2",
                align="center",
            ),
            rx.spacer(),
            rx.color_mode.button(size="1", variant="ghost"),
            spacing="2",
            align="center",
            width="100%",
        ),

        # ── Cuộc trò chuyện mới ───────────────────────────────────────
        rx.button(
            rx.icon("plus", size=15),
            "Cuộc trò chuyện mới",
            size="2",
            cursor="pointer",
            width="100%",
            background=ACCENT_GRAD,
            color="white",
            border_radius="10px",
            _hover={"opacity": "0.88", "transform": "translateY(-1px)"},
            transition="all 0.18s",
            on_click=AppState.clear_chat,
        ),

        # ── Lịch sử chat ──────────────────────────────────────────────
        rx.vstack(
            section_title("clock-3", "Lịch sử chat"),
            rx.cond(
                AppState.sessions.length() > 0,
                rx.vstack(
                    rx.foreach(AppState.sessions, session_item),
                    spacing="1",
                    width="100%",
                    max_height="260px",
                    overflow_y="auto",
                    overflow_x="hidden",
                    class_name="rag-session-list",
                ),
                rx.text(
                    "Chưa có cuộc trò chuyện nào",
                    size="1", color=MUTED, font_style="italic", padding_x="4px",
                ),
            ),
            spacing="3",
            width="100%",
            padding="14px",
            border_radius="12px",
            background="rgba(124,106,247,0.04)",
            border=f"1px solid {BORDER}",
        ),

        # ── Tài liệu ──────────────────────────────────────────────────
        rx.vstack(
            section_title(
                "files",
                "Tài liệu",
                rx.badge(
                    AppState.ready_count_str + " sẵn sàng",
                    variant="soft",
                    color_scheme="violet",
                    radius="full",
                    size="1",
                ),
            ),
            rx.cond(
                AppState.doc_count > 0,
                rx.vstack(
                    rx.text(
                        "Chọn để lọc • bỏ trống = tìm tất cả",
                        size="1",
                        color=MUTED,
                        font_style="italic",
                        padding_x="4px",
                    ),
                    rx.foreach(AppState.documents, doc_filter_item),
                    spacing="1",
                    width="100%",
                    max_height="240px",
                    overflow_y="auto",
                    overflow_x="hidden",
                ),
                rx.vstack(
                    rx.icon("inbox", size=22, color=MUTED),
                    rx.text("Chưa có tài liệu", size="2", color=MUTED),
                    rx.link(
                        rx.text("Upload tài liệu →", size="2", color=ACCENT),
                        href="/dashboard",
                    ),
                    align="center",
                    spacing="2",
                    padding_y="3",
                ),
            ),
            spacing="3",
            width="100%",
            padding="14px",
            border_radius="12px",
            background="rgba(124,106,247,0.04)",
            border=f"1px solid {BORDER}",
        ),

        rx.spacer(),

        width="276px",
        min_width="276px",
        height="100vh",
        background=SIDEBAR,
        padding="18px 14px",
        border_right=f"1px solid {BORDER}",
        spacing="4",
        position="sticky",
        top="0",
        overflow_y="auto",
        overflow_x="hidden",
        class_name="rag-sidebar",
    )


# ── Context meter ──────────────────────────────────────────────────────────────
def context_meter() -> rx.Component:
    ring = rx.box(
        rx.box(width="14px", height="14px", border_radius="50%", background=BG),
        width="26px", height="26px",
        border_radius="50%",
        background=AppState.ctx_ring,
        display="flex",
        align_items="center",
        justify_content="center",
        flex_shrink="0",
        transition="background 0.4s",
    )
    return rx.tooltip(
        rx.vstack(
            ring,
            rx.text(
                rx.cond(AppState.ctx_compressed, "nén✓", AppState.ctx_label),
                size="1",
                weight="bold",
                color=AppState.ctx_color,
                line_height="1",
                white_space="nowrap",
            ),
            spacing="1",
            align="center",
            justify="center",
            width="52px",
            min_width="52px",
            height="54px",
            border_radius="13px",
            border=f"1px solid {BORDER}",
            background="rgba(124,106,247,0.06)",
            cursor="default",
            align_self="end",
        ),
        content=AppState.ctx_detail,
    )


# ── Empty state ────────────────────────────────────────────────────────────────
def chat_empty_state() -> rx.Component:
    def suggest(icon_name: str, label: str, q: str) -> rx.Component:
        return rx.hstack(
            rx.icon(icon_name, size=15, color=ACCENT, flex_shrink="0"),
            rx.text(label, size="2", color=TEXT, line_height="1.3"),
            spacing="3",
            align="center",
            padding="12px 16px",
            border_radius="12px",
            border=f"1px solid {BORDER}",
            background=BOT_BG,
            width="100%",
            class_name="rag-suggest",
            on_click=AppState.set_query_suggest(q),
        )

    return rx.vstack(
        # Hero icon with glow
        rx.box(
            rx.icon("message-circle", size=30, color="white"),
            width="68px", height="68px",
            background=ACCENT_GRAD2,
            border_radius="22px",
            display="flex",
            align_items="center",
            justify_content="center",
            box_shadow="0 8px 32px rgba(124,106,247,0.38)",
        ),
        # Headline
        rx.vstack(
            rx.text(
                "Hỏi về tài liệu của bạn",
                size="6", weight="bold", color=TEXT,
                text_align="center",
                letter_spacing="-0.3px",
            ),
            rx.text(
                "AI tìm kiếm trong tài liệu và trả lời kèm nguồn trích dẫn",
                size="3", color=MUTED,
                text_align="center",
                line_height="1.5",
            ),
            spacing="2",
            align="center",
        ),
        # Suggestion grid 2×2
        rx.vstack(
            rx.hstack(
                suggest("file-text", "Tóm tắt nội dung chính",
                        "Tóm tắt nội dung chính của tài liệu?"),
                suggest("search",    "Tìm thông tin cụ thể",
                        "Tìm thông tin về..."),
                spacing="3",
                width="100%",
            ),
            rx.hstack(
                suggest("help-circle", "Giải thích khái niệm",
                        "Giải thích khái niệm... trong tài liệu?"),
                suggest("star",        "Các điểm quan trọng",
                        "Các điểm quan trọng cần chú ý là gì?"),
                spacing="3",
                width="100%",
            ),
            spacing="3",
            width="100%",
            max_width="460px",
        ),
        spacing="6",
        align="center",
        justify="center",
        height="100%",
        padding_x="6",
        padding_y="10",
    )


# ── Chat area ──────────────────────────────────────────────────────────────────
def chat_area() -> rx.Component:
    return rx.vstack(
        # Messages
        rx.box(
            rx.cond(
                AppState.has_messages,
                rx.vstack(
                    rx.foreach(AppState.messages, message_bubble),
                    thinking_indicator(),
                    spacing="1",
                    width="100%",
                    padding_y="5",
                    class_name="rag-messages-wrap",   # Fix #1
                ),
                chat_empty_state(),
            ),
            id="chat-scroll",
            flex="1",
            overflow_y="auto",
            width="100%",
        ),

        # ── Input bar ─────────────────────────────────────────────────
        rx.box(
            rx.form(
                rx.hstack(
                    context_meter(),
                    rx.text_area(
                        value=AppState.query,
                        on_change=AppState.set_query,
                        placeholder="Nhập câu hỏi… (Enter gửi · Shift+Enter xuống dòng)",
                        enter_key_submit=True,
                        rows="1",
                        min_height="54px",
                        max_height="200px",
                        resize="none",
                        flex="1",
                        background="rgba(124,106,247,0.06)",
                        border=f"1px solid {BORDER}",
                        border_radius="14px",
                        color=TEXT,
                        _placeholder={"color": MUTED},
                        _focus={
                            "border_color": ACCENT,
                            "outline": "none",
                            "box_shadow": "0 0 0 3px rgba(124,106,247,0.15)",
                        },
                        font_size="15px",
                        padding="16px 20px",
                        class_name="rag-input",
                        transition="border-color 0.2s, box-shadow 0.2s",
                    ),
                    rx.cond(
                        AppState.is_generating,
                        rx.button(
                            rx.icon("square", size=17),
                            type="button",
                            on_click=AppState.stop_generation,
                            background=DANGER,
                            border_radius="13px",
                            width="54px", height="54px",
                            cursor="pointer",
                            align_self="end",
                            _hover={"opacity": "0.85"},
                            title="Dừng tạo",
                        ),
                        rx.button(
                            rx.icon("send", size=18),
                            type="submit",
                            background=ACCENT_GRAD,
                            border_radius="13px",
                            width="54px", height="54px",
                            cursor="pointer",
                            align_self="end",
                            _hover={"opacity": "0.88", "transform": "scale(1.04)"},
                            transition="all 0.18s",
                            title="Gửi (Enter)",
                        ),
                    ),
                    spacing="3",
                    align="end",
                    width="100%",
                    max_width=CONTENT_WIDTH,
                    margin="0 auto",
                ),
                on_submit=AppState.send_query,
                reset_on_submit=False,
                width="100%",
            ),
            padding="14px 20px",
            border_top=f"1px solid {BORDER}",
            background=BG,
            width="100%",
            box_shadow="0 -8px 28px rgba(0,0,0,0.13)",   # Fix #3
        ),

        height="100vh",
        flex="1",
        spacing="0",
        overflow="hidden",
    )


# ── Page ───────────────────────────────────────────────────────────────────────
def chat_page() -> rx.Component:
    return rx.hstack(
        global_styles(),
        source_chunk_modal(),
        sidebar(),
        chat_area(),
        background=BG,
        spacing="0",
        height="100vh",
        overflow="hidden",
        on_mount=[AppState.load_documents, AppState.load_sessions],
    )
