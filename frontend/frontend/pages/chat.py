"""
pages/chat.py — Trang hỏi đáp RAG.
"""
import reflex as rx
from frontend.state import AppState, Message, Source, Document

BG = "#0f1117"
CARD = "#1a1d27"
ACCENT = "#7c6af7"
ACCENT2 = "#5eead4"
TEXT = "#e2e8f0"
MUTED = "#64748b"
USER_BG = "#2d2460"
BOT_BG = "#1e2235"
SUCCESS = "#22c55e"
DANGER = "#ef4444"

# Chiều rộng cột chat trung tâm — giống bố cục Claude (nội dung căn giữa, có lề 2 bên)
CONTENT_WIDTH = "820px"


def thinking_dots() -> rx.Component:
    """3 chấm nhảy nhảy báo hiệu AI đang suy nghĩ."""
    return rx.fragment(
        rx.html(
            """
            <style>
            @keyframes rag-dot-bounce {
              0%, 80%, 100% { transform: scale(0.6); opacity: 0.35; }
              40% { transform: scale(1); opacity: 1; }
            }
            .rag-dot {
              width: 7px; height: 7px; border-radius: 50%;
              background: linear-gradient(135deg, #7c6af7, #5eead4);
              display: inline-block;
              animation: rag-dot-bounce 1.1s infinite ease-in-out;
            }
            .rag-dot:nth-child(2) { animation-delay: 0.15s; }
            .rag-dot:nth-child(3) { animation-delay: 0.3s; }
            </style>
            """
        ),
        rx.hstack(
            rx.box(class_name="rag-dot"),
            rx.box(class_name="rag-dot"),
            rx.box(class_name="rag-dot"),
            spacing="1",
            align="center",
        ),
    )


def thinking_indicator() -> rx.Component:
    return rx.cond(
        AppState.is_loading,
        rx.hstack(
            rx.box(
                rx.text("AI", size="1", color="white", weight="bold"),
                width="28px",
                height="28px",
                background=f"linear-gradient(135deg, {ACCENT}, {ACCENT2})",
                border_radius="50%",
                display="flex",
                align_items="center",
                justify_content="center",
                flex_shrink="0",
            ),
            rx.box(
                rx.hstack(
                    thinking_dots(),
                    rx.text(AppState.thinking_stage, size="2", color=MUTED),
                    spacing="3",
                    align="center",
                ),
                padding="4",
                background=BOT_BG,
                border_radius="4px 16px 16px 16px",
                border="1px solid rgba(255,255,255,0.06)",
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


def source_chip(source: Source) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.icon("file-text", size=10, color=ACCENT),
            rx.text(
                source.document_name,
                size="1",
                color=ACCENT,
                no_wrap=True,
                max_width="120px",
                overflow="hidden",
                text_overflow="ellipsis",
            ),
            spacing="1",
            align="center",
        ),
        padding_x="2",
        padding_y="1",
        background="rgba(124,106,247,0.12)",
        border="1px solid rgba(124,106,247,0.25)",
        border_radius="20px",
    )


def user_bubble(msg: Message) -> rx.Component:
    return rx.hstack(
        rx.spacer(),
        rx.box(
            rx.text(msg.content, size="4", color="white", white_space="pre-wrap", line_height="1.6"),
            padding="4",
            padding_x="5",
            background=f"linear-gradient(135deg, {USER_BG}, #3b2d7e)",
            border_radius="18px 18px 4px 18px",
            max_width="75%",
            box_shadow="0 2px 8px rgba(0,0,0,0.3)",
        ),
        width="100%",
        max_width=CONTENT_WIDTH,
        margin="0 auto",
        padding_x="6",
        padding_y="3",
    )


def bot_bubble(msg: Message) -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.text("AI", size="1", color="white", weight="bold"),
            width="28px",
            height="28px",
            background=f"linear-gradient(135deg, {ACCENT}, {ACCENT2})",
            border_radius="50%",
            display="flex",
            align_items="center",
            justify_content="center",
            flex_shrink="0",
        ),
        rx.vstack(
            rx.box(
                rx.text(msg.content, size="4", color=TEXT, white_space="pre-wrap", line_height="1.6"),
                padding="4",
                padding_x="5",
                background=BOT_BG,
                border_radius="4px 18px 18px 18px",
                max_width="100%",
                border="1px solid rgba(255,255,255,0.06)",
            ),
            rx.cond(
                msg.sources.length() > 0,
                rx.hstack(
                    rx.text("Nguồn:", size="1", color=MUTED),
                    rx.foreach(msg.sources, source_chip),
                    spacing="2",
                    flex_wrap="wrap",
                    padding_left="1",
                ),
                rx.box(),
            ),
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
    )


def message_bubble(msg: Message) -> rx.Component:
    return rx.cond(
        msg.role == "user",
        user_bubble(msg),
        bot_bubble(msg),
    )


def doc_filter_item(doc: Document) -> rx.Component:
    is_selected = AppState.selected_doc_ids.contains(doc.id)
    return rx.hstack(
        rx.cond(
            is_selected,
            rx.icon("square-check", size=14, color=ACCENT),
            rx.icon("square", size=14, color=MUTED),
        ),
        rx.text(
            doc.original_filename,
            size="1",
            color=rx.cond(is_selected, TEXT, MUTED),
            no_wrap=True,
            overflow="hidden",
            text_overflow="ellipsis",
            max_width="155px",
        ),
        spacing="2",
        align="center",
        cursor="pointer",
        padding="2",
        border_radius="6px",
        background=rx.cond(is_selected, "rgba(124,106,247,0.15)", "transparent"),
        _hover={"background": "rgba(124,106,247,0.08)"},
        width="100%",
        on_click=AppState.toggle_doc_selection(doc.id),
    )


def sidebar() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.link(
                rx.icon("arrow-left", size=16, color=MUTED),
                href="/dashboard",
            ),
            rx.text("RAG QA", weight="bold", size="3", color=TEXT),
            spacing="3",
            align="center",
            width="100%",
        ),

        rx.box(height="1px", width="100%", background="rgba(255,255,255,0.08)"),

        rx.vstack(
            rx.hstack(
                rx.icon("filter", size=14, color=ACCENT),
                rx.text("Tài liệu", size="2", weight="medium", color=TEXT),
                rx.spacer(),
                rx.text(AppState.ready_count_str + " sẵn sàng", size="1", color=MUTED),
                width="100%",
                align="center",
            ),
            rx.cond(
                AppState.doc_count > 0,
                rx.vstack(
                    rx.text(
                        "Chọn để lọc (bỏ trống = tìm tất cả)",
                        size="1",
                        color=MUTED,
                        font_style="italic",
                    ),
                    rx.foreach(AppState.documents, doc_filter_item),
                    spacing="1",
                    width="100%",
                    max_height="380px",
                    overflow_y="auto",
                ),
                rx.vstack(
                    rx.icon("inbox", size=24, color=MUTED),
                    rx.text("Chưa có tài liệu", size="2", color=MUTED),
                    rx.link(
                        rx.text("Upload tài liệu →", size="2", color=ACCENT),
                        href="/dashboard",
                    ),
                    align="center",
                    spacing="2",
                    padding_y="4",
                ),
            ),
            spacing="3",
            width="100%",
        ),

        rx.spacer(),

        rx.button(
            rx.icon("trash-2", size=14),
            "Xóa lịch sử chat",
            variant="ghost",
            size="2",
            color=MUTED,
            cursor="pointer",
            width="100%",
            on_click=AppState.clear_chat,
        ),

        width="220px",
        min_width="220px",
        height="100vh",
        background=CARD,
        padding="5",
        border_right="1px solid rgba(255,255,255,0.06)",
        spacing="4",
        position="sticky",
        top="0",
        overflow_y="auto",
    )


def chat_area() -> rx.Component:
    return rx.vstack(
        # Tin nhắn
        rx.box(
            rx.cond(
                AppState.has_messages,
                rx.vstack(
                    rx.foreach(AppState.messages, message_bubble),
                    thinking_indicator(),
                    spacing="2",
                    width="100%",
                    padding_y="5",
                ),
                rx.vstack(
                    rx.box(
                        rx.icon("message-circle", size=52, color=ACCENT),
                        opacity="0.5",
                    ),
                    rx.text(
                        "Hỏi bất cứ điều gì về tài liệu của bạn",
                        size="5",
                        weight="medium",
                        color=TEXT,
                    ),
                    rx.text(
                        "AI sẽ tìm kiếm trong tài liệu và trả lời kèm nguồn trích dẫn",
                        size="3",
                        color=MUTED,
                        text_align="center",
                    ),
                    rx.hstack(
                        rx.box(
                            rx.text("Tóm tắt nội dung chính?", size="2", color=ACCENT),
                            padding="2",
                            padding_x="3",
                            border_radius="20px",
                            border="1px solid rgba(124,106,247,0.3)",
                            cursor="pointer",
                            _hover={"background": "rgba(124,106,247,0.1)"},
                            on_click=AppState.set_query_suggest(
                                "Tóm tắt nội dung chính của tài liệu?"
                            ),
                        ),
                        rx.box(
                            rx.text("Các điểm quan trọng?", size="2", color=ACCENT),
                            padding="2",
                            padding_x="3",
                            border_radius="20px",
                            border="1px solid rgba(124,106,247,0.3)",
                            cursor="pointer",
                            _hover={"background": "rgba(124,106,247,0.1)"},
                            on_click=AppState.set_query_suggest(
                                "Các điểm quan trọng cần chú ý là gì?"
                            ),
                        ),
                        spacing="3",
                        flex_wrap="wrap",
                        justify="center",
                    ),
                    spacing="4",
                    align="center",
                    justify="center",
                    height="100%",
                    padding="8",
                ),
            ),
            flex="1",
            overflow_y="auto",
            width="100%",
        ),

        # Input box — to hơn, giống khung chat Claude
        rx.box(
            rx.hstack(
                rx.text_area(
                    value=AppState.query,
                    on_change=AppState.set_query,
                    placeholder="Nhập câu hỏi... (Enter để gửi, Shift+Enter xuống dòng)",
                    on_key_down=AppState.handle_enter,
                    rows="1",
                    min_height="56px",
                    max_height="200px",
                    resize="none",
                    flex="1",
                    background="rgba(255,255,255,0.05)",
                    border="1px solid rgba(255,255,255,0.1)",
                    border_radius="16px",
                    color=TEXT,
                    _placeholder={"color": MUTED},
                    _focus={"border_color": ACCENT, "outline": "none"},
                    font_size="16px",
                    padding="4",
                ),
                rx.button(
                    rx.cond(
                        AppState.is_loading,
                        rx.spinner(size="2"),
                        rx.icon("send", size=20),
                    ),
                    on_click=AppState.send_query,
                    background=f"linear-gradient(135deg, {ACCENT}, #9d8df7)",
                    border_radius="14px",
                    width="56px",
                    height="56px",
                    cursor="pointer",
                    disabled=AppState.is_loading,
                    align_self="end",
                ),
                spacing="3",
                align="end",
                width="100%",
                max_width=CONTENT_WIDTH,
                margin="0 auto",
            ),
            padding="5",
            padding_x="6",
            border_top="1px solid rgba(255,255,255,0.06)",
            background=BG,
            width="100%",
        ),

        height="100vh",
        flex="1",
        spacing="0",
        overflow="hidden",
    )


def chat_page() -> rx.Component:
    return rx.hstack(
        sidebar(),
        chat_area(),
        background=BG,
        spacing="0",
        height="100vh",
        overflow="hidden",
        on_mount=AppState.load_documents,
    )
