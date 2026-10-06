"""
pages/chat.py — Trang hỏi đáp RAG.
"""
import reflex as rx
from frontend.state import AppState, Message, Source, Document, SessionItem
from frontend.theme import (
    BG, CARD, ACCENT, ACCENT2, TEXT, MUTED, USER_BG, BOT_BG, SUCCESS, DANGER, BORDER, DIVIDER,
)

# Chiều rộng cột chat trung tâm — đủ rộng để không bị bó hẹp/thừa khoảng trống
# 2 bên trên màn hình rộng, nhưng vẫn có giới hạn để dòng chữ không quá dài.
CONTENT_WIDTH = "min(1200px, 94%)"


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
                padding="20px 28px",
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


def source_chip(source: Source) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.icon("file-text", size=12, color=ACCENT, flex_shrink="0"),
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
        max_width="240px",
        min_width="0",
        flex_shrink="0",
        overflow="hidden",
        padding="5px 12px",
        background="rgba(124,106,247,0.12)",
        border="1px solid rgba(124,106,247,0.25)",
        border_radius="20px",
        cursor="pointer",
        _hover={"background": "rgba(124,106,247,0.25)", "border_color": ACCENT},
        transition="all 0.15s",
        on_click=AppState.open_source_chunk(
            source.document_name,
            source.chunk_index,
            source.page_number,
            source.content,
            source.relevance_score,
        ),
    )


def source_chunk_modal() -> rx.Component:
    """Modal xem toàn bộ nội dung 1 chunk nguồn được click từ câu trả lời."""
    return rx.cond(
        AppState.show_source_chunk,
        rx.box(
            rx.box(
                rx.box(
                    rx.hstack(
                        rx.vstack(
                            rx.text("Đoạn nguồn", size="4", weight="bold", color=TEXT),
                            rx.hstack(
                                rx.text(AppState.source_chunk_doc_name, size="2", color=MUTED),
                                rx.text("•", size="2", color=MUTED),
                                rx.text("Chunk #" + AppState.source_chunk_index_str, size="2", color=MUTED),
                                rx.cond(
                                    AppState.source_chunk_page > 0,
                                    rx.hstack(
                                        rx.text("•", size="2", color=MUTED),
                                        rx.text("Trang " + AppState.source_chunk_page_str, size="2", color=MUTED),
                                        spacing="2",
                                    ),
                                    rx.box(),
                                ),
                                rx.text("•", size="2", color=MUTED),
                                rx.badge(
                                    "Độ liên quan " + AppState.source_chunk_score_str,
                                    variant="soft",
                                    color_scheme="purple",
                                    radius="full",
                                ),
                                spacing="2",
                                flex_wrap="wrap",
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
                    padding="6",
                    padding_bottom="4",
                    border_bottom=f"1px solid {DIVIDER}",
                    flex_shrink="0",
                ),
                rx.box(
                    rx.text(
                        AppState.source_chunk_content,
                        size="2",
                        color=TEXT,
                        white_space="pre-wrap",
                        line_height="1.6",
                    ),
                    padding="6",
                    overflow_y="auto",
                    flex="1",
                ),
                background=CARD,
                border_radius="16px",
                width="min(720px, 92vw)",
                max_height="85vh",
                box_shadow="0 20px 60px rgba(0,0,0,0.5)",
                border=f"1px solid {DIVIDER}",
                display="flex",
                flex_direction="column",
                overflow="hidden",
            ),
            position="fixed",
            top="0",
            left="0",
            right="0",
            bottom="0",
            background="rgba(0,0,0,0.6)",
            display="flex",
            align_items="center",
            justify_content="center",
            z_index="1000",
            padding="4",
        ),
        rx.box(),
    )


def user_bubble(msg: Message) -> rx.Component:
    return rx.hstack(
        rx.spacer(),
        rx.box(
            rx.text(msg.content, size="4", color="white", white_space="pre-wrap", line_height="1.6"),
            padding="20px 28px",
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
                rx.markdown(
                    msg.content,
                    color=TEXT,
                    font_size="16px",
                    line_height="1.6",
                ),
                padding="20px 28px",
                background=BOT_BG,
                border_radius="4px 18px 18px 18px",
                max_width="100%",
                border=f"1px solid {BORDER}",
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
        # Tin nhắn AI còn rỗng (placeholder lúc đang suy nghĩ) → không vẽ avatar/khung trống;
        # chỉ hiện thanh "Đang phân tích…" bên dưới, tới khi có chữ đầu tiên mới hiện bong bóng.
        rx.cond(msg.content == "", rx.box(), bot_bubble(msg)),
    )


def section_title(icon: str, title: str, right: rx.Component | None = None) -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.icon(icon, size=14, color=ACCENT),
            padding="6px",
            border_radius="8px",
            background="rgba(124,106,247,0.14)",
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
        rx.icon("file-text", size=14, color=MUTED),
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
        padding="9px 12px",
        border_radius="10px",
        border=rx.cond(is_selected, "1px solid rgba(124,106,247,0.45)", "1px solid transparent"),
        background=rx.cond(is_selected, "rgba(124,106,247,0.14)", "transparent"),
        _hover={"background": "rgba(124,106,247,0.09)"},
        width="100%",
        on_click=AppState.toggle_doc_selection(doc.id),
    )


def session_item(s: SessionItem) -> rx.Component:
    is_active = AppState.session_id == s.id
    return rx.hstack(
        rx.icon("message-square", size=14, color=rx.cond(is_active, ACCENT, MUTED)),
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
            rx.icon("trash-2", size=13),
            color=MUTED,
            padding="4px",
            border_radius="6px",
            cursor="pointer",
            _hover={"color": DANGER, "background": "rgba(239,68,68,0.12)"},
            on_click=AppState.delete_session_item(s.id).stop_propagation,
        ),
        spacing="2",
        align="center",
        padding="9px 10px 9px 12px",
        border_radius="10px",
        cursor="pointer",
        border=rx.cond(is_active, "1px solid rgba(124,106,247,0.45)", "1px solid transparent"),
        background=rx.cond(is_active, "rgba(124,106,247,0.14)", "transparent"),
        _hover={"background": "rgba(124,106,247,0.09)"},
        width="100%",
        on_click=AppState.open_session(s.id),
    )


def sidebar() -> rx.Component:
    return rx.vstack(
        # ── Header ──
        rx.hstack(
            rx.link(
                rx.box(
                    rx.icon("arrow-left", size=16, color=MUTED),
                    padding="7px",
                    border_radius="8px",
                    _hover={"background": "rgba(124,106,247,0.12)"},
                ),
                href="/dashboard",
            ),
            rx.text("RAG QA", weight="bold", size="4", color=TEXT),
            rx.spacer(),
            rx.color_mode.button(size="1", variant="ghost"),
            spacing="3",
            align="center",
            width="100%",
        ),

        # ── Cuộc trò chuyện mới ──
        rx.button(
            rx.icon("plus", size=16),
            "Cuộc trò chuyện mới",
            size="3",
            cursor="pointer",
            width="100%",
            color_scheme="violet",
            on_click=AppState.clear_chat,
        ),

        # ── Lịch sử chat ──
        rx.vstack(
            section_title("history", "Lịch sử chat"),
            rx.cond(
                AppState.sessions.length() > 0,
                rx.vstack(
                    rx.foreach(AppState.sessions, session_item),
                    spacing="1",
                    width="100%",
                    max_height="280px",
                    overflow_y="auto",
                    overflow_x="hidden",
                ),
                rx.text("Chưa có cuộc trò chuyện nào", size="1", color=MUTED, font_style="italic", padding_x="4px"),
            ),
            spacing="3",
            width="100%",
            padding="14px",
            border_radius="14px",
            background="rgba(124,106,247,0.05)",
            border=f"1px solid {BORDER}",
        ),

        # ── Tài liệu ──
        rx.vstack(
            section_title(
                "files",
                "Tài liệu",
                rx.text(AppState.ready_count_str + " sẵn sàng", size="1", color=MUTED),
            ),
            rx.cond(
                AppState.doc_count > 0,
                rx.vstack(
                    rx.text(
                        "Chọn để lọc (bỏ trống = tìm tất cả)",
                        size="1",
                        color=MUTED,
                        font_style="italic",
                        padding_x="4px",
                    ),
                    rx.foreach(AppState.documents, doc_filter_item),
                    spacing="1",
                    width="100%",
                    max_height="260px",
                    overflow_y="auto",
                    overflow_x="hidden",
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
            padding="14px",
            border_radius="14px",
            background="rgba(124,106,247,0.05)",
            border=f"1px solid {BORDER}",
        ),

        rx.spacer(),

        width="280px",
        min_width="280px",
        height="100vh",
        background=CARD,
        padding="20px 16px",
        border_right=f"1px solid {BORDER}",
        spacing="4",
        position="sticky",
        top="0",
        overflow_y="auto",
        overflow_x="hidden",
    )


def context_meter() -> rx.Component:
    """Mức đầy bộ nhớ hội thoại — gọn, nằm cùng hàng bên trái ô nhập: vòng tròn tô dần theo %, đổi màu khi gần nén."""
    ring = rx.box(
        rx.box(width="14px", height="14px", border_radius="50%", background=BG),
        width="26px",
        height="26px",
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
                rx.cond(AppState.ctx_compressed, "đã nén", AppState.ctx_label),
                size="1",
                weight="bold",
                color=AppState.ctx_color,
                line_height="1",
                white_space="nowrap",
            ),
            spacing="1",
            align="center",
            justify="center",
            width="54px",
            min_width="54px",
            height="56px",
            border_radius="14px",
            border=f"1px solid {BORDER}",
            background="rgba(124,106,247,0.06)",
            cursor="default",
            align_self="end",
        ),
        content=AppState.ctx_detail,
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
            id="chat-scroll",
            flex="1",
            overflow_y="auto",
            width="100%",
        ),

        # Input box — to hơn, giống khung chat Claude
        rx.box(
            rx.form(
                rx.hstack(
                    context_meter(),
                    rx.text_area(
                        value=AppState.query,
                        on_change=AppState.set_query,
                        placeholder="Nhập câu hỏi... (Enter để gửi, Shift+Enter xuống dòng)",
                        enter_key_submit=True,
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
                        padding="18px 20px",
                    ),
                    rx.cond(
                        AppState.is_generating,
                        # ── Đang xử lý/sinh câu trả lời → nút Dừng ───────
                        rx.button(
                            rx.icon("square", size=18),
                            type="button",
                            on_click=AppState.stop_generation,
                            background=DANGER,
                            border_radius="14px",
                            width="56px",
                            height="56px",
                            cursor="pointer",
                            align_self="end",
                        ),
                        # ── Bình thường → nút Gửi ─────────────────────────
                        rx.button(
                            rx.icon("send", size=20),
                            type="submit",
                            background=f"linear-gradient(135deg, {ACCENT}, #9d8df7)",
                            border_radius="14px",
                            width="56px",
                            height="56px",
                            cursor="pointer",
                            align_self="end",
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
            padding="5",
            padding_x="6",
            border_top=f"1px solid {BORDER}",
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
        source_chunk_modal(),
        sidebar(),
        chat_area(),
        background=BG,
        spacing="0",
        height="100vh",
        overflow="hidden",
        on_mount=[AppState.load_documents, AppState.load_sessions],
    )
