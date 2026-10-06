"""
pages/dashboard.py — Trang quản lý tài liệu.
"""
import reflex as rx
from frontend.state import AppState, Chunk, Document
from frontend.theme import (
    BG, CARD, ACCENT, ACCENT2, TEXT, MUTED, SUCCESS, WARNING, DANGER, BORDER, DIVIDER,
)


def status_badge(doc: Document) -> rx.Component:
    return rx.cond(
        doc.status == "completed",
        rx.badge("✓ Sẵn sàng", color_scheme="green", variant="soft", radius="full"),
        rx.cond(
            doc.status == "failed",
            rx.badge("✗ Lỗi", color_scheme="red", variant="soft", radius="full"),
            rx.badge("⏳ Đang xử lý", color_scheme="yellow", variant="soft", radius="full"),
        ),
    )


def stat_card(label: str, value: str, icon: str, color: str) -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.icon(icon, size=18, color=color),
            width="38px",
            height="38px",
            border_radius="10px",
            background=f"{color}1f",
            display="flex",
            align_items="center",
            justify_content="center",
            flex_shrink="0",
        ),
        rx.vstack(
            rx.text(value, size="5", weight="bold", color=TEXT, line_height="1.1"),
            rx.text(label, size="1", color=MUTED, weight="medium"),
            spacing="0",
            align="start",
        ),
        spacing="3",
        align="center",
        padding="18px 24px",
        border_radius="12px",
        background=CARD,
        border=f"1px solid {BORDER}",
        flex="1",
        min_width="170px",
    )


def doc_row(doc: Document) -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.icon("file-text", size=16, color=ACCENT),
            width="36px",
            height="36px",
            border_radius="10px",
            background=f"{ACCENT}1f",
            display="flex",
            align_items="center",
            justify_content="center",
            flex_shrink="0",
        ),
        rx.vstack(
            rx.text(
                doc.original_filename,
                size="2",
                weight="medium",
                color=TEXT,
                no_wrap=True,
                overflow="hidden",
                text_overflow="ellipsis",
                max_width="360px",
            ),
            rx.text(doc.file_extension, size="1", color=MUTED),
            spacing="0",
            align="start",
        ),
        rx.spacer(),
        status_badge(doc),
        rx.cond(
            doc.status == "completed",
            rx.icon_button(
                rx.icon("layers", size=14),
                variant="ghost",
                color_scheme="purple",
                size="1",
                cursor="pointer",
                on_click=AppState.open_chunks(doc.id, doc.original_filename),
            ),
            rx.box(),
        ),
        rx.icon_button(
            rx.icon("trash-2", size=14),
            variant="ghost",
            color_scheme="red",
            size="1",
            cursor="pointer",
            on_click=AppState.delete_document(doc.id),
        ),
        width="100%",
        padding="16px 20px",
        border_radius="12px",
        border="1px solid transparent",
        transition="all 0.15s ease",
        _hover={"background": "rgba(124,106,247,0.08)", "border_color": BORDER},
        align="center",
        spacing="3",
    )


def chunk_card(c: Chunk) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.badge(f"#{c.chunk_index}", variant="soft", color_scheme="purple", radius="full"),
            rx.cond(
                c.chapter != "",
                rx.badge(c.chapter, variant="soft", color_scheme="cyan", radius="full"),
                rx.box(),
            ),
            rx.cond(
                c.article_number != "",
                rx.badge(f"Điều {c.article_number}", variant="soft", color_scheme="cyan", radius="full"),
                rx.box(),
            ),
            rx.cond(
                c.page_number > 0,
                rx.text(f"Trang {c.page_number}", size="1", color=MUTED),
                rx.box(),
            ),
            spacing="2",
            align="center",
            flex_wrap="wrap",
        ),
        rx.cond(
            c.article_title != "",
            rx.text(c.article_title, size="2", weight="medium", color=TEXT, margin_top="2"),
            rx.box(),
        ),
        rx.text(
            c.content,
            size="2",
            color=TEXT,
            white_space="pre-wrap",
            margin_top="2",
            line_height="1.5",
        ),
        padding="5",
        border_radius="10px",
        background=BG,
        border=f"1px solid {BORDER}",
        width="100%",
    )


def chunk_viewer_modal() -> rx.Component:
    return rx.cond(
        AppState.show_chunks,
        rx.box(
            rx.box(
                # ── Header cố định (không cuộn) ──────────────────────
                rx.box(
                    rx.hstack(
                        rx.vstack(
                            rx.text("Dữ liệu sau khi chunking", size="4", weight="bold", color=TEXT),
                            rx.hstack(
                                rx.text(AppState.chunks_doc_name, size="2", color=MUTED),
                                rx.text("•", size="2", color=MUTED),
                                rx.text(AppState.chunks_count_str + " chunk", size="2", color=MUTED),
                                spacing="2",
                            ),
                            spacing="0",
                            align="start",
                        ),
                        rx.spacer(),
                        rx.icon_button(
                            rx.icon("x", size=16),
                            variant="ghost",
                            cursor="pointer",
                            on_click=AppState.close_chunks,
                        ),
                        width="100%",
                        align="center",
                    ),
                    padding="6",
                    padding_bottom="3",
                    border_bottom=f"1px solid {DIVIDER}",
                    flex_shrink="0",
                ),
                # ── Nội dung cuộn được ────────────────────────────────
                rx.box(
                    rx.cond(
                        AppState.chunks_loading,
                        rx.center(rx.spinner(size="3"), padding="8", width="100%"),
                        rx.cond(
                            AppState.chunks.length() > 0,
                            rx.vstack(
                                rx.foreach(AppState.chunks, chunk_card),
                                spacing="3",
                                width="100%",
                                padding_top="4",
                            ),
                            rx.center(
                                rx.text("Chưa có chunk nào.", color=MUTED),
                                padding="8",
                                width="100%",
                            ),
                        ),
                    ),
                    padding="6",
                    padding_top="0",
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


def upload_zone() -> rx.Component:
    # Thanh upload GỌN theo chiều ngang (thay vì 1 khối vuông cao trống trải)
    # — upload là thao tác không thường xuyên nên không cần chiếm nhiều diện
    # tích, phần còn lại của trang dành cho danh sách tài liệu (nội dung
    # chính, dùng thường xuyên hơn).
    return rx.upload(
        rx.hstack(
            rx.cond(
                AppState.is_uploading,
                rx.spinner(size="3"),
                rx.icon("cloud-upload", size=26, color=ACCENT),
            ),
            rx.vstack(
                rx.text(
                    "Kéo thả file vào đây, hoặc click để chọn file",
                    size="3",
                    weight="medium",
                    color=TEXT,
                ),
                rx.text("PDF, DOCX, DOC, PPTX, TXT", size="1", color=MUTED),
                spacing="0",
                align="start",
            ),
            rx.spacer(),
            rx.cond(
                AppState.upload_msg != "",
                rx.text(
                    AppState.upload_msg,
                    size="2",
                    color=rx.cond(AppState.upload_ok, SUCCESS, DANGER),
                    weight="medium",
                ),
                rx.box(),
            ),
            spacing="4",
            align="center",
            width="100%",
        ),
        id="upload",
        accept={
            "application/pdf": [".pdf"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
            "application/msword": [".doc"],
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
            "text/plain": [".txt"],
        },
        max_files=1,
        on_drop=AppState.handle_upload(rx.upload_files(upload_id="upload")),
        border="2px dashed rgba(124,106,247,0.4)",
        border_radius="12px",
        background="rgba(124,106,247,0.05)",
        cursor="pointer",
        width="100%",
        padding="28px 36px",
        _hover={"border_color": ACCENT, "background": "rgba(124,106,247,0.1)"},
        transition="all 0.2s",
    )


def documents_empty_state() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.box(
                rx.icon("inbox", size=28, color=MUTED),
                width="56px",
                height="56px",
                border_radius="50%",
                background=f"{ACCENT}14",
                display="flex",
                align_items="center",
                justify_content="center",
            ),
            rx.text("Chưa có tài liệu nào", size="3", weight="medium", color=TEXT),
            rx.text("Hãy upload tài liệu ở trên để bắt đầu", size="2", color=MUTED),
            spacing="3",
            align="center",
        ),
        width="100%",
        padding_y="8",
    )


def dashboard_page() -> rx.Component:
    return rx.box(
        chunk_viewer_modal(),

        # ── Navbar ────────────────────────────────────────────────
        rx.hstack(
            rx.hstack(
                rx.box(
                    rx.text("R", weight="bold", size="4", color="white"),
                    width="32px",
                    height="32px",
                    background=f"linear-gradient(135deg, {ACCENT}, {ACCENT2})",
                    border_radius="8px",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                ),
                rx.text("RAG QA", weight="bold", size="4", color=TEXT),
                spacing="2",
                align="center",
            ),
            rx.spacer(),
            rx.hstack(
                rx.color_mode.button(),
                rx.link(
                    rx.button(
                        rx.icon("message-circle", size=16),
                        "Chat ngay",
                        variant="solid",
                        size="2",
                        cursor="pointer",
                        background=f"linear-gradient(135deg, {ACCENT}, #9d8df7)",
                        color="white",
                    ),
                    href="/chat",
                ),
                spacing="3",
                align="center",
            ),
            padding_x="6",
            padding_y="4",
            background=CARD,
            border_bottom=f"1px solid {BORDER}",
            box_shadow="0 1px 3px rgba(15,17,23,0.04)",
            position="sticky",
            top="0",
            z_index="10",
            width="100%",
        ),

        # ── Body: layout 1 cột, thẳng hàng theo chiều dọc — Header → Thống
        # kê (gọn) → Upload (thanh ngang gọn) → Danh sách tài liệu (chiếm
        # phần lớn không gian, là nội dung chính) — thay cho layout 2 cột
        # trước đây (upload | danh sách) hay bị lệch chiều cao và để lại
        # nhiều khoảng trống khi danh sách còn ít/trống tài liệu. ──────────
        rx.vstack(
            rx.vstack(
                rx.text("Quản lý tài liệu", size="7", weight="bold", color=TEXT),
                rx.text("Upload và quản lý tài liệu để hỏi đáp với AI", size="3", color=MUTED),
                spacing="1",
                width="100%",
            ),

            # Thống kê — thanh gọn, không chiếm nhiều chiều cao
            rx.hstack(
                stat_card("Tổng tài liệu", AppState.doc_count_str, "files", ACCENT),
                stat_card("Sẵn sàng", AppState.ready_count_str, "circle-check", SUCCESS),
                stat_card("Đang xử lý", AppState.pending_count_str, "loader", WARNING),
                spacing="3",
                width="100%",
                flex_wrap="wrap",
            ),

            # Upload — thanh ngang gọn, full width
            upload_zone(),

            # Danh sách tài liệu — full width, là khu vực chính của trang
            rx.vstack(
                rx.hstack(
                    rx.icon("folder", size=16, color=MUTED),
                    rx.text("Tài liệu của bạn", size="4", weight="bold", color=TEXT),
                    rx.spacer(),
                    rx.text(AppState.doc_count_str + " tài liệu", size="1", color=MUTED),
                    rx.icon_button(
                        rx.icon("refresh-cw", size=14),
                        variant="ghost",
                        size="1",
                        cursor="pointer",
                        on_click=AppState.load_documents,
                    ),
                    width="100%",
                    align="center",
                    spacing="3",
                ),
                rx.cond(
                    AppState.doc_count > 0,
                    rx.vstack(
                        rx.foreach(AppState.documents, doc_row),
                        spacing="1",
                        width="100%",
                    ),
                    documents_empty_state(),
                ),
                width="100%",
                padding="32px 36px",
                background=CARD,
                border_radius="12px",
                border=f"1px solid {BORDER}",
                spacing="4",
                flex="1",
            ),

            spacing="5",
            width="100%",
            align="stretch",
            max_width="920px",
            margin="0 auto",
            padding="6",
        ),

        min_height="100vh",
        background=BG,
        on_mount=AppState.load_documents,
    )
