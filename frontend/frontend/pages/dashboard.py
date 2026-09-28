"""
pages/dashboard.py — Trang quản lý tài liệu.
"""
import reflex as rx
from frontend.state import AppState, Chunk, Document

BG = "#0f1117"
CARD = "#1a1d27"
ACCENT = "#7c6af7"
ACCENT2 = "#5eead4"
TEXT = "#e2e8f0"
MUTED = "#64748b"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER = "#ef4444"


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
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.icon(icon, size=20, color=color),
                rx.text(label, size="2", color=MUTED, weight="medium"),
                spacing="2",
            ),
            rx.text(value, size="7", weight="bold", color=TEXT),
            spacing="1",
            align="start",
        ),
        padding="5",
        border_radius="12px",
        background=CARD,
        border="1px solid rgba(124,106,247,0.15)",
        flex="1",
        min_width="160px",
    )


def doc_row(doc: Document) -> rx.Component:
    return rx.hstack(
        rx.icon("file-text", size=16, color=ACCENT),
        rx.vstack(
            rx.text(
                doc.original_filename,
                size="2",
                weight="medium",
                color=TEXT,
                no_wrap=True,
                overflow="hidden",
                text_overflow="ellipsis",
                max_width="260px",
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
        padding="3",
        border_radius="8px",
        _hover={"background": "rgba(124,106,247,0.08)"},
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
        padding="4",
        border_radius="10px",
        background=BG,
        border="1px solid rgba(255,255,255,0.06)",
        width="100%",
    )


def chunk_viewer_modal() -> rx.Component:
    return rx.cond(
        AppState.show_chunks,
        rx.box(
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
                    padding_bottom="3",
                    border_bottom="1px solid rgba(255,255,255,0.08)",
                ),
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
                background=CARD,
                border_radius="16px",
                padding="6",
                width="min(720px, 92vw)",
                max_height="85vh",
                overflow_y="auto",
                box_shadow="0 20px 60px rgba(0,0,0,0.5)",
                border="1px solid rgba(255,255,255,0.08)",
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
    return rx.upload(
        rx.vstack(
            rx.cond(
                AppState.is_uploading,
                rx.spinner(size="3"),
                rx.vstack(
                    rx.icon("cloud-upload", size=36, color=ACCENT),
                    rx.text("Kéo thả file vào đây", size="3", weight="medium", color=TEXT),
                    rx.text("hoặc click để chọn file", size="2", color=MUTED),
                    rx.text("PDF, DOCX, DOC, PPTX, TXT", size="1", color=MUTED),
                    spacing="2",
                    align="center",
                ),
            ),
            rx.cond(
                AppState.upload_msg != "",
                rx.text(
                    AppState.upload_msg,
                    size="2",
                    color=rx.cond(AppState.upload_ok, SUCCESS, DANGER),
                    weight="medium",
                ),
                rx.text(""),
            ),
            spacing="3",
            align="center",
            padding="8",
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
        _hover={"border_color": ACCENT, "background": "rgba(124,106,247,0.1)"},
        transition="all 0.2s",
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
            padding_x="6",
            padding_y="4",
            background=CARD,
            border_bottom="1px solid rgba(255,255,255,0.06)",
            position="sticky",
            top="0",
            z_index="10",
            width="100%",
        ),

        # ── Body ──────────────────────────────────────────────────
        rx.box(
            rx.vstack(
                rx.text("Quản lý tài liệu", size="7", weight="bold", color=TEXT),
                rx.text("Upload và quản lý tài liệu để hỏi đáp với AI", size="3", color=MUTED),
                spacing="1",
                margin_bottom="6",
            ),

            # Stats
            rx.hstack(
                stat_card("Tổng tài liệu", AppState.doc_count_str, "files", ACCENT),
                stat_card("Sẵn sàng", AppState.ready_count_str, "circle-check", SUCCESS),
                stat_card("Đang xử lý", AppState.pending_count_str, "loader", WARNING),
                spacing="4",
                width="100%",
                flex_wrap="wrap",
                margin_bottom="6",
            ),

            # Upload + List
            rx.hstack(
                rx.vstack(
                    rx.text("Tải lên tài liệu", size="4", weight="bold", color=TEXT),
                    upload_zone(),
                    spacing="4",
                    width="100%",
                    min_width="300px",
                    flex="1",
                ),
                rx.vstack(
                    rx.hstack(
                        rx.text("Tài liệu của bạn", size="4", weight="bold", color=TEXT),
                        rx.spacer(),
                        rx.icon_button(
                            rx.icon("refresh-cw", size=14),
                            variant="ghost",
                            size="1",
                            cursor="pointer",
                            on_click=AppState.load_documents,
                        ),
                        width="100%",
                    ),
                    rx.cond(
                        AppState.doc_count > 0,
                        rx.vstack(
                            rx.foreach(AppState.documents, doc_row),
                            spacing="1",
                            width="100%",
                            max_height="400px",
                            overflow_y="auto",
                        ),
                        rx.vstack(
                            rx.icon("inbox", size=40, color=MUTED),
                            rx.text("Chưa có tài liệu nào", size="3", color=MUTED),
                            rx.text("Hãy upload tài liệu để bắt đầu", size="2", color=MUTED),
                            spacing="2",
                            align="center",
                            padding="8",
                        ),
                    ),
                    width="100%",
                    min_width="300px",
                    flex="1",
                    padding="5",
                    background=CARD,
                    border_radius="12px",
                    border="1px solid rgba(255,255,255,0.06)",
                    spacing="4",
                ),
                spacing="6",
                width="100%",
                align="start",
                flex_wrap="wrap",
            ),

            padding="6",
            max_width="1100px",
            margin="0 auto",
            width="100%",
        ),

        min_height="100vh",
        background=BG,
        on_mount=AppState.load_documents,
    )
