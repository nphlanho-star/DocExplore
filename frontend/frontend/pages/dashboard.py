"""
pages/dashboard.py — Trang quản lý tài liệu  (UI Pro Max).
"""
import reflex as rx
from frontend.state import AppState, Chunk, Document
from frontend.theme import (
    BG, CARD, SIDEBAR,
    ACCENT, ACCENT_GRAD, ACCENT_GRAD2,
    TEXT, MUTED,
    SUCCESS, WARNING, DANGER, BORDER, DIVIDER,
    SHADOW_SM, SHADOW_LG,
)


# ── Status badge ───────────────────────────────────────────────────────────────
def status_badge(doc: Document) -> rx.Component:
    return rx.cond(
        doc.status == "completed",
        rx.badge("✓ Sẵn sàng",    color_scheme="green",  variant="soft", radius="full"),
        rx.cond(
            doc.status == "failed",
            rx.badge("✗ Lỗi",     color_scheme="red",    variant="soft", radius="full"),
            rx.badge("⏳ Đang xử lý", color_scheme="yellow", variant="soft", radius="full"),
        ),
    )


# ── File-type badge ────────────────────────────────────────────────────────────
def file_type_badge(doc: Document) -> rx.Component:
    return rx.cond(
        doc.file_extension == ".pdf",
        rx.badge("PDF",  color_scheme="red",    variant="soft", radius="full", size="1"),
        rx.cond(
            doc.file_extension == ".docx",
            rx.badge("DOCX", color_scheme="blue",   variant="soft", radius="full", size="1"),
            rx.cond(
                doc.file_extension == ".doc",
                rx.badge("DOC",  color_scheme="blue",   variant="soft", radius="full", size="1"),
                rx.cond(
                    doc.file_extension == ".pptx",
                    rx.badge("PPTX", color_scheme="orange", variant="soft", radius="full", size="1"),
                    rx.badge("TXT",  color_scheme="gray",   variant="soft", radius="full", size="1"),
                ),
            ),
        ),
    )


# ── Stat card ──────────────────────────────────────────────────────────────────
def stat_card(label: str, value: str, icon: str, color: str) -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.icon(icon, size=18, color=color),
            width="42px", height="42px",
            border_radius="12px",
            background=f"{color}1a",
            display="flex",
            align_items="center",
            justify_content="center",
            flex_shrink="0",
        ),
        rx.vstack(
            rx.text(value, size="6", weight="bold", color=TEXT, line_height="1.1"),
            rx.text(label, size="1", color=MUTED, weight="medium"),
            spacing="0",
            align="start",
        ),
        spacing="3",
        align="center",
        padding="18px 22px",
        border_radius="12px",
        background=CARD,
        border=f"1px solid {BORDER}",
        border_top=f"2px solid {color}",
        flex="1",
        min_width="160px",
        box_shadow=SHADOW_SM,
        _hover={"transform": "translateY(-2px)", "box_shadow": "0 8px 24px rgba(0,0,0,0.13)"},
        transition="all 0.2s ease",
    )


# ── Document row ───────────────────────────────────────────────────────────────
def doc_row(doc: Document) -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.icon("file-text", size=16, color=ACCENT),
            width="38px", height="38px",
            border_radius="10px",
            background=f"{ACCENT}1a",
            display="flex",
            align_items="center",
            justify_content="center",
            flex_shrink="0",
        ),
        rx.vstack(
            rx.hstack(
                rx.text(
                    doc.original_filename,
                    size="2",
                    weight="medium",
                    color=TEXT,
                    no_wrap=True,
                    overflow="hidden",
                    text_overflow="ellipsis",
                    max_width="320px",
                ),
                file_type_badge(doc),
                spacing="2",
                align="center",
                flex_wrap="wrap",
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
                title="Xem chunks",
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
            title="Xóa",
            on_click=AppState.delete_document(doc.id),
        ),
        width="100%",
        padding="14px 18px",
        border_radius="10px",
        border="1px solid transparent",
        transition="all 0.15s ease",
        _hover={"background": "rgba(124,106,247,0.07)", "border_color": BORDER},
        align="center",
        spacing="3",
    )


# ── Chunk card ─────────────────────────────────────────────────────────────────
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
            line_height="1.6",
        ),
        padding="5",
        border_radius="10px",
        background=BG,
        border=f"1px solid {BORDER}",
        width="100%",
    )


# ── Chunk viewer modal ─────────────────────────────────────────────────────────
def chunk_viewer_modal() -> rx.Component:
    return rx.cond(
        AppState.show_chunks,
        rx.box(
            rx.box(
                # Header
                rx.box(
                    rx.hstack(
                        rx.vstack(
                            rx.hstack(
                                rx.box(
                                    rx.icon("layers", size=14, color=ACCENT),
                                    width="28px", height="28px",
                                    background="rgba(124,106,247,0.14)",
                                    border_radius="8px",
                                    display="flex",
                                    align_items="center",
                                    justify_content="center",
                                ),
                                rx.text("Dữ liệu sau khi chunking", size="4", weight="bold", color=TEXT),
                                spacing="2",
                                align="center",
                            ),
                            rx.hstack(
                                rx.text(AppState.chunks_doc_name, size="2", color=MUTED),
                                rx.text("·", size="2", color=MUTED),
                                rx.badge(
                                    AppState.chunks_count_str + " chunk",
                                    variant="soft", color_scheme="violet",
                                    radius="full", size="1",
                                ),
                                spacing="2",
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
                            on_click=AppState.close_chunks,
                        ),
                        width="100%",
                        align="center",
                    ),
                    padding="5",
                    padding_bottom="3",
                    border_bottom=f"1px solid {DIVIDER}",
                    flex_shrink="0",
                ),
                # Body
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
                    padding="5",
                    padding_top="0",
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


# ── Upload zone ────────────────────────────────────────────────────────────────
def upload_zone() -> rx.Component:
    return rx.upload(
        rx.hstack(
            rx.cond(
                AppState.is_uploading,
                rx.spinner(size="3", color=ACCENT),
                rx.box(
                    rx.icon("cloud-upload", size=24, color=ACCENT),
                    width="48px", height="48px",
                    background="rgba(124,106,247,0.10)",
                    border_radius="12px",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    flex_shrink="0",
                ),
            ),
            rx.vstack(
                rx.text(
                    "Kéo thả file vào đây, hoặc click để chọn",
                    size="3",
                    weight="medium",
                    color=TEXT,
                ),
                rx.text("Hỗ trợ: PDF · DOCX · DOC · PPTX · TXT", size="1", color=MUTED),
                spacing="1",
                align="start",
            ),
            rx.spacer(),
            rx.cond(
                AppState.upload_msg != "",
                rx.hstack(
                    rx.cond(
                        AppState.upload_ok,
                        rx.icon("circle-check", size=14, color=SUCCESS),
                        rx.icon("circle-x",    size=14, color=DANGER),
                    ),
                    rx.text(
                        AppState.upload_msg,
                        size="2",
                        color=rx.cond(AppState.upload_ok, SUCCESS, DANGER),
                        weight="medium",
                    ),
                    spacing="2",
                    align="center",
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
        border="2px dashed rgba(124,106,247,0.35)",
        border_radius="12px",
        background="rgba(124,106,247,0.04)",
        cursor="pointer",
        width="100%",
        padding="22px 28px",
        _hover={"border_color": ACCENT, "background": "rgba(124,106,247,0.09)"},
        transition="all 0.2s ease",
    )


# ── Empty state ────────────────────────────────────────────────────────────────
def documents_empty_state() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.box(
                rx.icon("inbox", size=28, color=ACCENT),
                width="60px", height="60px",
                border_radius="50%",
                background="rgba(124,106,247,0.10)",
                display="flex",
                align_items="center",
                justify_content="center",
            ),
            rx.text("Chưa có tài liệu nào", size="3", weight="medium", color=TEXT),
            rx.text("Upload tài liệu ở trên để bắt đầu hỏi đáp", size="2", color=MUTED),
            spacing="3",
            align="center",
        ),
        width="100%",
        padding_y="10",
    )


# ── Dashboard page ─────────────────────────────────────────────────────────────
def dashboard_page() -> rx.Component:
    return rx.box(
        chunk_viewer_modal(),

        # ── Navbar ────────────────────────────────────────────────────
        rx.hstack(
            rx.hstack(
                rx.box(
                    rx.icon("sparkles", size=14, color="white"),
                    width="32px", height="32px",
                    background=ACCENT_GRAD2,
                    border_radius="9px",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    box_shadow="0 2px 8px rgba(124,106,247,0.28)",
                ),
                rx.text("RAG QA", weight="bold", size="4", color=TEXT, letter_spacing="-0.2px"),
                spacing="2",
                align="center",
            ),
            rx.spacer(),
            rx.hstack(
                rx.color_mode.button(variant="ghost"),
                rx.link(
                    rx.button(
                        rx.icon("message-circle", size=15),
                        "Chat ngay",
                        variant="solid",
                        size="2",
                        cursor="pointer",
                        background=ACCENT_GRAD,
                        color="white",
                        border_radius="9px",
                        _hover={"opacity": "0.88"},
                        transition="opacity 0.18s",
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
            box_shadow="0 1px 4px rgba(15,17,23,0.05)",
            position="sticky",
            top="0",
            z_index="10",
            width="100%",
        ),

        # ── Body ──────────────────────────────────────────────────────
        rx.vstack(
            # Page title
            rx.vstack(
                rx.text("Quản lý tài liệu", size="7", weight="bold", color=TEXT, letter_spacing="-0.5px"),
                rx.text("Upload và quản lý tài liệu để hỏi đáp với AI", size="3", color=MUTED),
                spacing="1",
                width="100%",
            ),

            # Stats row
            rx.hstack(
                stat_card("Tổng tài liệu",  AppState.doc_count_str,     "files",        ACCENT),
                stat_card("Sẵn sàng",        AppState.ready_count_str,   "circle-check", SUCCESS),
                stat_card("Đang xử lý",      AppState.pending_count_str, "loader",       WARNING),
                spacing="4",
                width="100%",
                flex_wrap="wrap",
            ),

            # Upload zone
            upload_zone(),

            # Documents section
            rx.vstack(
                rx.hstack(
                    rx.hstack(
                        rx.icon("folder-open", size=16, color=MUTED),
                        rx.text("Tài liệu của bạn", size="4", weight="bold", color=TEXT),
                        spacing="2",
                        align="center",
                    ),
                    rx.spacer(),
                    rx.badge(
                        AppState.doc_count_str + " tài liệu",
                        variant="soft", color_scheme="gray", radius="full",
                    ),
                    rx.icon_button(
                        rx.icon("refresh-cw", size=13),
                        variant="ghost",
                        size="1",
                        cursor="pointer",
                        title="Làm mới",
                        on_click=AppState.load_documents,
                    ),
                    width="100%",
                    align="center",
                    spacing="3",
                ),
                # Divider
                rx.box(height="1px", background=DIVIDER, width="100%"),
                # List
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
                padding="28px 32px",
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
