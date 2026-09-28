"""
state.py — Global state cho RAG QA Frontend (không có auth).
Dùng pydantic.BaseModel — Reflex hỗ trợ qua __fields__ trong get_attribute_access_type.
"""
import json

import httpx
import reflex as rx
from pydantic import BaseModel

API = "http://localhost:8000"


class Source(BaseModel):
    document_name: str = ""
    document_id: str = ""
    content_snippet: str = ""


class Message(BaseModel):
    role: str = "user"
    content: str = ""
    sources: list[Source] = []


class Document(BaseModel):
    id: str = ""
    original_filename: str = ""
    file_extension: str = ""
    status: str = "pending"


class Chunk(BaseModel):
    id: str = ""
    chunk_index: int = 0
    content: str = ""
    page_number: int = 0
    chapter: str = ""
    article_number: str = ""
    article_title: str = ""


class AppState(rx.State):
    # ── Documents ─────────────────────────────────────────────────────
    documents: list[Document] = []
    upload_msg: str = ""
    upload_ok: bool = False
    is_uploading: bool = False

    # ── Chat ──────────────────────────────────────────────────────────
    messages: list[Message] = []
    query: str = ""
    is_loading: bool = False
    thinking_stage: str = ""
    selected_doc_ids: list[str] = []

    # ── Chunk viewer ──────────────────────────────────────────────────
    show_chunks: bool = False
    chunks_loading: bool = False
    chunks_doc_name: str = ""
    chunks: list[Chunk] = []

    # ── UI ────────────────────────────────────────────────────────────
    error_msg: str = ""

    # ── Computed vars ─────────────────────────────────────────────────
    @rx.var
    def doc_count(self) -> int:
        return len(self.documents)

    @rx.var
    def doc_count_str(self) -> str:
        return str(len(self.documents))

    @rx.var
    def ready_count_str(self) -> str:
        return str(sum(1 for d in self.documents if d.status == "completed"))

    @rx.var
    def pending_count_str(self) -> str:
        return str(sum(1 for d in self.documents if d.status in ("pending", "processing")))

    @rx.var
    def has_messages(self) -> bool:
        return len(self.messages) > 0

    @rx.var
    def chunks_count_str(self) -> str:
        return str(len(self.chunks))

    # ── Document handlers ─────────────────────────────────────────────
    async def load_documents(self):
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{API}/api/documents/")
            if r.status_code == 200:
                raw = r.json().get("items", [])
                self.documents = [
                    Document(
                        id=str(d.get("id", "")),
                        original_filename=d.get("original_filename", ""),
                        file_extension=d.get("file_extension", ""),
                        status=d.get("status", "pending"),
                    )
                    for d in raw
                ]
        except Exception as e:
            self.error_msg = f"Không tải được danh sách tài liệu: {e}"

    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            return
        self.is_uploading = True
        self.upload_msg = "Đang tải lên..."
        self.upload_ok = False
        file = files[0]
        data = await file.read()
        try:
            async with httpx.AsyncClient(timeout=60.0) as c:
                r = await c.post(
                    f"{API}/api/documents/upload",
                    files={"file": (file.filename, data)},
                )
            if r.status_code in (200, 201, 202):
                self.upload_msg = f"Đã tải lên: {file.filename}"
                self.upload_ok = True
                await self.load_documents()
            else:
                try:
                    detail = r.json().get("detail", f"HTTP {r.status_code}")
                except Exception:
                    detail = f"HTTP {r.status_code} — phản hồi không hợp lệ"
                self.upload_msg = f"Lỗi: {detail}"
                self.upload_ok = False
        except Exception as e:
            self.upload_msg = f"Lỗi kết nối: {e}"
            self.upload_ok = False
        finally:
            self.is_uploading = False

    async def delete_document(self, doc_id: str):
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                await c.delete(f"{API}/api/documents/{doc_id}")
            await self.load_documents()
        except Exception as e:
            self.error_msg = f"Lỗi xóa tài liệu: {e}"

    def toggle_doc_selection(self, doc_id: str):
        if doc_id in self.selected_doc_ids:
            self.selected_doc_ids = [d for d in self.selected_doc_ids if d != doc_id]
        else:
            self.selected_doc_ids = [*self.selected_doc_ids, str(doc_id)]

    # ── Chunk viewer ("xem lại data sau khi chunking") ─────────────────
    async def open_chunks(self, doc_id: str, doc_name: str):
        self.show_chunks = True
        self.chunks_loading = True
        self.chunks_doc_name = doc_name
        self.chunks = []
        yield
        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                r = await c.get(f"{API}/api/documents/{doc_id}/chunks")
            if r.status_code == 200:
                raw = r.json().get("items", [])
                result = []
                for item in raw:
                    meta = item.get("chunk_metadata") or {}
                    result.append(
                        Chunk(
                            id=str(item.get("id", "")),
                            chunk_index=item.get("chunk_index", 0),
                            content=item.get("content", ""),
                            page_number=item.get("page_number") or 0,
                            chapter=str(meta.get("chapter") or ""),
                            article_number=str(meta.get("article_number") or ""),
                            article_title=str(meta.get("article_title") or ""),
                        )
                    )
                self.chunks = result
            else:
                self.error_msg = f"Không tải được dữ liệu chunk: HTTP {r.status_code}"
        except Exception as e:
            self.error_msg = f"Không tải được dữ liệu chunk: {e}"
        finally:
            self.chunks_loading = False

    def close_chunks(self):
        self.show_chunks = False
        self.chunks = []

    # ── Chat handlers ─────────────────────────────────────────────────
    def set_query(self, value: str):
        self.query = value

    def set_query_suggest(self, q: str):
        self.query = q

    def clear_chat(self):
        self.messages = []

    def _replace_last_assistant(self, content: str, sources: list[Source] | None = None):
        if not self.messages:
            return
        last = self.messages[-1]
        self.messages = [
            *self.messages[:-1],
            Message(role="assistant", content=content, sources=sources if sources is not None else last.sources),
        ]

    async def send_query(self):
        q = self.query.strip()
        if not q or self.is_loading:
            return
        self.messages = [
            *self.messages,
            Message(role="user", content=q, sources=[]),
        ]
        self.query = ""
        self.is_loading = True
        self.thinking_stage = "Đang phân tích câu hỏi…"
        yield
        # placeholder tin nhắn assistant — sẽ được "đổ" dần chữ vào bằng streaming
        self.messages = [*self.messages, Message(role="assistant", content="", sources=[])]
        yield

        payload: dict = {"question": q, "top_k": 5}
        if self.selected_doc_ids:
            payload["document_ids"] = self.selected_doc_ids

        got_first_token = False
        try:
            async with httpx.AsyncClient(timeout=180.0) as c:
                async with c.stream("POST", f"{API}/api/chat/query/stream", json=payload) as r:
                    if r.status_code != 200:
                        body = await r.aread()
                        try:
                            detail = json.loads(body).get("detail", f"HTTP {r.status_code}")
                        except Exception:
                            detail = f"HTTP {r.status_code}"
                        self._replace_last_assistant(f"Lỗi: {detail}")
                        yield
                        return

                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw = line[len("data: "):]
                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        etype = event.get("type")

                        if etype == "status":
                            self.thinking_stage = event.get("message", "")
                            yield

                        elif etype == "sources":
                            srcs = [
                                Source(
                                    document_name=s.get("document_name", ""),
                                    document_id=str(s.get("document_id", "")),
                                    content_snippet=s.get("content_snippet", ""),
                                )
                                for s in event.get("sources", [])
                            ]
                            if self.messages:
                                last = self.messages[-1]
                                self.messages = [
                                    *self.messages[:-1],
                                    Message(role="assistant", content=last.content, sources=srcs),
                                ]
                            yield

                        elif etype == "token":
                            if not got_first_token:
                                got_first_token = True
                                self.is_loading = False
                            token = event.get("content", "")
                            if self.messages:
                                last = self.messages[-1]
                                self.messages = [
                                    *self.messages[:-1],
                                    Message(role="assistant", content=last.content + token, sources=last.sources),
                                ]
                            yield

                        elif etype == "error":
                            self._replace_last_assistant(f"Lỗi: {event.get('message', 'Có lỗi xảy ra.')}")
                            yield

                        elif etype == "done":
                            pass

        except Exception as e:
            self._replace_last_assistant(f"Lỗi kết nối: {e}")
            yield
        finally:
            self.is_loading = False
            self.thinking_stage = ""
            yield

    async def handle_enter(self, key: str):
        if key == "Enter":
            async for _ in self.send_query():
                yield
