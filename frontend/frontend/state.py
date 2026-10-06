"""
state.py — Global state cho RAG QA Frontend (không có auth).
Dùng pydantic.BaseModel — Reflex hỗ trợ qua __fields__ trong get_attribute_access_type.
"""
import json

import httpx
import reflex as rx

SCROLL_BOTTOM_JS = (
    "setTimeout(()=>{const e=document.getElementById('chat-scroll');"
    "if(e){e.scrollTo({top:e.scrollHeight,behavior:'smooth'});}},60)"
)
from pydantic import BaseModel

API = "http://localhost:8000"


class Source(BaseModel):
    document_name: str = ""
    document_id: str = ""
    content_snippet: str = ""
    content: str = ""
    chunk_index: int = 0
    page_number: int = 0
    relevance_score: float = 0.0


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


class SessionItem(BaseModel):
    id: str = ""
    title: str = ""


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
    # is_generating: True suốt từ lúc gửi câu hỏi tới lúc có "done"/"error"/bị
    # dừng — dùng để hiện nút Dừng (khác is_loading, vốn tắt ngay khi có token
    # đầu tiên để chuyển từ hiệu ứng "đang suy nghĩ" sang hiển thị chữ streaming).
    is_generating: bool = False
    stop_requested: bool = False
    # Case #10: giữ id phiên chat hiện tại để backend đọc được lịch sử hội
    # thoại (trước đây không gửi → mỗi câu hỏi bị coi là 1 phiên mới hoàn toàn).
    session_id: str = ""
    sessions: list[SessionItem] = []   # lịch sử các cuộc chat (chỉ id + tiêu đề)
    thinking_stage: str = ""
    ctx_percent: int = 0          # % bộ nhớ hội thoại đã dùng (cộng dồn qua các lượt); 60–85% (ở chỗ hết ý) → nén
    ctx_compressed: bool = False  # lượt vừa rồi vừa nén bộ nhớ
    ctx_detail: str = "Bộ nhớ hội thoại: 0%. Mỗi câu hỏi/trả lời tiêu hao một phần; đến khoảng 60–85% (ở chỗ hết ý) sẽ được nén lại."
    selected_doc_ids: list[str] = []

    # ── Chunk viewer ──────────────────────────────────────────────────
    show_chunks: bool = False
    chunks_loading: bool = False
    chunks_doc_name: str = ""
    chunks: list[Chunk] = []

    # ── Xem 1 chunk nguồn được click từ câu trả lời ─────────────────────
    show_source_chunk: bool = False
    source_chunk_doc_name: str = ""
    source_chunk_index: int = 0
    source_chunk_page: int = 0
    source_chunk_content: str = ""
    source_chunk_score: float = 0.0

    # ── UI ────────────────────────────────────────────────────────────
    error_msg: str = ""

    # ── Computed vars ─────────────────────────────────────────────────
    @rx.var
    def ctx_color(self) -> str:
        """Tím < 60% (thoải mái), vàng 60–85% (sắp nén / vừa nén), đỏ ≥ 85% (buộc nén)."""
        if self.ctx_percent >= 85:
            return "#ef4444"
        if self.ctx_percent >= 60 or self.ctx_compressed:
            return "#f59e0b"
        return "#7c6af7"

    @rx.var
    def ctx_ring(self) -> str:
        pct = max(0, min(100, self.ctx_percent))
        return f"conic-gradient({self.ctx_color} {pct}%, rgba(148,163,184,0.28) 0)"

    @rx.var
    def ctx_label(self) -> str:
        return f"{self.ctx_percent}%"

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

    @rx.var
    def source_chunk_index_str(self) -> str:
        return str(self.source_chunk_index)

    @rx.var
    def source_chunk_page_str(self) -> str:
        return str(self.source_chunk_page)

    @rx.var
    def source_chunk_score_str(self) -> str:
        return f"{self.source_chunk_score:.2f}"

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

    def open_source_chunk(
        self,
        document_name: str,
        chunk_index: int,
        page_number: int,
        content: str,
        relevance_score: float,
    ):
        """Mở modal xem toàn bộ nội dung 1 chunk nguồn được click từ câu trả lời."""
        self.source_chunk_doc_name = document_name
        self.source_chunk_index = chunk_index
        self.source_chunk_page = page_number
        self.source_chunk_content = content
        self.source_chunk_score = relevance_score
        self.show_source_chunk = True

    def close_source_chunk(self):
        self.show_source_chunk = False

    # ── Chat handlers ─────────────────────────────────────────────────
    def set_query(self, value: str):
        self.query = value

    def set_query_suggest(self, q: str):
        self.query = q

    def clear_chat(self):
        self.messages = []
        self.session_id = ""   # cuộc trò chuyện mới → không mang ngữ cảnh cũ sang
        self.ctx_percent = 0
        self.ctx_compressed = False

    async def load_sessions(self):
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{API}/api/chat/sessions", params={"limit": 50})
            if r.status_code == 200:
                self.sessions = [
                    SessionItem(id=str(s.get("id", "")), title=(s.get("title") or "Cuộc trò chuyện mới")[:60])
                    for s in r.json()
                ]
        except Exception as e:
            self.error_msg = f"Không tải được lịch sử chat: {e}"

    async def open_session(self, sid: str):
        if self.is_generating:
            return
        try:
            async with httpx.AsyncClient(timeout=15.0) as c:
                r = await c.get(f"{API}/api/chat/sessions/{sid}")
            if r.status_code != 200:
                self.error_msg = "Không mở được cuộc trò chuyện này."
                return
            msgs: list[Message] = []
            for m in r.json().get("messages", []):
                srcs = [
                    Source(
                        document_name=s.get("document_name", "") or "",
                        document_id=str(s.get("document_id", "") or ""),
                        content_snippet=s.get("content_snippet", "") or "",
                        content=s.get("content", "") or s.get("content_snippet", "") or "",
                        chunk_index=s.get("chunk_index", 0) or 0,
                        page_number=s.get("page_number", 0) or 0,
                        relevance_score=s.get("relevance_score", 0.0) or 0.0,
                    )
                    for s in (m.get("sources") or [])
                ]
                msgs.append(Message(role=m.get("role", "user"), content=m.get("content", ""), sources=srcs))
            self.messages = msgs
            self.session_id = sid
            self.ctx_percent = int(r.json().get("memory_percent", 0) or 0)
            self.ctx_compressed = False
        except Exception as e:
            self.error_msg = f"Không mở được cuộc trò chuyện: {e}"
            return
        yield rx.call_script(SCROLL_BOTTOM_JS)

    async def delete_session_item(self, sid: str):
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                await c.delete(f"{API}/api/chat/sessions/{sid}")
        except Exception as e:
            self.error_msg = f"Không xóa được cuộc trò chuyện: {e}"
            return
        if self.session_id == sid:
            self.messages = []
            self.session_id = ""
        await self.load_sessions()

    def stop_generation(self):
        # Chỉ set cờ — vòng lặp đọc SSE trong send_query sẽ tự phát hiện ở
        # lần lặp kế tiếp và đóng kết nối stream ngay (khiến backend cũng
        # nhận biết client đã ngắt kết nối và dừng xử lý theo).
        self.stop_requested = True

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
        self.is_generating = True
        self.stop_requested = False
        self.thinking_stage = "Đang phân tích câu hỏi…"
        self.ctx_compressed = False
        yield rx.call_script(SCROLL_BOTTOM_JS)
        # placeholder tin nhắn assistant — sẽ được "đổ" dần chữ vào bằng streaming
        self.messages = [*self.messages, Message(role="assistant", content="", sources=[])]
        yield

        payload: dict = {"question": q, "top_k": 5}
        if self.selected_doc_ids:
            payload["document_ids"] = self.selected_doc_ids
        if self.session_id:
            payload["session_id"] = self.session_id

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
                        if self.stop_requested:
                            # Đóng kết nối stream ngay — backend sẽ nhận biết
                            # client đã ngắt và dừng xử lý (kể cả khi đang gọi
                            # LLM) thay vì chạy tiếp trong vô ích.
                            if self.messages:
                                last = self.messages[-1]
                                stopped_content = (last.content + "\n\n*(Đã dừng theo yêu cầu)*").strip()
                                self._replace_last_assistant(stopped_content, sources=last.sources)
                            self.is_loading = False
                            yield
                            break

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
                                    content=s.get("content", ""),
                                    chunk_index=s.get("chunk_index", 0) or 0,
                                    page_number=s.get("page_number", 0) or 0,
                                    relevance_score=s.get("relevance_score", 0.0) or 0.0,
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

                        elif etype == "memory":
                            self.ctx_percent = int(event.get("percent", 0) or 0)
                            self.ctx_compressed = bool(event.get("compacted", False))
                            n = int(event.get("compactions", 0) or 0)
                            self.ctx_detail = (
                                f"Bộ nhớ hội thoại: {self.ctx_percent}%. "
                                + ("Vừa nén bộ nhớ: các câu hỏi cũ được gộp thành ghi chú ngắn, giữ nguyên văn vài lượt gần nhất. "
                                   if self.ctx_compressed else "Sẽ nén ở chỗ hết ý, khoảng 60–85%. ")
                                + (f"(đã nén {n} lần)" if n else "")
                            )
                            yield

                        elif etype == "token":
                            if not got_first_token:
                                got_first_token = True
                                self.is_loading = False
                                yield rx.call_script(SCROLL_BOTTOM_JS)
                            token = event.get("content", "")
                            if self.messages:
                                last = self.messages[-1]
                                self.messages = [
                                    *self.messages[:-1],
                                    Message(role="assistant", content=last.content + token, sources=last.sources),
                                ]
                            yield

                        elif etype == "error":
                            err_msg = event.get("message", "Có lỗi xảy ra.")
                            if "Phiên chat không tồn tại" in err_msg:
                                self.session_id = ""   # phiên cũ đã bị xóa → lần sau tự tạo phiên mới
                            self._replace_last_assistant(f"Lỗi: {err_msg}")
                            yield

                        elif etype == "done":
                            if event.get("session_id"):
                                self.session_id = str(event["session_id"])

        except Exception as e:
            self._replace_last_assistant(f"Lỗi kết nối: {e}")
            yield
        finally:
            self.is_loading = False
            self.is_generating = False
            self.stop_requested = False
            self.thinking_stage = ""
            yield
            await self.load_sessions()   # cập nhật danh sách lịch sử (tiêu đề phiên mới)

    async def handle_enter(self, key: str, key_info: rx.event.KeyInputInfo):
        # Enter de gui, Shift+Enter de xuong dong.
        if key == "Enter" and not key_info["shift_key"]:
            # Chan hanh vi mac dinh cua textarea (tu chen ky tu xuong dong) -
            # neu khong chan, no se "ghi de" lai noi dung ngay sau khi query
            # vua duoc xoa, khien o nhap tuong nhu khong bi xoa.
            yield rx.prevent_default
            async for _ in self.send_query():
                yield
