\# Planning: Xây dựng hệ thống RAG hỏi đáp tài liệu



Mục tiêu: xây dựng hệ thống cho phép người dùng upload tài liệu → xử lý/OCR → chunking → embedding → lưu trữ → retrieval + reranking → LLM trả lời có nguồn, với Frontend React và Backend FastAPI.



\-Epic 1 — Frontend

+Feature 1.1 — Giao diện quản lý tài liệu

* Task: Tạo giao diện upload file.
* Task: Hiển thị danh sách tài liệu.
* Task: Hiển thị trạng thái xử lý tài liệu.
* Task: Thêm chức năng xem/xóa tài liệu.

\+Feature 1.2 — Giao diện Chat

* Task: Tạo giao diện nhập câu hỏi.
* Task: Hiển thị câu trả lời từ Backend.
* Task: Hiển thị lịch sử chat.
* Task: Hiển thị nguồn/citation của câu trả lời.

\+Feature 1.3 — Kết nối Backend

* Task: Tạo API client.
* Task: Kết nối API upload file.
* Task: Kết nối API chat.
* Task: Xử lý loading/error từ API.

\-Epic 2 — Backend API

\+Feature 2.1 — FastAPI

* Task: Khởi tạo FastAPI project.
* Task: Thiết kế cấu trúc API.
* Task: Tạo API upload file.
* Task: Tạo API quản lý tài liệu.
* Task: Tạo API chat/query.
* Task: Tạo API lấy lịch sử chat.

\+Feature 2.2 — User \& Access Control

* Task: Quản lý user.
* Task: Quản lý quyền truy cập tài liệu.
* Task: Kiểm tra quyền khi retrieval.
* Task: Lưu metadata user trong PostgreSQL.

\-Epic 3 — Document Processing

\+Feature 3.1 — File Upload \& Storage

* Task: Nhận file từ Frontend.
* Task: Upload file vào MinIO.
* Task: Lưu metadata file vào PostgreSQL.
* Task: Kiểm tra định dạng file.
* Task: Kiểm tra dung lượng file.

\+Feature 3.2 — Document Parser

* Task: Tích hợp Docling.
* Task: Nhận file Word (.docx).
* Task: Parse nội dung Word bằng Docling.
* Task: Trích xuất text và cấu trúc tài liệu.
* Task: Chuẩn hóa nội dung sau khi parse.
* Task: Chuyển nội dung sang bước Chunking.

+Feature 3.3 — OCR cho Word

* Task: Xác định tài liệu Word có nội dung cần OCR.
* Task: Tích hợp Docling OCR.
* Task: Tích hợp PaddleOCR cho nội dung hình ảnh trong Word.
* Task: Trích xuất chữ từ hình ảnh được nhúng trong file Word.
* Task: Kiểm tra và chuẩn hóa kết quả OCR.Feature 3.4 — Chunking
* Task: Thiết kế chiến lược chunking.
* Task: Chia tài liệu thành các chunk.
* Task: Giữ metadata cho từng chunk.
* Task: Lưu thông tin page/document/chunk.

\-Epic 4 — RAG Pipeline

\+Feature 4.1 — Embedding

* Task: Tích hợp BGE-M3.
* Task: Convert document chunks thành vector.
* Task: Convert user query thành vector.
* Task: Xử lý batch embedding.
* Task: Kiểm tra kích thước vector.

\+Feature 4.2 — Vector Database

* Task: Thiết lập Qdrant.
* Task: Tạo collection.
* Task: Lưu embedding vào Qdrant.
* Task: Lưu metadata cùng vector.
* Task: Implement vector search.

+Feature 4.3 — Retrieval

* Task: Nhận query từ người dùng.
* Task: Tạo query embedding.
* Task: Search các chunk liên quan.
* Task: Filter theo document/user/permission.
* Task: Xử lý top-K kết quả.

\+Feature 4.4 — Reranking

* Task: Tích hợp bge-reranker-v2-m3.
* Task: Rerank kết quả retrieval.
* Task: Chọn các chunk có relevance cao.
* Task: Thiết lập threshold/top-K.
* Task: So sánh retrieval trước và sau reranking.

\-Epic 5 — LLM \& Answer Generation

\+Feature 5.1 — Qwen

* Task: Thiết lập Qwen3.5 0.8B qua Ollama.
* Task: Kết nối Backend với Ollama.
* Task: Thiết kế prompt RAG.
* Task: Đưa context từ retrieval vào prompt.
* Task: Sinh câu trả lời.

\+Feature 5.2 — Citation \& Source

* Task: Truyền metadata nguồn vào LLM.
* Task: Xác định document nguồn.
* Task: Xác định page/chunk nguồn.
* Task: Hiển thị citation cho người dùng.
* Task: Hạn chế LLM trả lời ngoài context.

\-Epic 6 — Background Processing

\+Feature 6.1 — Celery

* Task: Thiết lập Celery.
* Task: Tạo task xử lý file.
* Task: Tạo task OCR.
* Task: Tạo task chunking.
* Task: Tạo task embedding.
* Task: Tạo task indexing vào Qdrant.

+Feature 6.2 — Redis

* Task: Thiết lập Redis.
* Task: Kết nối Redis với Celery.
* Task: Quản lý message queue.
* Task: Theo dõi trạng thái background job.
* Task: Xử lý retry khi job thất bại.

\-Epic 7 — Database \& Storage

\+Feature 7.1 — PostgreSQL

* Task: Thiết kế database schema.
* Task: Tạo bảng User.
* Task: Tạo bảng Document.
* Task: Tạo bảng Document Metadata.
* Task: Tạo bảng Chat History.
* Task: Tạo bảng Feedback.
* Task: Tạo bảng Permission.

\+Feature 7.2 — MinIO

* Task: Thiết lập MinIO.
* Task: Tạo bucket.
* Task: Upload file gốc.
* Task: Lưu file đã xử lý.
* Task: Lưu kết quả OCR.
* Task: Quản lý file theo user/document.

\-Epic 8 — Chat \& User Interaction

\+Feature 8.1 — Chat Pipeline

* Task: Nhận câu hỏi.
* Task: Search Qdrant.
* Task: Rerank bằng BGE Reranker.
* Task: Tạo context.
* Task: Gửi context cho Qwen.
* Task: Nhận câu trả lời.
* Task: Trả citation về Frontend.

\+Feature 8.2 — Chat History

* Task: Lưu câu hỏi.
* Task: Lưu câu trả lời.
* Task: Lưu nguồn được sử dụng.
* Task: Hiển thị lịch sử chat.
* Task: Cho phép tiếp tục conversation.

\-Epic 9 — System Integration

\+Feature 9.1 — Document Processing Flow

* Task: React upload file.
* Task: FastAPI nhận file.
* Task: MinIO lưu file.
* Task: Celery tạo background job.
* Task: Docling/PaddleOCR xử lý.
* Task: Chunking.
* Task: BGE-M3 embedding.
* Task: Qdrant indexing.

+Feature 9.2 — Question Answering Flow

* Task: React gửi query.
* Task: FastAPI nhận query.
* Task: BGE-M3 tạo query embedding.
* Task: Qdrant retrieval.
* Task: BGE Reranker rerank.
* Task: Qwen3.5 tạo answer.
* Task: Backend trả answer + citation.

