
# Tech Stack

## Frontend

- **React**
  - Frontend được phát triển dựa trên React, sử dụng mô hình component-based để xây dựng giao diện linh hoạt và dễ mở rộng.
  - Có thể phát triển theo các hướng tương tự như ReactPy hoặc Reflex khi muốn sử dụng Python thay cho JavaScript.
  - Frontend chịu trách nhiệm hiển thị giao diện, xử lý tương tác người dùng và kết nối với Backend thông qua API.

## Backend API

- **FastAPI**
  - Xây dựng REST API cho frontend.
  - Xử lý upload file, quản lý tài liệu, chat và kết nối với hệ thống RAG.
  - Nhanh, dễ viết và phù hợp với Python ecosystem của AI/RAG.

## RAG Framework

- **LlamaIndex**
  - Framework chính cho hệ thống RAG.
  - Phù hợp với bài toán tài liệu dài, nhiều loại nội dung và nhiều định dạng file.
  - Hỗ trợ document loader, indexing, retrieval, metadata và citation.

- **LangGraph** *(có thể dùng ở giai đoạn sau)*
  - Dùng khi hệ thống cần workflow phức tạp.
  - Ví dụ: chatbot cần hỏi lại người dùng, gọi nhiều bước tìm kiếm hoặc xử lý nhiều agent.

## Document Parser

- **Docling**
  - Dùng để đọc và trích xuất nội dung từ PDF, Word, PowerPoint, Excel,...
  - Phù hợp với tài liệu phức tạp có bảng, tiêu đề, hình ảnh và layout nhiều cột.
  - Hỗ trợ chuẩn hóa nội dung trước khi chunking.

## OCR

- **Docling OCR**
  - Dùng cho PDF dạng scan hoặc ảnh.

- **PaddleOCR**
  - Dùng làm phương án bổ sung cho tiếng Việt.
  - Hữu ích khi chất lượng OCR của tài liệu tiếng Việt chưa tốt.

## Database

- **PostgreSQL**
  - Lưu thông tin có cấu trúc.
  - Ví dụ: tài khoản người dùng, metadata file, trạng thái xử lý, quyền truy cập, lịch sử chat và feedback.
  - Không dùng PostgreSQL để lưu file vật lý.

## Vector Database

- **Qdrant**
  - Lưu embedding của các đoạn tài liệu.
  - Hỗ trợ vector search, hybrid search và metadata filtering.
  - Có thể filter theo loại file, phòng ban, ngày upload hoặc quyền người dùng.
  - Phù hợp khi số lượng tài liệu tăng lớn.

## Embedding Model

- **BGE-M3**
  - Hỗ trợ đa ngôn ngữ, phù hợp với tiếng Việt.
  - Dùng để chuyển các đoạn tài liệu và câu hỏi thành vector.
  - Phù hợp cho semantic search và hybrid retrieval.

## Reranker

- **bge-reranker-v2-m3**
  - Dùng để sắp xếp lại các kết quả tìm kiếm.
  - Giúp chọn đúng đoạn tài liệu liên quan nhất trước khi gửi cho LLM.
  - Tăng độ chính xác của câu trả lời.

## File Storage

- **MinIO**
  - Lưu file gốc như PDF, Word, Excel và PowerPoint.
  - Có thể dùng để lưu file upload, file đã xử lý và kết quả OCR.
  - Metadata của file sẽ được lưu trong PostgreSQL.

## Background Jobs

- **Celery**
  - Xử lý các tác vụ chạy ngầm.
  - Ví dụ: parse file, OCR, chunking, tạo embedding và index vào Qdrant.

- **Redis**
  - Làm message broker và hàng đợi cho Celery.
  - Giúp các tác vụ xử lý file không làm chậm API chính.

## Large Language Model

  - qwen3.5: 2b (ollama)

## Tổng quan luồng dữ liệu

1. Người dùng upload file lên MinIO.
2. FastAPI gửi background job qua Celery.
3. Docling/PaddleOCR đọc và trích xuất nội dung.
4. Nội dung được chia thành nhiều chunk.
5. BGE-M3 tạo embedding cho từng chunk.
6. Embedding và metadata được lưu vào Qdrant.
7. Metadata, user, quyền truy cập và lịch sử chat được lưu trong PostgreSQL.
8. Khi người dùng hỏi, hệ thống tìm kiếm bằng Qdrant.
9. bge-reranker-v2-m3 chọn các đoạn phù hợp nhất.
10. GPT-4o tạo câu trả lời dựa trên các đoạn đã tìm được và trả về nguồn tham khảo.
