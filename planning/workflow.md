
# Workflow RAG

```mermaid
flowchart TD
    A[Người dùng upload tài liệu] --> B[Hệ thống đọc file]
    B --> C[OCR nếu file là ảnh scan]
    C --> D[Chia tài liệu thành các đoạn nhỏ]
    D --> E[Tạo embedding]
    E --> F[Lưu vào Vector Database]

    G[Người dùng đặt câu hỏi] --> H[Kiểm tra quyền truy cập]
    H --> I[Tìm các đoạn liên quan]
    F --> I
    I --> J{Có tìm thấy thông tin phù hợp không?}

    J -->|Có| K[Đưa nội dung cho LLM]
    K --> L[Trả lời kèm nguồn tài liệu]

    J -->|Không| M[Thông báo không tìm thấy thông tin]
```

## Luồng hoạt động

1. Người dùng upload tài liệu.
2. Hệ thống đọc, xử lý và chia nhỏ nội dung.
3. Các đoạn nội dung được lưu vào vector database.
4. Người dùng đặt câu hỏi.
5. Hệ thống kiểm tra quyền của người dùng.
6. Hệ thống tìm các đoạn tài liệu liên quan.
7. Nếu tìm được, LLM tạo câu trả lời kèm nguồn.
8. Nếu không tìm được, hệ thống yêu cầu người dùng hỏi rõ hơn hoặc báo không có thông tin.
