
# Hướng giải quyết

## Xử lý nhiều loại tài liệu

- Dùng tool phù hợp để đọc PDF, Word, Excel và PowerPoint.
- Với PDF scan, dùng OCR để lấy nội dung chữ.
- Lưu thêm thông tin như tên file, ngày upload, người upload và loại file.

## Chia nhỏ tài liệu

- Chia tài liệu thành các đoạn nhỏ trước khi đưa vào vector database.
- Không nên chia quá ngắn vì dễ mất ngữ cảnh.
- Có thể chia theo tiêu đề, đoạn văn hoặc từng phần trong tài liệu.

## Tìm kiếm tốt hơn

- Kết hợp tìm kiếm theo từ khóa và semantic search.
- Lấy nhiều kết quả liên quan rồi chọn các đoạn phù hợp nhất.
- Ưu tiên tài liệu mới hơn hoặc tài liệu chính thức hơn.

## Giảm việc chatbot trả lời sai

- Chỉ cho chatbot trả lời dựa trên nội dung tìm được.
- Luôn hiển thị nguồn hoặc link tới tài liệu gốc.
- Nếu không tìm được thông tin phù hợp, chatbot nên nói không tìm thấy thay vì tự đoán.

## Bảo mật

- Kiểm tra quyền người dùng trước khi cho tìm kiếm tài liệu.
- Chỉ trả về những tài liệu mà người dùng được phép xem.
- Lưu log để biết ai đã tìm kiếm hoặc xem tài liệu nào.

## Cải thiện dần

- Cho người dùng đánh giá câu trả lời tốt hay không tốt.
- Lưu các câu hỏi chatbot trả lời chưa tốt.
- Dùng các câu hỏi đó để cải thiện chunking, search và prompt.
