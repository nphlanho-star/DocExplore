
# Features, pain points và edge cases

## Feature đề xuất

| Nhóm | Feature | Giá trị |
| --- | --- | --- |
| Nguồn dữ liệu | Upload file, kết nối Drive, SharePoint, S3, wiki | Gom toàn bộ tri thức về một nơi |
| Xử lý tài liệu | Parser theo định dạng, OCR cho scan, nhận diện bảng/heading | Biến file thô thành dữ liệu có thể truy xuất |
| Tìm kiếm | Hybrid search, metadata filter, semantic search, reranking | Tăng khả năng tìm đúng nội dung |
| Hỏi đáp | Chat đa lượt, citation, xem đoạn nguồn, không đủ bằng chứng | Câu trả lời đáng tin và dễ kiểm chứng |
| Quản trị | Quản lý connector, trạng thái ingest, version tài liệu | Dễ vận hành kho dữ liệu lớn |
| Chất lượng | Feedback, dashboard, bộ câu hỏi benchmark | Cải thiện dựa trên dữ liệu |
| Bảo mật | SSO, RBAC, ACL, audit log, PII protection | Bảo vệ dữ liệu nội bộ |

## Pain points

| Pain point | Vì sao khó | Rủi ro |
| --- | --- | --- |
| Tài liệu dài và nhiều định dạng | PDF, Word, slide, bảng tính, scan có cấu trúc khác nhau | Chunk sai, mất nội dung quan trọng |
| Tài liệu cũ hoặc mâu thuẫn | Có nhiều version và nhiều nguồn | Trả lời thông tin lỗi thời |
| Retrieval chưa chính xác | Câu hỏi ngắn, typo, acronym, đa ngôn ngữ | Hallucination hoặc câu trả lời lan man |
| Mất ngữ cảnh khi chunk | Quy tắc/bảng có thể kéo dài nhiều trang | Thiếu điều kiện và ngoại lệ |
| Quyền truy cập phức tạp | ACL theo file, thư mục, nhóm hoặc thời điểm | Rò rỉ dữ liệu nhạy cảm |
| Tài liệu cập nhật liên tục | Vector index có thể stale hoặc trùng dữ liệu | Người dùng mất niềm tin |
| Chi phí và độ trễ | OCR, embedding, rerank và LLM đều tốn tài nguyên | Hệ thống chậm hoặc vượt ngân sách |
| Thiếu đo lường chất lượng | Không thể đánh giá bằng demo cảm tính | Không biết cần tối ưu khâu nào |
| Prompt injection từ tài liệu | File có thể chứa chỉ dẫn độc hại | Agent bị thao túng hoặc lộ dữ liệu |

## Edge cases cần test

1. PDF scan bị xoay, mờ hoặc OCR sai.
2. Bảng kéo dài qua nhiều trang và mất header cột.
3. File hỏng, có mật khẩu, quá lớn hoặc định dạng không hỗ trợ.
4. Hai tài liệu cùng tên nhưng nội dung khác nhau.
5. Chính sách cũ có độ tương đồng cao hơn chính sách mới.
6. Người dùng vừa bị thu hồi quyền nhưng cache vẫn trả dữ liệu.
7. Câu hỏi pha tiếng Việt, tiếng Anh, acronym và typo.
8. Câu hỏi mơ hồ, thiếu phòng ban, quốc gia hoặc sản phẩm.
9. Nguồn trả về mâu thuẫn hoặc không đủ bằng chứng.
10. Người dùng yêu cầu tổng hợp hàng trăm tài liệu.
11. Nội dung tài liệu chứa prompt injection.
12. Dịch vụ embedding hoặc LLM timeout khi ingest tăng đột biến.

## Acceptance criteria cho MVP

- Mỗi câu trả lời factual có ít nhất một citation.
- ACL được kiểm tra khi query, không chỉ khi ingest.
- Không đủ evidence thì hỏi làm rõ hoặc từ chối trả lời.
- UI hiển thị trạng thái ingest, version và lỗi.
- Có benchmark gồm câu hỏi đúng, không thể trả lời và câu hỏi nhạy cảm theo quyền.
