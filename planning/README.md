# RAG cho kho tài liệu lớn và đa dạng

Tài liệu planning cho web app có tính năng chính là Retrieval-Augmented Generation (RAG): người dùng hỏi bằng ngôn ngữ tự nhiên, hệ thống truy xuất phần nội dung đáng tin cậy từ kho tài liệu dài, nhiều định dạng và trả lời kèm trích dẫn.

## Mục tiêu sản phẩm

- Tìm đúng thông tin trong PDF, Word, PowerPoint, bảng tính, trang wiki và tài liệu scan.
- Trả lời có nguồn dẫn, trang/đoạn liên quan và mức độ tin cậy.
- Tôn trọng quyền truy cập theo người dùng, nhóm và tài liệu.
- Đồng bộ thay đổi tài liệu mà không làm kết quả bị lỗi thời.

## Phạm vi MVP

1. Đăng nhập và quản lý quyền truy cập.
2. Upload/kết nối nguồn tài liệu, theo dõi trạng thái ingest.
3. Hybrid search: keyword search kết hợp vector search.
4. Chat hỏi đáp có citation mở được đúng vị trí nguồn.
5. Dashboard quản trị ingest, lỗi và feedback người dùng.

## Tài liệu liên quan

- `features-pain-points-edge-cases.md`: Danh sách feature, pain point và edge case.
- `solution-directions.md`: Hướng giải quyết và lộ trình triển khai.
- `workflow.md`: Workflow RAG end-to-end bằng Mermaid.

## Nguyên tắc thiết kế

- Retrieval trước generation: chỉ trả lời khi có bằng chứng đủ tốt.
- Security by design: luôn kiểm tra quyền trước khi trả dữ liệu cho người dùng.
- Có thể đo lường: lưu trace, citation, feedback và benchmark.
- Ưu tiên MVP có thể kiểm chứng trước khi mở rộng quy mô.

## KPI gợi ý

| Nhóm | Chỉ số |
| --- | --- |
| Retrieval | Recall@k, MRR, nDCG |
| Câu trả lời | Citation precision, groundedness, tỷ lệ từ chối đúng |
| Vận hành | Ingest success rate, p95 latency, thời gian cập nhật index |
| Sản phẩm | Helpful-answer rate, tỷ lệ feedback tích cực |
