# CHIẾN LƯỢC TỐI ƯU HÓA PROMPT (PROMPT ENGINEERING STRATEGY V2)

## 1. Triết Lý Thiết Kế: "Cognitive Chain of Investigation" (Chuỗi Tư Duy Điều Tra)
Thay vì yêu cầu LLM "làm tất cả cùng lúc", ta chia nhỏ quy trình thành các bước tư duy tuần tự (Chain of Thought) để tối ưu hóa độ sâu tri thức.

### Mô hình 4 Bước Chiến Lược:
1.  **Bước 1: Perception (Nhận thức ngữ cảnh)**
    *   *Mục tiêu*: Xác định loại hình hội thoại (Giao dịch, Tình cảm, Xung đột, Tội phạm).
    *   *Kỹ thuật*: Context Anchoring.
2.  **Bước 2: Extraction (Trích xuất dữ liệu thô)**
    *   *Mục tiêu*: Đầy đủ 5W1H, Con số, Thực thể.
    *   *Kỹ thuật*: Entity Linking.
3.  **Bước 3: Profiling (Phân tích chiều sâu)**
    *   *Mục tiêu*: Áp dụng tri thức chuyên gia (SVA, SCAN, Psychology VN) để "đọc vị" đối tượng.
    *   *Kỹ thuật*: Expert Persona (Đóng vai chuyên gia).
4.  **Bước 4: Synthesis (Tổng hợp chiến lược)**
    *   *Mục tiêu*: Viết báo cáo điều hành (Executive Brief) định lượng và định tính.
    *   *Kỹ thuật*: Structured Summarization.

## 2. Tổng Hợp Tri Thức (Knowledge Integration)
*   **SVA (Statement Validity Analysis)**: Tự động đánh giá độ tin cậy lời khai dựa trên 19 tiêu chí.
*   **SCAN (Scientific Content Analysis)**: Phân tích ngôn ngữ để tìm sự lảng tránh.
*   **Vietnamese Psychology**: Văn hóa "giữ kẽ", "nói tránh", "giả danh" trong tội phạm Việt Nam.

## 3. Cấu Trúc Master Prompt Mới
```jinja2
SYSTEM: Bạn là Tổng Chỉ Huy Phân Tích (Chief Intelligence Officer).
Quy trình tư duy:
1.  [Suy luận] Đọc lướt để nắm ngữ cảnh.
2.  [Khai thác] Kích hoạt module 5W1H.
3.  [Phân tích] Kích hoạt module Tâm lý/SVA.
4.  [Báo cáo] Tổng hợp Strategic Report.
```
