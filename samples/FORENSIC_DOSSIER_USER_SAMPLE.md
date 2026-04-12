# HỒ SƠ PHÂN TÍCH FORENSIC (MẪU: TEST_AUDIO.MP3)

Tài liệu này minh họa chi tiết tác dụng của hệ thống **Deep Prompting** lên dữ liệu mẫu của người dùng. Hệ thống không chỉ "ghi chép" mà thực sự "tư duy" để trích xuất tình báo.

---

## 1. Dữ Liệu Đầu Vào (Transcript)
> **User**: "Tôi muốn đặt phòng khách sạn cho chuyến công tác vào ngày mai. Cho tôi hỏi phòng có bao gồm bữa sáng không?"

---

## 2. Phân Tích Tác Dụng Của Từng Module Prompt

### 🧩 Module 1: Narrative Intelligence (Tóm tắt Tình báo)
*   **Prompt**: "Đừng chỉ tóm tắt. Hãy đánh giá ý nghĩa, mục đích và viết dạng báo cáo nghiệp vụ."
*   **Kết Quả Thực Tế**:
    > "Đối tượng chủ động liên hệ đặt phòng phục vụ chuyến công tác... Đây là một hoạt động di chuyển theo kế hoạch (planned movement) của nhóm đối tượng văn phòng."
*   **Tác Dụng**: Chuyển hóa từ câu nói đơn giản thành một nhận định nghiệp vụ (đánh giá mức độ chủ động và phân loại hành vi).

### 🧩 Module 2: Deep Relationships (Mạng lưới Quan hệ)
*   **Prompt**: "Xác định rõ vai trò và mạng lưới quan hệ (Relationship Network) thay vì chỉ gán nhãn."
*   **Kết Quả Thực Tế**:
    *   **Quyên**: "Nhân sự đi công tác (Quan hệ với tổ chức/công ty chưa xác định)".
    *   **Lễ tân**: "Đại diện cơ sở lưu trú".
*   **Tác Dụng**: Hệ thống tự suy luận ra mối quan hệ "Người đi công tác - Tổ chức" dù trong câu không hề nhắc đến tên công ty.

### 🧩 Module 3: Shadow Profiling (Hồ sơ Ngầm)
*   **Prompt**: "Nếu là hội thoại bình thường, hãy trích xuất thói quen (Life Patterns) và tài chính."
*   **Kết Quả Thực Tế**:
    *   **Thói quen**: "Đang trong chu kỳ di chuyển thường xuyên."
    *   **Sở thích**: "Ưu tiên sự tiện lợi (Bữa sáng tại chỗ)."
*   **Tác Dụng**: Biến một yêu cầu "hỏi bữa sáng" bình thường thành dữ liệu về thói quen/hành vi tiêu dùng.

---

## 3. Tổng Hợp Tình Báo (Capability Matrix)

| Loại Thông Tin | Trước khi có Prompt mới | Sau khi kịch hoạt Prompt "Omniscient" |
| :--- | :--- | :--- |
| **Bản chất** | "Người dùng đặt phòng" | "Hoạt động di chuyển theo kế hoạch (Planned Movement)" |
| **Quan hệ** | "Khách hàng" | "Nhân sự công tác - Đại diện lưu trú" |
| **Rủi ro** | "An toàn" | "Low Risk (Nhưng tiếp tục giám sát lịch trình)" |
| **Định lượng** | "1 phòng" | "Thời gian: Ngày mai (Immediate Future)" |

---

## 4. File Kết Quả Nguyên Bản (JSON)
*(Dữ liệu này được sinh ra tự động bởi hệ thống)*

```json
{
  "header": {
    "intelligence_narrative": "Bản tóm tắt tình báo: Đối tượng (Người dùng) chủ động liên hệ đặt phòng khách sạn phục vụ chuyến công tác sắp tới...",
    "risk_level": "LOW (Shadow Profiling Active)"
  },
  "shadow_profile": {
    "life_patterns": {
      "travel": "Đang trong chu kỳ di chuyển công tác.",
      "preferences": "Ưu tiên sự tiện lợi (Bữa sáng tại chỗ)."
    }
  }
}
```
