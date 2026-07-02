# 📊 Chấm điểm tín dụng theo khung 5C — Dự báo xác suất vỡ nợ (PD)

Ứng dụng Streamlit tái hiện quy trình xây dựng mô hình trong notebook `kiểm_định_5c.ipynb`:
huấn luyện một mô hình **Hồi quy Logistic (Logistic Regression)** để dự báo khả năng vỡ nợ
(`PD`: 0 = không vỡ nợ, 1 = vỡ nợ) của khách hàng, dựa trên **24 biến khảo sát** (thang Likert 1–5)
thuộc 5 nhóm tiêu chí tín dụng "5C":

| Nhóm | Ý nghĩa | Các cột |
|---|---|---|
| TC | Tư cách (Character) | TC1–TC5 |
| NL | Năng lực (Capacity) | NL1–NL4 |
| DK | Điều kiện (Conditions) | DK1–DK5 |
| V | Vốn (Capital) | V1–V6 |
| TS | Tài sản đảm bảo (Collateral) | TS1–TS4 |

Notebook gốc không dùng bất kỳ bước tiền xử lý nào (không scaler/encoder) — mô hình được huấn
luyện trực tiếp trên 24 biến số nguyên (1–5), với `train_test_split(test_size=0.2, random_state=23)`
và `LogisticRegression()` (tham số mặc định của scikit-learn). Ứng dụng cho phép người dùng tuỳ
chỉnh các tham số này ở thanh bên; giá trị mặc định trùng khớp với notebook.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

## Cấu trúc dữ liệu đầu vào

Tệp CSV cần có tối thiểu các cột sau (các cột khác như `Dấu thời gian`, `NN` nếu có sẽ bị bỏ qua
khi huấn luyện, vì notebook gốc không sử dụng chúng làm biến đầu vào của mô hình):

- 24 cột biến đầu vào (giá trị nguyên 1–5): `TC1, TC2, TC3, TC4, TC5, NL1, NL2, NL3, NL4, DK1, DK2, DK3, DK4, DK5, V1, V2, V3, V4, V5, V6, TS1, TS2, TS3, TS4`
- 1 cột biến mục tiêu: `PD` (0 hoặc 1)

Dữ liệu mẫu tham khảo: `5c__1_.csv` (150 dòng, 27 cột, gồm cả `Dấu thời gian` và `NN` không dùng
trong mô hình).

## Mô tả các tab

- **⚙️ Sidebar — Cấu hình & Tải dữ liệu**: tải tệp CSV, chỉnh tham số mô hình
  (`C`, `solver`, `max_iter`), tham số nâng cao (`test_size`, `random_state` chia tập, `random_state`
  mô hình), và nút huấn luyện.
- **📋 Tổng quan dữ liệu**: kích thước dữ liệu, xem nhanh bảng dữ liệu thô, thống kê mô tả 24 biến
  đầu vào và biến mục tiêu `PD`.
- **📈 Trực quan hóa dữ liệu**: biểu đồ phân phối biến mục tiêu `PD` và tối đa 3 biến đầu vào
  do người dùng chọn (mặc định là 3 biến có tương quan tuyệt đối cao nhất với `PD`).
- **🎯 Kết quả huấn luyện & kiểm định**: Accuracy, Precision, Recall, F1-score, ROC-AUC, ma trận
  nhầm lẫn, đường cong ROC, báo cáo phân loại chi tiết, và bảng kết quả dự báo trên tập kiểm tra.
- **🔮 Sử dụng mô hình**: dự báo cho một khách hàng mới (nhập tay từng câu trả lời khảo sát) hoặc
  dự báo hàng loạt bằng cách tải lên tệp CSV có đúng 24 cột đầu vào, kèm tải xuống kết quả.

## Ghi chú

- Mô hình chỉ huấn luyện lại khi người dùng bấm nút "🚀 Huấn luyện & Kiểm định mô hình"; kết quả
  được lưu trong `st.session_state` nên chuyển tab không làm mất kết quả hay huấn luyện lại.
- Khuyến nghị dùng phiên bản Streamlit mới (`>=1.38`) để tương thích tốt nhất với `st.container(height=...)`
  và các thành phần bố cục đã dùng trong ứng dụng.
