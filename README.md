# Data-Analysis-System

## Chạy ứng dụng

Yêu cầu Python 3.10 trở lên. Từ thư mục gốc của dự án, tạo môi trường và cài
các thư viện (nếu chưa cài):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Sau đó chạy ứng dụng:

```powershell
streamlit run pages\5_Data_Analysis.py
```

Trang **Phân tích dữ liệu** dùng `data\Student_Performance.csv` làm dữ liệu mặc định.
Bạn cũng có thể tải lên một tệp CSV khác. Trang cung cấp thống kê tổng quan và mô
tả, trung bình/tổng/lớn nhất/nhỏ nhất, biểu đồ cột/đường/tròn/phân tán, ma trận
tương quan và bảng dữ liệu có thể tải xuống.

## Kiểm thử

```powershell
python -m unittest discover -s data -p "test_*.py"
```