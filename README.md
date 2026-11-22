# Data-Analysis-System

## Chạy toàn bộ ứng dụng

Yêu cầu Python 3.10 trở lên. Từ thư mục gốc của dự án, tạo môi trường và cài
các thư viện (nếu chưa cài):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Sau đó chạy từ thư mục gốc:

```powershell
streamlit run main.py
```

Các trang **Upload Data**, **Data Management**, **Data Cleaning**,
**Data Analysis**, **Prediction** và **Reports** sẽ xuất hiện ở thanh điều hướng.
Trang phân tích dùng `data\Student_Performance.csv` làm dữ liệu mặc định; các
trang làm sạch, dự đoán và báo cáo dùng dữ liệu trong phiên làm việc.

Các trang **Prediction**, **Reports**, **Data Cleaning** và **Data Analysis**
có thể chạy với dữ liệu trong phiên làm việc. Trang **Upload Data** chỉ cần
PostgreSQL khi bấm lưu dữ liệu; trang **Data Management** cần PostgreSQL để đọc
và quản lý dữ liệu.

Để dùng PostgreSQL, tạo `database\.env` với các biến sau (thay bằng thông tin
của máy bạn):

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=student_performance
DB_USER=postgres
DB_PASSWORD=your_password
```

Không tải file `.env` lên GitHub. File này đã được loại khỏi Git bằng
`.gitignore`.

## Kiểm thử

```powershell
python -m unittest discover -s data -p "test_*.py"
```