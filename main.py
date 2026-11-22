"""Trang chính của ứng dụng phân tích dữ liệu Streamlit."""

from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Student_Performance.csv"

st.set_page_config(
    page_title="Hệ thống phân tích dữ liệu",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Hệ thống phân tích dữ liệu")
st.write(
    "Ứng dụng hỗ trợ tải lên, làm sạch, phân tích, dự đoán và xuất báo cáo. "
    "Chọn một chức năng ở thanh điều hướng bên trái để bắt đầu."
)

st.subheader("Các chức năng")
st.markdown(
    """
    - **Upload dữ liệu**: tải tệp CSV hoặc Excel lên hệ thống.
    - **Quản lý dữ liệu**: xem và tìm kiếm dữ liệu.
    - **Làm sạch dữ liệu**: xử lý dữ liệu thiếu, trùng lặp và ngoại lệ.
    - **Phân tích dữ liệu**: xem thống kê mô tả, biểu đồ và tương quan.
    - **Prediction**: dự đoán kết quả bằng hồi quy tuyến tính.
    - **Reports**: xem tổng hợp và tải báo cáo Excel.

    Các trang chức năng nằm trong thư mục `pages/`; Streamlit sẽ tự hiển thị
    chúng trong thanh điều hướng khi chạy ứng dụng này.
    """
)

if not DEFAULT_DATA_PATH.is_file():
    st.warning(
        "Chưa tìm thấy dữ liệu mẫu `data/Student_Performance.csv`. "
        "Hãy tải dữ liệu lên ở trang Upload Data để dùng các trang khác."
    )

st.caption("Chạy ứng dụng từ thư mục gốc bằng lệnh: `streamlit run main.py`.")
