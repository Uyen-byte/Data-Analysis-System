import streamlit as st

from database.dataset_repository import DatasetRepository
from modules.data_loader import load_file, validate_and_clean

st.set_page_config(page_title="Upload Data", page_icon="📤", layout="wide")
st.title("📤 Upload dữ liệu")
st.write("Tải dữ liệu kết quả học tập từ file CSV hoặc Excel lên PostgreSQL.")

uploaded_file = st.file_uploader("Chọn file dữ liệu", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        raw_data = load_file(uploaded_file)
        clean_data = validate_and_clean(raw_data)

        # Cập nhật session_state để các module tiền xử lý tiếp theo của nhóm có thể sử dụng
        st.session_state["dataset"] = clean_data.copy()

        col1, col2 = st.columns(2)
        col1.metric("Số dòng ban đầu", f"{len(raw_data):,}")
        col2.metric("Số dòng sau khi loại trùng", f"{len(clean_data):,}")

        st.subheader("Xem trước dữ liệu")
        st.dataframe(clean_data.head(20), use_container_width=True)

        if st.button("Tải dữ liệu lên hệ thống", type="primary"):
            repository = DatasetRepository()
            repository.create_table()
            saved_count = repository.save_dataframe(clean_data)
            st.success(f"Đã tải {saved_count:,} bản ghi lên PostgreSQL thành công!")

    except Exception as error:
        st.error(f"Không thể xử lý file: {error}")