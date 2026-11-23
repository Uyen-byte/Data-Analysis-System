import hashlib

import pandas as pd
import streamlit as st

from modules.data_loader import load_file, validate_dataset

st.set_page_config(page_title="Upload Data", page_icon="📤", layout="wide")
st.title("📤 Upload dữ liệu")
st.write(
    "Tải dữ liệu kết quả học tập từ CSV hoặc Excel. "
    "Dữ liệu được giữ nguyên để bạn xử lý tại trang Data Cleaning trước khi lưu."
)
uploaded_file = st.file_uploader(
    "Chọn file dữ liệu",
    type=["csv", "xlsx", "xls"],
    key="upload_data_file",
)
uploaded_dataset = st.session_state.get("uploaded_dataset")

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        file_signature = (
            uploaded_file.name
            + ":"
            + hashlib.sha256(file_bytes).hexdigest()
        )

        if file_signature != st.session_state.get("_uploaded_file_signature"):
            dataset = validate_dataset(load_file(uploaded_file))
            st.session_state["uploaded_dataset"] = dataset.copy()
            st.session_state["uploaded_file_bytes"] = file_bytes
            st.session_state["uploaded_file_name"] = uploaded_file.name
            st.session_state["_uploaded_file_signature"] = file_signature
            st.session_state["dataset"] = dataset.copy()

            for key in (
                "cleaned_dataset",
                "data_cleaner",
                "_cleaning_source",
                "trained_prediction_service",
                "prediction_table",
                "prediction_metrics",
                "prediction_target",
                "prediction_features",
                "model_name",
                "analysis_uploaded_dataset",
                "analysis_uploaded_file_name",
            ):
                st.session_state.pop(key, None)

        uploaded_dataset = st.session_state.get("uploaded_dataset")

    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Không thể xử lý file: {error}")

if isinstance(uploaded_dataset, pd.DataFrame):
    if not isinstance(st.session_state.get("dataset"), pd.DataFrame):
        st.session_state["dataset"] = uploaded_dataset.copy()

    metric_columns = st.columns(2)
    metric_columns[0].metric("Số dòng được tải", f"{len(uploaded_dataset):,}")
    metric_columns[1].metric("Số cột", f"{len(uploaded_dataset.columns):,}")

    file_name = st.session_state.get("uploaded_file_name", "tệp đã tải")
    st.caption(f"Đang giữ dữ liệu từ `{file_name}` trong phiên làm việc.")
    st.subheader("Xem trước dữ liệu gốc")
    st.dataframe(uploaded_dataset.head(20), use_container_width=True)
