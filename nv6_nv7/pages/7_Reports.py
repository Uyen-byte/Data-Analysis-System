"""
Reports Page - Trang tổng hợp thông tin dữ liệu, thống kê mô tả và tải báo cáo Excel.
- Tổng quan kích thước dữ liệu và số ô dữ liệu thiếu
- Xem trước dữ liệu phân tích và dữ liệu gốc
- Thống kê mô tả (Descriptive Statistics)
- Hiển thị kết quả dự đoán (Prediction) & đánh giá mô hình (Model Metrics)
- Xuất báo cáo tổng hợp định dạng Excel (.xlsx)
"""

import streamlit as st
import pandas as pd

from modules.report import ReportService


def configure_page():
    """Thiết lập cấu hình giao diện trang Streamlit."""
    st.set_page_config(
        page_title="Reports",
        page_icon="📊",
        layout="wide"
    )
    st.title("📊 Báo cáo tổng hợp (Reports)")
    st.caption("Xem dữ liệu, thống kê, kết quả dự đoán và xuất Excel.")


def load_session_data():
    """Trích xuất và chuẩn hóa dữ liệu cùng các kết quả mô hình từ st.session_state."""
    # Dữ liệu gốc
    df = st.session_state.get("df")
    if not isinstance(df, pd.DataFrame):
        df = st.session_state.get("data")

    # Dữ liệu đã Cleaning
    cleaned_df = st.session_state.get("cleaned_df")
    if not isinstance(cleaned_df, pd.DataFrame):
        cleaned_df = st.session_state.get("cleaned_data")

    if not isinstance(cleaned_df, pd.DataFrame) or cleaned_df.empty:
        cleaned_df = None

    if not isinstance(df, pd.DataFrame) or df.empty:
        if cleaned_df is not None:
            df = cleaned_df.copy()
        else:
            st.warning(
                "Chưa có dữ liệu. "
                "Vui lòng tải dữ liệu ở trang Quản lý dữ liệu."
            )
            st.stop()

    # Kết quả Prediction
    prediction_table = st.session_state.get("prediction_table")
    if not isinstance(prediction_table, pd.DataFrame):
        prediction_table = pd.DataFrame()

    # Đồng bộ Metrics mô hình từ trang Prediction (ưu tiên prediction_metrics)
    model_metrics = st.session_state.get(
        "prediction_metrics",
        st.session_state.get("model_metrics", {})
    )
    
    # Lấy tên mô hình từ session_state hoặc từ đối tượng service đã huấn luyện
    trained_service = st.session_state.get("trained_prediction_service")
    default_model_name = "Linear Regression" if trained_service is not None else "Chưa xác định"
    model_name = st.session_state.get("model_name", default_model_name)

    return df, cleaned_df, prediction_table, model_metrics, model_name

def remove_identifier_columns(df: pd.DataFrame):
    """Loại bỏ các cột định danh như student_id khỏi dữ liệu báo cáo."""

    if not isinstance(df, pd.DataFrame):
        return df

    excluded_keywords = ["id", "code", "index", "key", "stt"]

    columns_to_remove = []

    for col in df.columns:
        col_lower = str(col).strip().lower()

        column_parts = (
            col_lower
            .replace("-", "_")
            .replace(" ", "_")
            .split("_")
        )

        is_identifier = any(
            keyword in column_parts
            for keyword in excluded_keywords
        )

        if is_identifier:
            columns_to_remove.append(col)

    return df.drop(columns=columns_to_remove).copy()

def render_summary_section(report: ReportService, cleaned_df):
    """Hiển thị thông tin tổng quan (số dòng, số cột, số ô dữ liệu thiếu)."""
    st.subheader("1. Thông tin tổng quan")
    summary = report.get_summary_info()

    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng số dòng", f"{summary['rows']:,}")
    c2.metric("Tổng số cột", f"{summary['cols']:,}")
    c3.metric("Số ô dữ liệu thiếu", f"{summary['missing_cells']:,}")

    if cleaned_df is not None:
        st.success("Đang sử dụng dữ liệu đã làm sạch từ Cleaning.")
    else:
        st.info("Chưa có dữ liệu Cleaning. Báo cáo đang sử dụng dữ liệu gốc.")


def render_data_preview(analysis_df: pd.DataFrame, raw_df: pd.DataFrame):
    """Hiển thị bảng xem trước dữ liệu phân tích và dữ liệu gốc."""
    st.subheader("2. Xem trước dữ liệu")

    tab1, tab2 = st.tabs(["Dữ liệu phân tích", "Dữ liệu gốc"])

    with tab1:
        st.caption("Dữ liệu được sử dụng cho thống kê và báo cáo.")
        st.dataframe(analysis_df.head(100), use_container_width=True)

    with tab2:
        st.dataframe(raw_df.head(100), use_container_width=True)


def render_statistics_section(report: ReportService):
    """Hiển thị bảng thống kê mô tả của dữ liệu."""
    st.subheader("3. Thống kê mô tả")

    try:
        stats_df = report.generate_statistics()
        if stats_df.empty:
            st.warning("Không có dữ liệu để thống kê.")
        else:
            st.dataframe(stats_df, use_container_width=True)
    except Exception as err:
        st.error(f"Không thể tạo thống kê: {err}")


def render_prediction_section(report: ReportService):
    """Hiển thị bảng kết quả dự đoán và nút tải file CSV riêng."""
    st.subheader("4. Kết quả dự đoán")

    if not report.prediction_df.empty:
        st.dataframe(report.prediction_df, use_container_width=True)

        st.download_button(
            "⬇️ Tải riêng kết quả Prediction",
            data=report.export_prediction_csv(),
            file_name="prediction_results.csv",
            mime="text/csv"
        )
    else:
        st.info(
            "Chưa có kết quả dự đoán. "
            "Hãy huấn luyện mô hình tại trang Prediction."
        )


def render_metrics_section(report: ReportService):
    """Hiển thị các chỉ số đánh giá hiệu năng mô hình (MAE, RMSE, R2)."""
    st.subheader("5. Đánh giá mô hình")

    if report.model_metrics:
        st.write(f"**Mô hình:** {report.model_name}")

        metric_cols = st.columns(3)

        if "MAE" in report.model_metrics:
            metric_cols[0].metric("MAE", f"{report.model_metrics['MAE']:.4f}")

        if "RMSE" in report.model_metrics:
            metric_cols[1].metric("RMSE", f"{report.model_metrics['RMSE']:.4f}")

        if "R2" in report.model_metrics:
            metric_cols[2].metric("R²", f"{report.model_metrics['R2']:.4f}")

        st.dataframe(
            report.get_metrics_dataframe(),
            use_container_width=True
        )
    else:
        st.info("Chưa có thông tin đánh giá mô hình.")


def render_excel_export_section(report: ReportService):
    """Hiển thị khu vực xuất báo cáo tổng hợp ra tệp Excel (.xlsx)."""
    st.divider()
    st.subheader("6. Xuất báo cáo Excel")

    try:
        excel_data = report.export_to_excel()
        file_name = ReportService.create_filename()

        st.download_button(
            label="⬇️ Tải xuống báo cáo Excel (.xlsx)",
            data=excel_data,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

        st.caption(
            "File Excel gồm Data, Cleaned Data, Statistics, "
            "Prediction và Model Metrics."
        )

    except Exception as err:
        st.error(f"Không thể tạo tệp báo cáo Excel: {err}")

def main():
    """Hàm chính điều phối toàn bộ luồng xử lý và hiển thị trang Reports."""
    configure_page()

    df, cleaned_df, prediction_table, model_metrics, model_name = load_session_data()

    # Loại bỏ các cột định danh khỏi dữ liệu báo cáo
    df = remove_identifier_columns(df)

    if cleaned_df is not None:
        cleaned_df = remove_identifier_columns(cleaned_df)

    # Lựa chọn dữ liệu phân tích
    analysis_df = cleaned_df if cleaned_df is not None else df

    # Khởi tạo ReportService
    report = ReportService(
        df=df,
        cleaned_df=cleaned_df,
        prediction_df=prediction_table,
        model_metrics=model_metrics,
        model_name=model_name
    )

    # Hiển thị các thành phần báo cáo
    render_summary_section(report, cleaned_df)
    render_data_preview(analysis_df, df)
    render_statistics_section(report)
    render_prediction_section(report)
    render_metrics_section(report)
    render_excel_export_section(report)


if __name__ == "__main__":
    main()
