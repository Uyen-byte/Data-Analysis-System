"""
Prediction Page - Ứng dụng Dự đoán Dữ liệu Machine Learning
- Mô hình: Linear Regression / Ridge Regression
- Hỗ trợ: Numeric + Categorical Features
- Đánh giá: MAE, RMSE, R2
- Trực quan hóa: Biểu đồ Actual vs Predicted (Line & Scatter)
- Dự đoán mẫu dữ liệu mới
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modules.prediction import PredictionService


def configure_page():
    """Thiết lập cấu hình giao diện trang Streamlit."""
    st.set_page_config(
        page_title="Prediction",
        page_icon="🔮",
        layout="wide"
    )
    st.title("🔮 Dự đoán dữ liệu (Prediction)")
    st.caption(
        "Huấn luyện mô hình Machine Learning, "
        "đánh giá và dự đoán dữ liệu mới."
    )


def load_dataset():
    """Lấy dữ liệu từ st.session_state theo thứ tự ưu tiên."""
    return st.session_state.get(
        "cleaned_df",
        st.session_state.get(
            "df",
            st.session_state.get("data")
        )
    )

def render_model_parameters(service: PredictionService):
    """Thiết lập tham số mô hình."""

    # Lấy danh sách cột số
    numeric_cols = service.get_target_candidates()

    # Chỉ loại student_id khỏi danh sách Target
    target_candidates = [
        col for col in numeric_cols
        if str(col).strip().lower() != "student_id"
    ]

    if not target_candidates:
        st.error("Không có cột phù hợp làm Target.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        target_col = st.selectbox(
            "🎯 Target - Cột cần dự đoán",
            options=target_candidates
        )

    with col2:
        model_type = st.selectbox(
            "🤖 Chọn mô hình",
            ["Linear Regression", "Ridge Regression"]
        )

    # Lấy danh sách Features
    feature_candidates = (
        service.get_feature_candidates(target_col)
        if hasattr(service, "get_feature_candidates")
        else [
            c for c in service.raw_df.columns
            if c != target_col
        ]
    )

    # Giữ nguyên student_id trong Features nếu dữ liệu có cột này
    feature_candidates = [
        col for col in feature_candidates
        if col != target_col
    ]

    default_features = feature_candidates[:2]

    feature_cols = st.multiselect(
        "📌 Features - Các cột đầu vào",
        options=feature_candidates,
        default=default_features
    )

    test_percent = st.slider(
        "Tỷ lệ dữ liệu kiểm thử (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    )

    st.info(
        "Target là biến cần dự đoán. "
        "Cột student_id chỉ bị loại khỏi danh sách Target."
    )

    return target_col, model_type, feature_cols, test_percent


    # Hàm kiểm tra cột định danh
 def is_identifier_column(col):
        col_lower = str(col).strip().lower()

        excluded_keywords = [
            "id",
            "code",
            "index",
            "key",
            "stt"
        ]

        # Tách tên cột thành các thành phần
        column_parts = col_lower.replace("-", "_").split("_")

        return any(
            keyword in column_parts
            for keyword in excluded_keywords
        )

    # Loại bỏ cột ID khỏi danh sách Target
    target_candidates = [
        col for col in numeric_cols
        if not is_identifier_column(col)
    ]

    # Không cho phép ID quay trở lại danh sách Target
    if not target_candidates:
        st.error(
            "Dataset không có cột số phù hợp làm Target. "
            "Các cột định danh như student_id đã được loại bỏ. "
            "Hãy kiểm tra dữ liệu."
        )
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        target_col = st.selectbox(
            "🎯 Target - Cột cần dự đoán",
            target_candidates
        )

    with col2:
        model_type = st.selectbox(
            "🤖 Chọn mô hình",
            ["Linear Regression", "Ridge Regression"]
        )

    # Lấy danh sách Features
    feature_candidates = (
        service.get_feature_candidates(target_col)
        if hasattr(service, "get_feature_candidates")
        else [
            c for c in service.raw_df.columns
            if c != target_col
        ]
    )

    # Loại bỏ cột ID khỏi Features
    feature_candidates = [
        col for col in feature_candidates
        if not is_identifier_column(col)
        and col != target_col
    ]

    default_features = feature_candidates[:2]

    feature_cols = st.multiselect(
        "📌 Features - Các cột đầu vào",
        options=feature_candidates,
        default=default_features
    )

    test_percent = st.slider(
        "Tỷ lệ dữ liệu kiểm thử (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    )

    st.info(
        "Target là biến cần dự đoán. "
        "Các cột định danh (như student_id) "
        "đã được hệ thống tự động loại bỏ."
    )

    return target_col, model_type, feature_cols, test_percent

def handle_training(service: PredictionService, feature_cols, target_col, test_percent, model_type):
    """Xử lý sự kiện huấn luyện mô hình với cơ chế bắt ngoại lệ."""
    if st.button("🚀 Huấn luyện mô hình", type="primary"):
        try:
            if not feature_cols:
                raise ValueError("Vui lòng chọn ít nhất một Feature.")

            if target_col in feature_cols:
                raise ValueError("Target không được nằm trong Features.")

            service.train_and_evaluate(
                feature_cols=feature_cols,
                target_col=target_col,
                test_size=test_percent / 100,
                model_type=model_type
            )

            # Lưu trữ service và kết quả vào session_state
            st.session_state["trained_prediction_service"] = service
            st.session_state["prediction_table"] = service.result_df.copy()
            st.session_state["prediction_metrics"] = service.metrics.copy()
            st.session_state["prediction_target"] = target_col
            st.session_state["prediction_features"] = feature_cols

            st.success("Huấn luyện mô hình thành công!")

        except Exception as err:
            st.error(f"Huấn luyện thất bại: {err}")


def render_evaluation_metrics(trained_service: PredictionService):
    """Hiển thị các chỉ số đánh giá mô hình (MAE, RMSE, R2)."""
    st.divider()
    st.subheader("2. Kết quả đánh giá mô hình")

    metrics = trained_service.metrics
    m1, m2, m3 = st.columns(3)

    m1.metric("MAE", f"{metrics['MAE']:.4f}")
    m2.metric("RMSE", f"{metrics['RMSE']:.4f}")
    m3.metric("R² Score", f"{metrics['R2']:.4f}")

    st.caption(
        "MAE và RMSE càng thấp càng tốt. "
        "R² càng gần 1 thì mô hình càng phù hợp."
    )


def render_visualizations(trained_service: PredictionService, target_col: str):
    """Trực quan hóa kết quả biểu đồ đường và biểu đồ phân tán (Scatter)."""
    st.subheader("3. Trực quan hóa kết quả")
    result = trained_service.result_df
    actual_col = "Thực tế (Actual)"
    predicted_col = "Dự đoán (Predicted)"

    tab1, tab2 = st.tabs([
        "📈 Biểu đồ đường",
        "📊 Scatter Actual vs Predicted"
    ])

    # Tab 1: Biểu đồ đường
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            range(len(result)),
            result[actual_col],
            label="Actual",
            marker="o"
        )
        ax.plot(
            range(len(result)),
            result[predicted_col],
            label="Predicted",
            marker="x",
            linestyle="--"
        )
        ax.set_xlabel("Test samples")
        ax.set_ylabel(target_col)
        ax.set_title(f"Actual vs Predicted - {target_col}")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Tab 2: Biểu đồ phân tán (Scatter)
    with tab2:
        fig, ax = plt.subplots(figsize=(8, 6))
        actual = result[actual_col]
        predicted = result[predicted_col]

        ax.scatter(
            actual,
            predicted,
            alpha=0.7,
            label="Predicted samples"
        )

        min_value = min(actual.min(), predicted.min())
        max_value = max(actual.max(), predicted.max())

        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            "r--",
            label="Perfect prediction"
        )
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted Scatter")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)


def render_prediction_table(trained_service: PredictionService):
    """Hiển thị bảng đối chiếu Actual vs Predicted và cung cấp nút tải CSV."""
    st.subheader("4. Bảng đối chiếu Actual vs Predicted")
    result = trained_service.result_df

    st.dataframe(result, use_container_width=True)

    st.download_button(
        "📥 Tải bảng dự đoán CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="prediction_results.csv",
        mime="text/csv"
    )


def render_new_sample_prediction(trained_service: PredictionService, df: pd.DataFrame):
    """Xây dựng giao diện nhập liệu để dự đoán mẫu dữ liệu mới."""
    st.divider()
    st.subheader("5. Dự đoán dữ liệu mới")
    st.write("Nhập các giá trị Features để mô hình dự đoán Target.")

    input_values = {}
    num_features = len(trained_service.feature_cols)
    cols_ui = st.columns(min(3, num_features))

    for idx, feat in enumerate(trained_service.feature_cols):
        with cols_ui[idx % len(cols_ui)]:
            feat_type = trained_service._feature_types.get(feat, "numeric") if hasattr(trained_service, "_feature_types") else "numeric"
            if feat_type == "numeric":
                numeric_data = pd.to_numeric(df[feat], errors="coerce")
                median_val = (
                    float(numeric_data.median())
                    if numeric_data.notna().any()
                    else 0.0
                )
                input_values[feat] = st.number_input(
                    f"Nhập {feat}",
                    value=median_val,
                    key=f"numeric_{feat}"
                )
            else:
                categories = (
                    df[feat]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                if categories:
                    input_values[feat] = st.selectbox(
                        f"Chọn {feat}",
                        categories,
                        key=f"category_{feat}"
                    )
                else:
                    input_values[feat] = st.text_input(
                        f"Nhập {feat}",
                        key=f"text_{feat}"
                    )

    if st.button("✨ Dự đoán dữ liệu mới", type="primary"):
        try:
            predicted_value = trained_service.predict_new_sample(input_values)
            st.success(
                f"Kết quả dự đoán {trained_service.target_col}: {predicted_value:,.4f}"
            )
            st.session_state["new_prediction"] = {
                "inputs": input_values,
                "target": trained_service.target_col,
                "predicted": predicted_value
            }
        except Exception as err:
            st.error(f"Lỗi dự đoán dữ liệu mới: {err}")


def main():
    """Hàm chính điều phối toàn bộ logic trang Prediction."""
    configure_page()

    df = load_dataset()
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.warning(
            "Chưa có dữ liệu. "
            "Vui lòng tải dữ liệu ở trang Quản lý dữ liệu."
        )
        st.stop()

    try:
        # Khởi tạo mới service từ df hiện tại
        service = PredictionService(df)
    except Exception as err:
        st.error(f"Lỗi khởi tạo dữ liệu: {err}")
        st.stop()

    target_col, model_type, feature_cols, test_percent = render_model_parameters(service)
    
    handle_training(service, feature_cols, target_col, test_percent, model_type)

    trained_service = st.session_state.get("trained_prediction_service")
    if trained_service is not None and getattr(trained_service, "is_trained", False):
        render_evaluation_metrics(trained_service)
        render_visualizations(trained_service, target_col)
        render_prediction_table(trained_service)
        render_new_sample_prediction(trained_service, df)


if __name__ == "__main__":
    main()