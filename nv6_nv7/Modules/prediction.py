"""
Module: modules/prediction.py
Mô tả: Xử lý nghiệp vụ huấn luyện mô hình hồi quy tuyến tính (Linear Regression),
      đánh giá sai số và dự đoán dữ liệu mới theo hướng đối tượng (OOP).
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score


class PredictionService:
    """Lớp quản lý quy trình tiền xử lý, huấn luyện và dự đoán dữ liệu."""

    def __init__(self, df: pd.DataFrame):
        """Khởi tạo đối tượng với DataFrame đầu vào."""

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(
                "Dữ liệu đầu vào (DataFrame) không hợp lệ hoặc đang trống."
            )

        self.raw_df = df.copy()

        self.model = LinearRegression()

        self.feature_cols = []
        self.target_col = ""
        self.is_trained = False

        # Lưu trữ kết quả sau huấn luyện
        self.result_df = pd.DataFrame()

        self.metrics = {
            "MAE": 0.0,
            "RMSE": 0.0,
            "R2": 0.0
        }

        self._feature_types = {}

    # =====================================================
    # 1. KIỂM TRA CỘT ĐỊNH DANH
    # =====================================================

    def _is_identifier_column(self, col: str) -> bool:
        """
        Kiểm tra cột có phải cột định danh hay không.
        Ví dụ: student_id, customer_id, product_code.
        """

        col_lower = str(col).strip().lower()

        excluded_keywords = [
            "id",
            "code",
            "index",
            "key",
            "stt"
        ]

        # Tách tên cột theo dấu gạch dưới, gạch ngang và khoảng trắng
        column_parts = (
            col_lower
            .replace("-", "_")
            .replace(" ", "_")
            .split("_")
        )

        return any(
            keyword in column_parts
            for keyword in excluded_keywords
        )

    # =====================================================
    # 2. LẤY DANH SÁCH CỘT SỐ
    # =====================================================

    def get_numeric_columns(self) -> list:
        """Trả về danh sách tên các cột có kiểu dữ liệu số."""

        return self.raw_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

    # =====================================================
    # 3. LẤY DANH SÁCH TARGET HỢP LỆ
    # =====================================================

    def get_target_candidates(self) -> list:
        """Trả về danh sách cột số hợp lệ làm Target, loại bỏ student_id và các cột định danh."""
        numeric_cols = self.get_numeric_columns()

        return [
            col for col in numeric_cols
            if not self._is_identifier_column(col)
        ]

    # =====================================================
    # 4. LẤY DANH SÁCH FEATURES
    # =====================================================

    def get_feature_candidates(self, target_col: str) -> list:
        """
        Trả về danh sách Features hợp lệ.
        Loại bỏ Target và các cột định danh.
        """

        return [
            col
            for col in self.raw_df.columns
            if col != target_col
            and not self._is_identifier_column(col)
        ]

    # =====================================================
    # 5. TIỀN XỬ LÝ DỮ LIỆU
    # =====================================================

    def _prepare_data(
        self,
        feature_cols: list,
        target_col: str
    ) -> pd.DataFrame:
        """Kiểm tra và làm sạch dữ liệu trước khi huấn luyện."""

        if not feature_cols:
            raise ValueError(
                "Vui lòng chọn ít nhất một cột đặc trưng (Features)."
            )

        if target_col in feature_cols:
            raise ValueError(
                "Cột mục tiêu (Target) không được trùng với cột đầu vào."
            )

        # Kiểm tra Target có phải cột định danh không
        if self._is_identifier_column(target_col):
            raise ValueError(
                f"Không được sử dụng cột '{target_col}' làm Target "
                "vì đây là cột định danh."
            )

        # Kiểm tra Features có chứa cột định danh không
        invalid_features = [
            col for col in feature_cols
            if self._is_identifier_column(col)
        ]

        if invalid_features:
            raise ValueError(
                "Không được sử dụng các cột định danh làm Features: "
                + ", ".join(invalid_features)
            )

        # Kiểm tra sự tồn tại của các cột
        selected_cols = feature_cols + [target_col]

        for col in selected_cols:
            if col not in self.raw_df.columns:
                raise KeyError(
                    f"Không tìm thấy cột '{col}' trong tập dữ liệu."
                )

        # Lấy dữ liệu được chọn
        clean_df = self.raw_df[selected_cols].copy()

        # Ép kiểu dữ liệu sang dạng số
        for col in selected_cols:
            clean_df[col] = pd.to_numeric(
                clean_df[col],
                errors="coerce"
            )

        # Loại bỏ dòng chứa giá trị NaN
        clean_df.dropna(inplace=True)

        if len(clean_df) < 5:
            raise ValueError(
                "Cần tối thiểu 5 dòng dữ liệu số hợp lệ "
                "để huấn luyện mô hình."
            )

        if clean_df[target_col].nunique() < 2:
            raise ValueError(
                "Cột mục tiêu phải có ít nhất 2 giá trị khác nhau."
            )

        return clean_df

    # =====================================================
    # 6. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH
    # =====================================================

    def train_and_evaluate(
        self,
        feature_cols: list,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        model_type: str = "Linear Regression"
    ) -> dict:
        """
        Huấn luyện mô hình Linear Regression hoặc Ridge Regression
        và tính toán các chỉ số đánh giá.
        """

        clean_df = self._prepare_data(
            feature_cols,
            target_col
        )

        self.feature_cols = feature_cols.copy()
        self.target_col = target_col

        # Lưu kiểu dữ liệu Features
        self._feature_types = {
            col: "numeric"
            for col in self.feature_cols
        }

        X = clean_df[self.feature_cols]
        y = clean_df[self.target_col]

        # Chia dữ liệu Train và Test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

        # Lựa chọn mô hình
        if model_type == "Ridge Regression":
            self.model = Ridge(alpha=1.0)
        else:
            self.model = LinearRegression()

        # Huấn luyện mô hình
        self.model.fit(X_train, y_train)

        # Dự đoán dữ liệu kiểm thử
        y_pred_array = self.model.predict(X_test)

        y_actual_array = y_test.to_numpy()

        # Tính toán sai số
        errors = y_actual_array - y_pred_array

        mae = float(
            np.mean(np.abs(errors))
        )

        rmse = float(
            np.sqrt(np.mean(errors ** 2))
        )

        r2 = (
            float(r2_score(y_actual_array, y_pred_array))
            if len(y_test) >= 2
            else 0.0
        )

        self.metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

        # Tạo bảng đối chiếu kết quả
        self.result_df = pd.DataFrame(
            {
                "Thực tế (Actual)": y_actual_array,
                "Dự đoán (Predicted)": y_pred_array,
                "Sai số (Error)": errors
            },
            index=y_test.index
        ).sort_index()

        self.is_trained = True

        return self.metrics

    # =====================================================
    # 7. DỰ ĐOÁN DỮ LIỆU MỚI
    # =====================================================

    def predict_new_sample(self, input_dict: dict) -> float:
        """
        Dự đoán giá trị đầu ra cho một mẫu dữ liệu mới.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Mô hình chưa được huấn luyện. "
                "Hãy gọi train_and_evaluate() trước."
            )

        # Kiểm tra dữ liệu đầu vào
        row_values = []

        for col in self.feature_cols:

            if col not in input_dict:
                raise KeyError(
                    f"Thiếu giá trị đầu vào cho cột '{col}'."
                )

            row_values.append(
                float(input_dict[col])
            )

        # Tạo DataFrame theo đúng thứ tự Features
        input_df = pd.DataFrame(
            [row_values],
            columns=self.feature_cols
        )

        # Dự đoán
        prediction = self.model.predict(input_df)[0]

        return float(prediction)