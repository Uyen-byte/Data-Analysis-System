"""
Module: modules/prediction.py
Mô tả: Xử lý nghiệp vụ huấn luyện mô hình hồi quy tuyến tính (Linear Regression),
      đánh giá sai số và dự đoán dữ liệu mới theo hướng đối tượng (OOP).
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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

        self.metrics: dict[str, float | None] = {
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

    def _is_numeric_column(self, column: str) -> bool:
        """Return whether a column contains numeric values, including Decimals."""
        values = self.raw_df[column]
        if pd.api.types.is_bool_dtype(values):
            return False
        if pd.api.types.is_numeric_dtype(values):
            return True

        converted = pd.to_numeric(values, errors="coerce")
        return bool(
            values.notna().any()
            and converted.notna().sum() == values.notna().sum()
        )

    def get_numeric_columns(self) -> list:
        """Return numeric and numeric-like columns, including database decimals."""
        numeric_columns = []
        for column in self.raw_df.columns:
            if self._is_numeric_column(column):
                numeric_columns.append(column)

        return numeric_columns

    def get_categorical_columns(self) -> list:
        """Return non-numeric columns that contain at least one usable value."""
        return [
            column
            for column in self.raw_df.columns
            if self.raw_df[column].notna().any()
            and not self._is_numeric_column(column)
        ]

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
        Trả về các cột số và phân loại hợp lệ làm Features.
        Loại bỏ Target và các cột định danh.
        """

        return [
            col
            for col in self.raw_df.columns
            if col != target_col
            and not self._is_identifier_column(col)
            and self.raw_df[col].notna().any()
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

        if not self._is_numeric_column(target_col):
            raise TypeError(f"Cột Target '{target_col}' phải chứa dữ liệu số.")

        self._feature_types = {}
        for col in feature_cols:
            if self._is_numeric_column(col):
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
                self._feature_types[col] = "numeric"
            else:
                clean_df[col] = clean_df[col].map(
                    lambda value: str(value) if pd.notna(value) else np.nan
                )
                self._feature_types[col] = "categorical"

        clean_df[target_col] = pd.to_numeric(
            clean_df[target_col],
            errors="coerce",
        )

        # Loại bỏ dòng chứa giá trị NaN
        clean_df.dropna(inplace=True)

        if len(clean_df) < 3:
            raise ValueError(
                "Cần tối thiểu 3 dòng dữ liệu số hợp lệ sau khi bỏ "
                f"giá trị trống; hiện chỉ còn {len(clean_df)} dòng."
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

        if not 0 < test_size < 1:
            raise ValueError("Tỷ lệ dữ liệu kiểm thử phải nằm giữa 0 và 1.")

        self.feature_cols = feature_cols.copy()
        self.target_col = target_col

        X = clean_df[self.feature_cols]
        y = clean_df[self.target_col]

        # Chia dữ liệu Train và Test
        minimum_test_rows = 2 if len(clean_df) >= 4 else 1
        test_rows = min(
            max(minimum_test_rows, int(np.ceil(len(clean_df) * test_size))),
            len(clean_df) - 2,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_rows,
            random_state=random_state
        )

        estimator = (
            Ridge(alpha=1.0)
            if model_type == "Ridge Regression"
            else LinearRegression()
        )

        categorical_features = [
            column for column in self.feature_cols
            if self._feature_types[column] == "categorical"
        ]
        numeric_features = [
            column for column in self.feature_cols
            if self._feature_types[column] == "numeric"
        ]
        if categorical_features:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric", "passthrough", numeric_features),
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore"),
                        categorical_features,
                    ),
                ],
                remainder="drop",
            )
            self.model = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("regressor", estimator),
                ]
            )
        else:
            self.model = estimator

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

        r2 = float(r2_score(y_actual_array, y_pred_array)) if len(y_test) >= 2 else None

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

            if self._feature_types[col] == "numeric":
                try:
                    row_values.append(float(input_dict[col]))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Giá trị của Feature '{col}' phải là số."
                    ) from error
            else:
                row_values.append(str(input_dict[col]))

        # Tạo DataFrame theo đúng thứ tự Features
        input_df = pd.DataFrame(
            [row_values],
            columns=self.feature_cols
        )

        # Dự đoán
        prediction = self.model.predict(input_df)[0]

        return float(prediction)