import pandas as pd
import numpy as np

class DataCleaning:

    def __init__(self, df):

        # Kiểm tra dữ liệu đầu vào
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Dữ liệu đầu vào phải là DataFrame."
            )

        # Lưu dữ liệu ban đầu
        self.original_df = df.copy(deep=True)

        # Tạo dữ liệu phục vụ xử lý
        self.df = df.copy(deep=True)

        # Lưu lịch sử làm sạch
        self.cleaning_history = []

    def _validate_columns(self, columns):

        if isinstance(columns, str):
            columns = [columns]

        invalid_columns = [
            col for col in columns
             if col not in self.df.columns
        ]

        if invalid_columns:
             raise KeyError(
                f"Không tìm thấy cột: {invalid_columns}"
            )

        return columns
    def check_missing(self):

        # Đếm số giá trị bị thiếu
        missing_count = self.df.isnull().sum()

        # Tính tỷ lệ phần trăm
        if len(self.df) > 0:
            missing_percent = (
                self.df.isnull().mean() * 100
            ).round(2)
        else:
            missing_percent = pd.Series(
                0, index=self.df.columns
            )

        # Tạo bảng thống kê
        result = pd.DataFrame({
            "Column": self.df.columns,
            "Missing Count": missing_count.values,
            "Missing Percent": missing_percent.values
        })

        return result
    def check_duplicates(self, subset=None):

        # Kiểm tra tên cột nếu người dùng lựa chọn
        if subset is not None:
            subset = self._validate_columns(subset)

        # Xác định các dòng dữ liệu trùng lặp
        duplicate_mask = self.df.duplicated(
            subset=subset,
            keep=False
        )

        # Trả về các dòng bị trùng lặp
        result = self.df.loc[duplicate_mask].copy()

        return result
    def remove_duplicates(self, subset=None):

        # Kiểm tra tên cột nếu người dùng lựa chọn
        if subset is not None:
            subset = self._validate_columns(subset)

        # Đếm số dòng dữ liệu ban đầu
        before = len(self.df)

        # Xóa các dòng dữ liệu trùng lặp
        self.df = self.df.drop_duplicates(
            subset=subset,
            keep="first"
        ).copy()

        # Tính số dòng đã bị xóa
        removed = before - len(self.df)

        # Lưu lịch sử làm sạch dữ liệu
        self.cleaning_history.append(
            f"Đã xóa {removed} dòng dữ liệu trùng lặp."
        )

        return removed
    
    def fill_missing(self, strategy="mean",
                     columns=None, value=None):

        # Các phương pháp xử lý được hỗ trợ
        strategies = ["mean", "median", "mode", "value"]

        if strategy not in strategies:
            raise ValueError("Phương pháp xử lý không hợp lệ.")

        # Kiểm tra giá trị nhập vào
        if strategy == "value" and value is None:
            raise ValueError("Bạn cần nhập giá trị thay thế.")

        # Xác định các cột cần xử lý
        if columns is None:
            if strategy in ["mean", "median"]:
                columns = [
                    col for col in self.df.columns
                    if pd.api.types.is_numeric_dtype(self.df[col])
                ]
            else:
                columns = list(self.df.columns)
        else:
            columns = self._validate_columns(columns)

        filled_counts = {}

        # Duyệt qua từng cột
        for col in columns:

            missing_before = int(self.df[col].isnull().sum())

            if missing_before == 0:
                continue

            # Thay thế bằng giá trị trung bình
            if strategy == "mean":

                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    raise TypeError(f"Cột {col} không phải dữ liệu số.")

                fill_value = self.df[col].mean()

            # Thay thế bằng giá trị trung vị
            elif strategy == "median":

                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    raise TypeError(f"Cột {col} không phải dữ liệu số.")

                fill_value = self.df[col].median()

            # Thay thế bằng giá trị phổ biến nhất
            elif strategy == "mode":

                modes = self.df[col].mode(dropna=True)

                if modes.empty:
                    continue

                fill_value = modes.iloc[0]

            # Thay thế bằng giá trị người dùng nhập
            else:
                fill_value = value

            # Kiểm tra cột có dữ liệu hợp lệ để tính toán
            if pd.isna(fill_value):
                raise ValueError(
                    f"Cột {col} không có giá trị hợp lệ để thay thế."
                )

            # Thực hiện thay thế
            self.df[col] = self.df[col].fillna(fill_value)

            # Thống kê số lượng đã xử lý
            filled_counts[col] = (
                missing_before - int(self.df[col].isnull().sum())
            )

        # Lưu lịch sử xử lý
        self.cleaning_history.append(
            f"Xử lý dữ liệu thiếu bằng phương pháp {strategy}."
        )

        return filled_counts
    
    def remove_missing(self, how="any", subset=None):

        # Kiểm tra phương pháp xóa
        if how not in ["any", "all"]:
            raise ValueError("how chỉ nhận 'any' hoặc 'all'.")

        # Kiểm tra tên cột
        if subset is not None:
            subset = self._validate_columns(subset)

        # Đếm số dòng ban đầu
        before = len(self.df)

        # Xóa các dòng chứa dữ liệu thiếu
        self.df = self.df.dropna(
            how=how,
            subset=subset
        ).copy()

        # Tính số dòng đã xóa
        removed = before - len(self.df)

        # Lưu lịch sử xử lý
        self.cleaning_history.append(
            f"Đã xóa {removed} dòng có dữ liệu thiếu."
        )

        return removed

    def get_data_types(self):

        # Lấy kiểu dữ liệu của từng cột
        result = pd.DataFrame({
            "Column": self.df.columns,
            "Data Type": [
                str(dtype) for dtype in self.df.dtypes
            ],
            "Missing Count": self.df.isnull().sum().values
        })

        return result
    
    def convert_data_types(self, column, target_type):

        # Kiểm tra tên cột
        columns = self._validate_columns(column)

        if len(columns) != 1:
            raise ValueError("Chỉ được chọn một cột để chuyển đổi.")

        column = columns[0]

        # Chuẩn hóa tên kiểu dữ liệu
        target_type = target_type.lower().strip()

        # Lưu dữ liệu ban đầu của cột
        original = self.df[column]

        # Chuyển đổi sang kiểu số nguyên
        if target_type in ["int", "integer"]:

            numeric = pd.to_numeric(
                original,
                errors="coerce"
            )

            # Chỉ chấp nhận những giá trị số nguyên hợp lệ
            valid_integer = (
                numeric.notna()
                & np.isfinite(numeric)
                & (numeric % 1 == 0)
            )

            converted = numeric.where(
                valid_integer
            ).astype("Int64")

        # Chuyển đổi sang kiểu số thực
        elif target_type == "float":

            converted = pd.to_numeric(
                original,
                errors="coerce"
            ).astype("float64")

        # Chuyển đổi sang kiểu chuỗi
        elif target_type in ["str", "string"]:

            converted = original.astype("string")

        # Chuyển đổi sang kiểu ngày tháng
        elif target_type in ["datetime", "date"]:

            converted = pd.to_datetime(
                original,
                errors="coerce"
            )

        else:
            raise ValueError(
                "Kiểu dữ liệu không được hỗ trợ."
            )

        # Xác định các giá trị không chuyển đổi được
        failed = original.notna() & converted.isna()

        failed_count = int(failed.sum())

        failed_values = original.loc[failed].astype(str).tolist()

        # Cập nhật dữ liệu sau chuyển đổi
        self.df[column] = converted

        # Lưu lịch sử xử lý
        self.cleaning_history.append(
            f"Chuyển cột {column} sang kiểu {target_type}."
        )

        # Trả về kết quả chuyển đổi
        return {
            "column": column,
            "target_type": target_type,
            "failed_count": failed_count,
            "failed_values": failed_values
        }
    
    def standardize_data(self, columns=None, case=None):

        # Các chế độ chuẩn hóa được hỗ trợ
        valid_cases = [None, "lower", "upper", "title"]

        if case not in valid_cases:
            raise ValueError(
                "Chế độ chuẩn hóa không hợp lệ."
            )

        # Xác định các cột cần chuẩn hóa
        if columns is None:
            columns = [
                col for col in self.df.columns
                if self.df[col].dropna().map(
                    lambda x: isinstance(x, str)
                ).any()
            ]
        else:
            columns = self._validate_columns(columns)

        standardized_columns = []
        skipped_columns = []

        # Duyệt qua từng cột
        for col in columns:

            # Kiểm tra cột có chứa dữ liệu văn bản không
            has_text = self.df[col].dropna().map(
                lambda x: isinstance(x, str)
            ).any()

            if not has_text:
                skipped_columns.append(col)
                continue

            # Chuẩn hóa từng giá trị văn bản
            def normalize_text(value):

                if isinstance(value, str):

                    # Loại bỏ khoảng trắng thừa
                    value = " ".join(value.split())

                    # Chuyển thành chữ thường
                    if case == "lower":
                        value = value.lower()

                    # Chuyển thành chữ hoa
                    elif case == "upper":
                        value = value.upper()

                    # Viết hoa chữ cái đầu mỗi từ
                    elif case == "title":
                        value = value.title()

                return value

            # Cập nhật dữ liệu sau chuẩn hóa
            self.df[col] = self.df[col].map(
                normalize_text
            )

            standardized_columns.append(col)

        # Lưu lịch sử xử lý
        self.cleaning_history.append(
            "Đã chuẩn hóa dữ liệu văn bản."
        )

        # Trả về kết quả
        return {
            "standardized": standardized_columns,
            "skipped": skipped_columns
        }
    
    @staticmethod
    def _calculate_iqr_bounds(series, factor=1.5):

        # Chuyển dữ liệu về dạng số
        numeric = pd.to_numeric(
            series,
            errors="coerce"
        )

        # Chỉ lấy những giá trị số hữu hạn
        finite_mask = (
            numeric.notna()
            & np.isfinite(numeric.fillna(0))
        )

        values = numeric.loc[finite_mask]

        # Kiểm tra dữ liệu có giá trị hợp lệ không
        if values.empty:
            return None, None

        # Tính tứ phân vị thứ nhất và thứ ba
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        # Tính khoảng tứ phân vị
        iqr = q3 - q1

        # Xác định giới hạn dưới và giới hạn trên
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        return lower, upper
    
    def detect_outliers(self, columns=None, factor=1.5):

        # Kiểm tra hệ số IQR
        if (
            not isinstance(factor, (int, float, np.number))
            or not np.isfinite(factor)
            or factor <= 0
        ):
            raise ValueError("factor phải là số hữu hạn lớn hơn 0.")

        # Xác định các cột cần kiểm tra
        if columns is None:

            columns = [
                col for col in self.df.columns
                if (
                    pd.api.types.is_numeric_dtype(self.df[col])
                    and not pd.api.types.is_bool_dtype(self.df[col])
                )
            ]

        else:
            columns = self._validate_columns(columns)

        outliers = []

        # Duyệt qua từng cột
        for col in columns:

            # Kiểm tra kiểu dữ liệu
            if (
                not pd.api.types.is_numeric_dtype(self.df[col])
                or pd.api.types.is_bool_dtype(self.df[col])
            ):
                raise TypeError(
                    f"Cột {col} không phải dữ liệu số."
                )

            # Tính giới hạn ngoại lệ
            lower, upper = self._calculate_iqr_bounds(
                self.df[col], factor
            )

            if lower is None:
                continue

            # Xác định các giá trị ngoại lệ
            numeric = pd.to_numeric(
                self.df[col],
                errors="coerce"
            )

            mask = (
                numeric.notna()
                & (
                    (numeric < lower)
                    | (numeric > upper)
                )
            ).fillna(False)

            # Lưu thông tin các giá trị ngoại lệ
            for index, value in self.df.loc[mask, col].items():

                outliers.append({
                    "Index": index,
                    "Column": col,
                    "Value": value,
                    "Lower Bound": lower,
                    "Upper Bound": upper
                })

        # Trả về bảng thống kê ngoại lệ
        result = pd.DataFrame(
            outliers,
            columns=[
                "Index",
                "Column",
                "Value",
                "Lower Bound",
                "Upper Bound"
            ]
        )

        return result
    
    def handle_outliers(self, columns=None,
                        factor=1.5, action="clip"):

        # Kiểm tra phương pháp xử lý
        if action not in ["clip", "remove"]:
            raise ValueError(
                "action chỉ nhận 'clip' hoặc 'remove'."
            )

        # Kiểm tra hệ số IQR
        if (
            not isinstance(factor, (int, float, np.number))
            or not np.isfinite(factor)
            or factor <= 0
        ):
            raise ValueError("factor phải là số hữu hạn lớn hơn 0.")

        # Xác định các cột cần xử lý
        if columns is None:

            columns = [
                col for col in self.df.columns
                if (
                    pd.api.types.is_numeric_dtype(self.df[col])
                    and not pd.api.types.is_bool_dtype(self.df[col])
                )
            ]

        else:
            columns = self._validate_columns(columns)

        # Tạo mặt nạ đánh dấu dòng có ngoại lệ
        row_outlier_mask = np.zeros(
            len(self.df),
            dtype=bool
        )

        affected_values = 0

        # Duyệt qua từng cột
        for col in columns:

            if (
                not pd.api.types.is_numeric_dtype(self.df[col])
                or pd.api.types.is_bool_dtype(self.df[col])
            ):
                raise TypeError(
                    f"Cột {col} không phải dữ liệu số."
                )

            # Tính giới hạn ngoại lệ
            lower, upper = self._calculate_iqr_bounds(
                self.df[col], factor
            )

            if lower is None:
                continue

            numeric = pd.to_numeric(
                self.df[col],
                errors="coerce"
            )

            # Xác định vị trí ngoại lệ
            mask = (
                numeric.notna()
                & (
                    (numeric < lower)
                    | (numeric > upper)
                )
            ).fillna(False)

            # Đếm số giá trị ngoại lệ
            affected_values += int(mask.sum())

            # Phương pháp điều chỉnh giá trị
            if action == "clip":

                self.df.loc[mask, col] = numeric.loc[mask].clip(
                    lower=lower,
                    upper=upper
                ).to_numpy()

            # Phương pháp xóa dòng chứa ngoại lệ
            else:

                row_outlier_mask |= mask.to_numpy(dtype=bool)

        removed_rows = 0

        if action == "remove":

            removed_rows = int(row_outlier_mask.sum())

            self.df = self.df.loc[
                ~row_outlier_mask
            ].copy()

        # Lưu lịch sử xử lý
        self.cleaning_history.append(
            f"Xử lý {affected_values} giá trị ngoại lệ "
            f"bằng phương pháp {action}."
        )

        # Trả về kết quả xử lý
        return {
            "action": action,
            "affected_values": affected_values,
            "removed_rows": removed_rows
        }
    def get_cleaned_data(self):

        # Trả về bản sao dữ liệu đã làm sạch
        return self.df.copy(deep=True)
    def get_cleaning_summary(self):

        # Tổng hợp thông tin dữ liệu
        summary = {
            "original_rows": len(self.original_df),

            "current_rows": len(self.df),

            "columns": len(self.df.columns),

            "missing_values": int(
                self.df.isnull().sum().sum()
            ),

            "duplicate_rows": int(
                self.df.duplicated().sum()
            ),

            "cleaning_history": self.cleaning_history.copy()
        }

        return summary
    def reset_data(self):

        # Khôi phục dữ liệu ban đầu
        self.df = self.original_df.copy(deep=True)

        # Lưu lịch sử khôi phục
        self.cleaning_history.append(
            "Đã khôi phục dữ liệu ban đầu."
        )