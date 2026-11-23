
import pandas as pd


IDENTIFIER_KEYWORDS = {"id", "code", "index", "key", "stt"}


def is_identifier_column(column):
    """Return whether a column name indicates a row identifier."""
    parts = (
        str(column)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .split("_")
    )
    return any(part in IDENTIFIER_KEYWORDS for part in parts)


class DataAnalyzer:
    """Xử lý thống kê và phân tích dữ liệu."""

    def __init__(self, df):
        self.df = df.copy()


    def get_overview(self):
        return {
            "total_rows": self.df.shape[0],
            "total_columns": self.df.shape[1],
            "missing_values": int(self.df.isnull().sum().sum()),
            "duplicate_rows": int(self.df.duplicated().sum())
        }

    def get_numeric_data(self):
        """Return usable numeric columns, including numeric-like database values."""
        numeric_data = {}
        for column in self.df.columns:
            values = self.df[column]
            if (
                is_identifier_column(column)
                or pd.api.types.is_bool_dtype(values)
                or not values.notna().any()
            ):
                continue

            converted = pd.to_numeric(values, errors="coerce")
            if pd.api.types.is_numeric_dtype(values) or (
                converted.notna().sum() == values.notna().sum()
            ):
                numeric_data[column] = converted

        return pd.DataFrame(numeric_data, index=self.df.index)

    def get_descriptive_statistics(self):
        numeric_df = self.get_numeric_data()
        return numeric_df.describe().T


    def get_sum(self, column):
        return self.df[column].sum()

    def get_mean(self, column):
        return self.df[column].mean()

    def get_median(self, column):
        return self.df[column].median()

    def get_max(self, column):
        return self.df[column].max()

    def get_min(self, column):
        return self.df[column].min()

    def get_std(self, column):
        return self.df[column].std()


    def get_correlation(self):
        numeric_df = self.get_numeric_data()
        return numeric_df.corr()

    def analyze_column(self, column):
        numeric_data = self.get_numeric_data()
        if column not in self.df.columns:
            raise ValueError(f"Không tìm thấy cột: {column}")
        if column not in numeric_data.columns:
            raise ValueError("Vui lòng chọn cột dữ liệu dạng số.")

        return {
            "Sum": numeric_data[column].sum(),
            "Mean": numeric_data[column].mean(),
            "Median": numeric_data[column].median(),
            "Max": numeric_data[column].max(),
            "Min": numeric_data[column].min(),
            "Std": numeric_data[column].std(),
        }
