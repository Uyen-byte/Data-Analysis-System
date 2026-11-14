
import pandas as pd

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

    def get_descriptive_statistics(self):
        numeric_df = self.df.select_dtypes(include="number")
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
        numeric_df = self.df.select_dtypes(include="number")
        return numeric_df.corr()

    def analyze_column(self, column):
        if column not in self.df.columns:
            raise ValueError(f"Không tìm thấy cột: {column}")

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError("Vui lòng chọn cột dữ liệu dạng số.")

        return {
            "Sum": self.get_sum(column),
            "Mean": self.get_mean(column),
            "Median": self.get_median(column),
            "Max": self.get_max(column),
            "Min": self.get_min(column),
            "Std": self.get_std(column)
        }
