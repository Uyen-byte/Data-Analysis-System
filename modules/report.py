
"""
Module: modules/report.py
Mô tả:
- Tổng hợp thống kê dữ liệu.
- Ưu tiên dữ liệu đã Cleaning.
- Xuất báo cáo Excel gồm 5 sheet (Data, Cleaned Data, Statistics, Prediction, Model Metrics).
- Định dạng tự động worksheet (freeze panes, auto filter, độ rộng cột).
- Lưu kết quả Prediction và Model Metrics.
"""

from io import BytesIO
from datetime import datetime
from typing import Optional

import pandas as pd


class ReportService:
    """Lớp chịu trách nhiệm tạo báo cáo thống kê và xuất Excel."""

    def __init__(
        self,
        df: pd.DataFrame,
        prediction_df: Optional[pd.DataFrame] = None,
        cleaned_df: Optional[pd.DataFrame] = None,
        model_metrics: Optional[dict] = None,
        model_name: str = "Chưa xác định"
    ):
        """Khởi tạo đối tượng báo cáo và thiết lập nguồn dữ liệu ưu tiên."""
        # Dữ liệu gốc
        self.df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

        # Dữ liệu đã Cleaning
        self.cleaned_df = (
            cleaned_df.copy()
            if isinstance(cleaned_df, pd.DataFrame)
            else None
        )

        # Ưu tiên dữ liệu Cleaning khi có
        if self.cleaned_df is not None and not self.cleaned_df.empty:
            self.analysis_df = self.cleaned_df
        else:
            self.analysis_df = self.df

        # Kết quả Prediction
        self.prediction_df = (
            prediction_df.copy()
            if isinstance(prediction_df, pd.DataFrame)
            else pd.DataFrame()
        )

        # Model Metrics
        self.model_metrics = (
            model_metrics.copy()
            if isinstance(model_metrics, dict)
            else {}
        )

        self.model_name = model_name or "Chưa xác định"

    # =====================================================
    # TỔNG QUAN DỮ LIỆU
    # =====================================================

    def get_summary_info(self) -> dict:
        """Trả về thông tin tổng quan dataset (số dòng, số cột, số ô thiếu)."""
        if self.analysis_df.empty:
            return {
                "rows": 0,
                "cols": 0,
                "missing_cells": 0
            }

        return {
            "rows": int(self.analysis_df.shape[0]),
            "cols": int(self.analysis_df.shape[1]),
            "missing_cells": int(self.analysis_df.isna().sum().sum())
        }

    # =====================================================
    # THỐNG KÊ MÔ TẢ
    # =====================================================

    def generate_statistics(self) -> pd.DataFrame:
        """Tạo bảng thống kê mô tả cho toàn bộ các cột trong dữ liệu phân tích."""
        if self.analysis_df.empty:
            return pd.DataFrame()

        stats_df = self.analysis_df.describe(include="all").transpose()
        stats_df.index.name = "Feature"

        return stats_df

    # =====================================================
    # MODEL METRICS
    # =====================================================

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Chuyển đổi dictionary Model Metrics thành DataFrame."""
        if not self.model_metrics:
            return pd.DataFrame({"Thông báo": ["Chưa có thông tin đánh giá mô hình."]})

        rows = [
            {
                "Metric": "Model",
                "Value": self.model_name
            }
        ]

        for key, value in self.model_metrics.items():
            rows.append({
                "Metric": str(key),
                "Value": round(value, 4) if isinstance(value, float) else value
            })

        return pd.DataFrame(rows)

    # =====================================================
    # XUẤT PREDICTION CSV
    # =====================================================

    def export_prediction_csv(self) -> bytes:
        """Xuất kết quả Prediction thành dữ liệu CSV chuẩn UTF-8 BOM."""
        return self.prediction_df.to_csv(index=False).encode("utf-8-sig")

    # =====================================================
    # XUẤT EXCEL
    # =====================================================

    def export_to_excel(self) -> bytes:
        """
        Xuất báo cáo Excel gồm 5 sheet:
        1. Data
        2. Cleaned Data
        3. Statistics
        4. Prediction
        5. Model Metrics
        """
        if self.df.empty and (self.cleaned_df is None or self.cleaned_df.empty):
            raise ValueError("Không có dữ liệu để xuất báo cáo.")

        buffer = BytesIO()

        # Chuẩn bị dữ liệu cho các sheet (kèm thông báo nếu chưa có dữ liệu)
        if self.cleaned_df is not None and not self.cleaned_df.empty:
            cleaned_data = self.cleaned_df
        else:
            cleaned_data = pd.DataFrame({"Thông báo": ["Chưa có dữ liệu Cleaning."]})

        stats_df = self.generate_statistics()
        if stats_df.empty:
            stats_df = pd.DataFrame({"Thông báo": ["Không có dữ liệu thống kê."]})

        if self.prediction_df.empty:
            prediction_data = pd.DataFrame({"Thông báo": ["Chưa có kết quả Prediction."]})
            include_pred_index = False
        else:
            prediction_data = self.prediction_df
            include_pred_index = True

        metrics_df = self.get_metrics_dataframe()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            # Sheet 1: Data
            self.df.to_excel(writer, sheet_name="Data", index=False)

            # Sheet 2: Cleaned Data
            cleaned_data.to_excel(writer, sheet_name="Cleaned Data", index=False)

            # Sheet 3: Statistics
            stats_df.to_excel(writer, sheet_name="Statistics", index=True)

            # Sheet 4: Prediction
            prediction_data.to_excel(writer, sheet_name="Prediction", index=include_pred_index)

            # Sheet 5: Model Metrics
            metrics_df.to_excel(writer, sheet_name="Model Metrics", index=False)

            # Định dạng tất cả worksheet
            self._format_worksheets(writer.book.worksheets)

        buffer.seek(0)
        return buffer.getvalue()

    # =====================================================
    # ĐỊNH DẠNG EXCEL
    # =====================================================

    def _format_worksheets(self, worksheets) -> None:
        """Định dạng cố định hàng đầu (freeze panes), auto filter và độ rộng cột."""
        for sheet in worksheets:
            # Cố định hàng đầu
            sheet.freeze_panes = "A2"

            # Auto filter nếu có dữ liệu
            if sheet.max_row > 0 and sheet.max_column > 0:
                sheet.auto_filter.ref = sheet.dimensions

            # Tự động tính toán và căn chỉnh độ rộng cột
            for column_cells in sheet.columns:
                max_len = 0
                for cell in column_cells:
                    if cell.value is not None:
                        cell_len = len(str(cell.value))
                        max_len = max(max_len, cell_len)

                col_letter = column_cells[0].column_letter
                sheet.column_dimensions[col_letter].width = min(
                    max(max_len + 3, 12),
                    35
                )

    # =====================================================
    # TẠO TÊN FILE
    # =====================================================

    @staticmethod
    def create_filename(prefix: str = "BaoCao_PhanTich") -> str:
        """Tạo tên file Excel kèm timestamp hiện tại."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.xlsx"