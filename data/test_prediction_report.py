"""Tests for the prediction and report features."""

import unittest
from io import BytesIO

import pandas as pd
from openpyxl import load_workbook

from modules.prediction import PredictionService
from modules.report import ReportService


class PredictionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "student_id": range(100, 112),
                "study_hours": range(1, 13),
                "group": ["A", "B"] * 6,
                "score": [value * 2 + 5 for value in range(1, 13)],
            }
        )
        self.service = PredictionService(self.df)

    def test_feature_candidates_are_numeric_and_exclude_identifiers(self) -> None:
        self.assertEqual(
            self.service.get_feature_candidates("score"),
            ["study_hours"],
        )

    def test_training_and_prediction_with_numeric_features(self) -> None:
        metrics = self.service.train_and_evaluate(
            feature_cols=["study_hours"],
            target_col="score",
        )

        self.assertTrue(self.service.is_trained)
        self.assertEqual(set(metrics), {"MAE", "RMSE", "R2"})
        self.assertIsInstance(
            self.service.predict_new_sample({"study_hours": 13}),
            float,
        )


class ReportServiceTests(unittest.TestCase):
    def test_excel_report_contains_all_sheets(self) -> None:
        df = pd.DataFrame({"hours": [1, 2], "score": [3, 5]})
        report = ReportService(
            df,
            cleaned_df=df,
            prediction_df=pd.DataFrame({"prediction": [7]}),
            model_metrics={"MAE": 0.1},
        )

        workbook = load_workbook(
            BytesIO(report.export_to_excel()),
            read_only=True,
        )
        self.assertEqual(
            workbook.sheetnames,
            ["Data", "Cleaned Data", "Statistics", "Prediction", "Model Metrics"],
        )
        workbook.close()


if __name__ == "__main__":
    unittest.main()
