"""Tests for the prediction and report features."""

import unittest
from decimal import Decimal
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

    def test_feature_candidates_include_categories_and_exclude_identifiers(self) -> None:
        self.assertEqual(
            self.service.get_feature_candidates("score"),
            ["study_hours", "group"],
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

    def test_training_and_prediction_with_categorical_feature(self) -> None:
        service = PredictionService(
            pd.DataFrame(
                {
                    "age": [14, 18, 17, 16, 15, 19, 16, 17],
                    "gender": [
                        "male", "female", "female", "other",
                        "male", "female", "other", "male",
                    ],
                    "study_hours": [3.1, 3.7, 7.9, 1.1, 2.0, 4.0, 5.2, 6.1],
                }
            )
        )

        self.assertIn("gender", service.get_feature_candidates("age"))
        service.train_and_evaluate(
            feature_cols=["gender", "study_hours"],
            target_col="age",
        )
        prediction = service.predict_new_sample(
            {"gender": "female", "study_hours": 4.5}
        )

        self.assertEqual(len(service.result_df), 2)
        self.assertIsInstance(prediction, float)

    def test_decimal_columns_are_available_for_prediction(self) -> None:
        decimal_df = pd.DataFrame(
            {
                "student_id": range(100, 108),
                "age": range(17, 25),
                "study_hours": [Decimal(str(value)) for value in range(1, 9)],
                "overall_score": [
                    Decimal(str(value * 5 + 40)) for value in range(1, 9)
                ],
            }
        )
        service = PredictionService(decimal_df)

        self.assertEqual(
            service.get_feature_candidates("age"),
            ["study_hours", "overall_score"],
        )
        metrics = service.train_and_evaluate(
            feature_cols=["study_hours"],
            target_col="overall_score",
        )
        self.assertGreaterEqual(metrics["R2"], 0.9)

    def test_three_rows_train_and_report_r2_as_unavailable(self) -> None:
        df = pd.DataFrame(
            {
                "study_hours": [1, 2, 3],
                "score": [10, 20, 30],
            }
        )
        service = PredictionService(df)

        metrics = service.train_and_evaluate(
            feature_cols=["study_hours"],
            target_col="score",
        )

        self.assertEqual(len(service.result_df), 1)
        self.assertIsNone(metrics["R2"])

    def test_two_rows_give_a_clear_training_error(self) -> None:
        service = PredictionService(
            pd.DataFrame({"study_hours": [1, 2], "score": [10, 20]})
        )

        with self.assertRaisesRegex(ValueError, "tối thiểu 3 dòng"):
            service.train_and_evaluate(
                feature_cols=["study_hours"],
                target_col="score",
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
