"""Tests for preserving uploaded data and preparing cleaned rows for storage."""

import unittest

import pandas as pd

from modules.data_loader import (
    EXPECTED_COLUMNS,
    prepare_for_database,
    validate_dataset,
)


def make_dataset(student_ids: list[int]) -> pd.DataFrame:
    row_count = len(student_ids)
    return pd.DataFrame(
        {
            "student_id": student_ids,
            "age": [16] * row_count,
            "gender": ["female"] * row_count,
            "school_type": ["public"] * row_count,
            "parent_education": ["graduate"] * row_count,
            "study_hours": [2.5] * row_count,
            "attendance_percentage": [85.0] * row_count,
            "internet_access": ["yes"] * row_count,
            "travel_time": ["short"] * row_count,
            "extra_activities": ["no"] * row_count,
            "study_method": ["group"] * row_count,
            "math_score": [75.0] * row_count,
            "science_score": [80.0] * row_count,
            "english_score": [78.0] * row_count,
            "overall_score": [77.0] * row_count,
            "final_grade": ["B"] * row_count,
        }
    )


class UploadedDatasetTests(unittest.TestCase):
    def test_validation_preserves_duplicate_rows_missing_values_and_extra_columns(self):
        source = make_dataset([1, 1])
        source.loc[1, "math_score"] = None
        source["teacher_note"] = ["first", "second"]

        validated = validate_dataset(source)

        pd.testing.assert_frame_equal(validated, source)
        self.assertEqual(len(validated), 2)

    def test_validation_reports_missing_required_columns(self):
        source = make_dataset([1]).drop(columns=["overall_score"])

        with self.assertRaisesRegex(ValueError, "overall_score"):
            validate_dataset(source)

    def test_database_preparation_selects_required_columns_and_converts_numbers(self):
        source = make_dataset([1, 2])
        source["teacher_note"] = ["first", "second"]
        source["student_id"] = source["student_id"].astype(str)

        prepared = prepare_for_database(source)

        self.assertEqual(prepared.columns.tolist(), EXPECTED_COLUMNS)
        self.assertTrue(pd.api.types.is_integer_dtype(prepared["student_id"]))
        self.assertNotIn("teacher_note", prepared.columns)

    def test_database_preparation_rejects_duplicate_student_ids(self):
        with self.assertRaisesRegex(ValueError, "trùng student_id"):
            prepare_for_database(make_dataset([1, 1]))


if __name__ == "__main__":
    unittest.main()
