"""Focused tests for data analysis and visualization features."""

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.data_analysis import DataAnalyzer
from modules.visualization import (
    create_bar_chart,
    create_correlation_heatmap,
    create_line_chart,
    create_pie_chart,
    create_scatter_chart,
)


class DataAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "score": [10.0, 20.0, 30.0, 40.0],
                "hours": [1.0, 2.0, 3.0, 4.0],
                "group": ["A", "A", "B", "B"],
            }
        )
        self.analyzer = DataAnalyzer(self.df)

    def test_overview_and_descriptive_statistics(self) -> None:
        overview = self.analyzer.get_overview()
        self.assertEqual(overview["total_rows"], 4)
        self.assertEqual(overview["total_columns"], 3)
        self.assertEqual(overview["missing_values"], 0)
        self.assertEqual(overview["duplicate_rows"], 0)

        statistics = self.analyzer.get_descriptive_statistics()
        self.assertIn("score", statistics.index)
        self.assertNotIn("group", statistics.index)
        self.assertEqual(statistics.loc["score", "mean"], 25.0)

    def test_selected_column_statistics(self) -> None:
        statistics = self.analyzer.analyze_column("score")
        self.assertEqual(statistics["Sum"], 100.0)
        self.assertEqual(statistics["Mean"], 25.0)
        self.assertEqual(statistics["Median"], 25.0)
        self.assertEqual(statistics["Max"], 40.0)
        self.assertEqual(statistics["Min"], 10.0)

    def test_column_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Không tìm thấy cột"):
            self.analyzer.analyze_column("not_a_column")
        with self.assertRaisesRegex(ValueError, "dạng số"):
            self.analyzer.analyze_column("group")

    def test_correlation_uses_numeric_columns(self) -> None:
        correlation = self.analyzer.get_correlation()
        self.assertIn("score", correlation.columns)
        self.assertNotIn("group", correlation.columns)
        self.assertAlmostEqual(correlation.loc["score", "hours"], 1.0)


class VisualizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "score": [10.0, 20.0, 30.0, 40.0],
                "hours": [1.0, 2.0, 3.0, 4.0],
                "group": ["A", "A", "B", "B"],
            }
        )

    def test_all_required_charts_return_plotly_figures(self) -> None:
        figures = [
            create_bar_chart(self.df, "group", "score"),
            create_line_chart(self.df, "hours", "score"),
            create_pie_chart(self.df, "group"),
            create_scatter_chart(self.df, "hours", "score"),
            create_correlation_heatmap(self.df[["score", "hours"]].corr()),
        ]
        for figure in figures:
            self.assertGreater(len(figure.data), 0)


if __name__ == "__main__":
    unittest.main()
