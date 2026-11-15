import pandas as pd

EXPECTED_COLUMNS = [
    "student_id", "age", "gender", "school_type", "parent_education",
    "study_hours", "attendance_percentage", "internet_access", "travel_time",
    "extra_activities", "study_method", "math_score", "science_score",
    "english_score", "overall_score", "final_grade",
]

NUMERIC_COLUMNS = [
    "student_id", "age", "study_hours", "attendance_percentage",
    "math_score", "science_score", "english_score", "overall_score",
]


def load_file(uploaded_file):
    """Read a CSV or Excel file and return a DataFrame."""
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        dataframe = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xlsx", ".xls")):
        dataframe = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Chỉ hỗ trợ file CSV hoặc Excel (.xlsx, .xls).")
    
    return dataframe  # Đã bổ sung return để truyền dữ liệu đi tiếp!


def validate_and_clean(dataframe):
    """Check required columns, convert numeric fields and remove duplicate rows."""
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError("File thiếu các cột: " + ", ".join(missing_columns))

    dataframe = dataframe[EXPECTED_COLUMNS].copy()

    for column in NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    if dataframe.isnull().any().any():
        invalid_columns = dataframe.columns[dataframe.isnull().any()].tolist()
        raise ValueError(
            "Dữ liệu có ô trống hoặc sai định dạng tại: " + ", ".join(invalid_columns)
        )

    dataframe["student_id"] = dataframe["student_id"].astype(int)
    dataframe["age"] = dataframe["age"].astype(int)

    if dataframe["student_id"].duplicated().any():
        dataframe = dataframe.drop_duplicates(subset="student_id", keep="first")

    return dataframe.drop_duplicates().reset_index(drop=True)