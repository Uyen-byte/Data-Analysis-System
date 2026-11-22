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


def validate_dataset(dataframe):
    """Validate the required schema without modifying or filtering uploaded rows."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Dữ liệu đầu vào phải là DataFrame.")
    if dataframe.empty:
        raise ValueError("Tệp dữ liệu không có dòng nào.")

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError("File thiếu các cột: " + ", ".join(missing_columns))

    return dataframe.copy()


def prepare_for_database(dataframe):
    """Validate and shape cleaned data for the PostgreSQL student table."""
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError("File thiếu các cột: " + ", ".join(missing_columns))

    prepared = dataframe[EXPECTED_COLUMNS].copy()
    if prepared.empty:
        raise ValueError("Không có dữ liệu đã làm sạch để lưu.")
    if prepared.isnull().any().any():
        invalid_columns = prepared.columns[prepared.isnull().any()].tolist()
        raise ValueError(
            "Dữ liệu còn ô trống tại: " + ", ".join(invalid_columns)
        )

    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    if prepared[NUMERIC_COLUMNS].isnull().any().any():
        invalid_columns = prepared[NUMERIC_COLUMNS].columns[
            prepared[NUMERIC_COLUMNS].isnull().any()
        ].tolist()
        raise ValueError(
            "Có giá trị không phải số tại: " + ", ".join(invalid_columns)
        )

    for column in ("student_id", "age"):
        if (prepared[column] % 1 != 0).any():
            raise ValueError(f"Cột {column} chỉ được chứa số nguyên.")
        prepared[column] = prepared[column].astype(int)

    duplicate_count = int(prepared["student_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"Còn {duplicate_count} dòng trùng student_id. "
            "Hãy xử lý trùng lặp trong Data Cleaning trước khi lưu."
        )

    return prepared.reset_index(drop=True)
