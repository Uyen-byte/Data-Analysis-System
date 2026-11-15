from psycopg2.extras import execute_values
from database.db_connection import connect_db


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS student_performance (
    student_id INTEGER PRIMARY KEY,
    age INTEGER NOT NULL,
    gender VARCHAR(20) NOT NULL,
    school_type VARCHAR(20) NOT NULL,
    parent_education VARCHAR(50) NOT NULL,
    study_hours NUMERIC(5, 2) NOT NULL,
    attendance_percentage NUMERIC(5, 2) NOT NULL,
    internet_access VARCHAR(10) NOT NULL,
    travel_time VARCHAR(20) NOT NULL,
    extra_activities VARCHAR(10) NOT NULL,
    study_method VARCHAR(30) NOT NULL,
    math_score NUMERIC(5, 2) NOT NULL,
    science_score NUMERIC(5, 2) NOT NULL,
    english_score NUMERIC(5, 2) NOT NULL,
    overall_score NUMERIC(5, 2) NOT NULL,
    final_grade VARCHAR(5) NOT NULL
);
"""

COLUMNS = [
    "student_id", "age", "gender", "school_type", "parent_education",
    "study_hours", "attendance_percentage", "internet_access", "travel_time",
    "extra_activities", "study_method", "math_score", "science_score",
    "english_score", "overall_score", "final_grade",
]


class DatasetRepository:
    def create_table(self):
        """Create the student_performance table if it does not exist."""
        conn = None
        try:
            conn = connect_db()
            with conn.cursor() as cursor:
                cursor.execute(CREATE_TABLE_SQL)
            conn.commit()
        except Exception:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def save_dataframe(self, dataframe):
        """Insert records; update an existing record with the same student_id."""
        if dataframe.empty:
            return 0

        # Kiểm tra xem DataFrame có thiếu cột bắt buộc nào không
        missing_cols = [col for col in COLUMNS if col not in dataframe.columns]
        if missing_cols:
            raise ValueError(f"DataFrame thiếu các cột bắt buộc sau: {missing_cols}")

        rows = [tuple(row) for row in dataframe[COLUMNS].itertuples(index=False, name=None)]
        update_columns = [column for column in COLUMNS if column != "student_id"]
        update_clause = ", ".join(
            f"{column} = EXCLUDED.{column}" for column in update_columns
        )
        sql = f"""
            INSERT INTO student_performance ({', '.join(COLUMNS)})
            VALUES %s
            ON CONFLICT (student_id) DO UPDATE SET {update_clause}
        """

        conn = None
        try:
            conn = connect_db()
            with conn.cursor() as cursor:
                execute_values(cursor, sql, rows, page_size=1000)
            conn.commit()
            return len(rows)
        except Exception:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def get_all(self, limit=1000, offset=0):
        conn = None
        try:
            conn = connect_db()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM student_performance ORDER BY student_id LIMIT %s OFFSET %s",
                    (limit, offset),
                )
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
            return rows, columns
        finally:
            if conn:
                conn.close()

    def search(self, keyword, limit=1000):
        conn = None
        try:
            conn = connect_db()
            with conn.cursor() as cursor:
                pattern = f"%{keyword}%"
                cursor.execute(
                    """
                    SELECT * FROM student_performance
                    WHERE CAST(student_id AS TEXT) ILIKE %s
                       OR gender ILIKE %s
                       OR school_type ILIKE %s
                       OR parent_education ILIKE %s
                       OR study_method ILIKE %s
                       OR final_grade ILIKE %s
                    ORDER BY student_id
                    LIMIT %s
                    """,
                    (pattern, pattern, pattern, pattern, pattern, pattern, limit),
                )
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
            return rows, columns
        finally:
            if conn:
                conn.close()

    def count_records(self):
        conn = None
        try:
            conn = connect_db()
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM student_performance")
                return cursor.fetchone()[0]
        finally:
            if conn:
                conn.close()