import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def connect_db():
    """Create and return a PostgreSQL connection."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )
        return conn
    except psycopg2.Error as error:
        raise ConnectionError(f"Không thể kết nối PostgreSQL: {error}") from error