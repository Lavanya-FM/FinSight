# services/db.py
import psycopg2
import os

def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "loan_analytics"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS", "your_password"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432)
    )
