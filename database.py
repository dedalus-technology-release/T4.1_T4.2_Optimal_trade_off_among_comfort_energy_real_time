import os
import sqlite3
from pathlib import Path

#DB_PATH = os.path.join(os.path.dirname(__file__), "data", "users.db")
DB_PATH = os.getenv("DB_PATH", "/app/data/users.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
        """)
        conn.commit()
