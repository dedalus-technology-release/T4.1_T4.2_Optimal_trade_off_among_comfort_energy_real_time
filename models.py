import sqlite3
from passlib.context import CryptContext
from database import get_connection

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_user(username: str, password: str, is_admin: bool = False):
    password_hash = pwd_context.hash(password)
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                (username, password_hash, int(is_admin))
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError("Username already exists")

def get_user_by_username(username):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        return user

def authenticate_user(username: str, password: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if not row:
            return False
        return pwd_context.verify(password, row[0])

def is_admin(username: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT is_admin FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        return bool(row[0]) if row else False
