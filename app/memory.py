import sqlite3
import json
import logging
from datetime import datetime, timezone
from app.config import DB_PATH

logger = logging.getLogger("tdk.memory")


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_memory (
                user_id TEXT PRIMARY KEY,
                preferences TEXT DEFAULT '[]',
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                score INTEGER,
                iterations INTEGER,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)")
        conn.commit()
    logger.info("Database initialized")


def get_memory(user_id: str) -> list:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT preferences FROM user_memory WHERE user_id = ?", (user_id,)
        ).fetchone()
        return json.loads(row["preferences"]) if row else []


def save_memory(user_id: str, info: str):
    prefs = get_memory(user_id)
    prefs.append(info)
    with _get_connection() as conn:
        conn.execute(
            """INSERT INTO user_memory (user_id, preferences, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET preferences=excluded.preferences, updated_at=excluded.updated_at""",
            (user_id, json.dumps(prefs), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def save_conversation(user_id: str, query: str, answer: str, score: int, iterations: int):
    with _get_connection() as conn:
        conn.execute(
            """INSERT INTO conversations (user_id, query, answer, score, iterations, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, query, answer, score, iterations, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def get_conversations(user_id: str, limit: int = 20) -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            """SELECT id, query, answer, score, iterations, created_at
               FROM conversations WHERE user_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


def clear_conversations(user_id: str):
    with _get_connection() as conn:
        conn.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
        conn.commit()
