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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                items TEXT NOT NULL,
                total INTEGER NOT NULL,
                payment_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS coupons (
                code TEXT PRIMARY KEY,
                discount_type TEXT NOT NULL,
                discount_value INTEGER NOT NULL,
                max_usage INTEGER DEFAULT 0,
                usage_count INTEGER DEFAULT 0,
                active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wishlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                product_name TEXT NOT NULL,
                product_price INTEGER NOT NULL,
                UNIQUE(user_id, product_name)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                user_name TEXT NOT NULL,
                product_name TEXT NOT NULL,
                rating INTEGER NOT NULL,
                review_text TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS loyalty_points (
                user_id TEXT PRIMARY KEY,
                points INTEGER DEFAULT 0,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS referral_codes (
                code TEXT PRIMARY KEY,
                user_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS referral_uses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                new_user_id TEXT NOT NULL UNIQUE,
                payment_id TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        for alter in [
            "ALTER TABLE orders ADD COLUMN status TEXT DEFAULT 'confirmed'",
            "ALTER TABLE orders ADD COLUMN address TEXT DEFAULT ''",
            "ALTER TABLE orders ADD COLUMN discount INTEGER DEFAULT 0",
            "ALTER TABLE orders ADD COLUMN coupon_code TEXT DEFAULT ''",
            "ALTER TABLE orders ADD COLUMN user_email TEXT DEFAULT ''",
            "ALTER TABLE orders ADD COLUMN user_name TEXT DEFAULT ''",
        ]:
            try:
                conn.execute(alter)
            except Exception:
                pass
        # Seed default coupons
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            INSERT OR IGNORE INTO coupons (code, discount_type, discount_value, max_usage, created_at)
            VALUES
            ('WELCOME10', 'percent', 10, 0, ?),
            ('MEHANDI50', 'flat', 50, 100, ?),
            ('BRIDAL15', 'percent', 15, 50, ?)
        """, (now, now, now))
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


def save_order(user_id: str, items: list, total: int, payment_id: str,
               address: str = "", discount: int = 0, coupon_code: str = "",
               user_email: str = "", user_name: str = ""):
    with _get_connection() as conn:
        conn.execute(
            """INSERT INTO orders (user_id, items, total, payment_id, created_at, status,
               address, discount, coupon_code, user_email, user_name)
               VALUES (?, ?, ?, ?, ?, 'confirmed', ?, ?, ?, ?, ?)""",
            (user_id, json.dumps(items), total, payment_id,
             datetime.now(timezone.utc).isoformat(), address, discount, coupon_code,
             user_email, user_name),
        )
        conn.commit()


def get_orders(user_id: str, limit: int = 20) -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            """SELECT id, items, total, payment_id, created_at, status, address, discount
               FROM orders WHERE user_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["items"] = json.loads(d["items"])
            result.append(d)
        return result


def get_all_orders(limit: int = 200) -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["items"] = json.loads(d["items"])
            result.append(d)
        return result


def update_order_status(order_id: int, status: str) -> dict:
    with _get_connection() as conn:
        conn.execute("UPDATE orders SET status=? WHERE id=?", (status, order_id))
        conn.commit()
        row = conn.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
        if row:
            d = dict(row)
            d["items"] = json.loads(d.get("items", "[]"))
            return d
        return {}


def validate_coupon(code: str):
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM coupons WHERE code=? AND active=1", (code.upper(),)
        ).fetchone()
        if not row:
            return None
        r = dict(row)
        if r["max_usage"] > 0 and r["usage_count"] >= r["max_usage"]:
            return None
        return r


def use_coupon(code: str):
    with _get_connection() as conn:
        conn.execute("UPDATE coupons SET usage_count=usage_count+1 WHERE code=?", (code.upper(),))
        conn.commit()


def create_coupon(code: str, discount_type: str, discount_value: int, max_usage: int):
    with _get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO coupons (code, discount_type, discount_value, max_usage, usage_count, active, created_at)
               VALUES (?, ?, ?, ?, 0, 1, ?)""",
            (code.upper(), discount_type, discount_value, max_usage, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def list_coupons() -> list:
    with _get_connection() as conn:
        rows = conn.execute("SELECT * FROM coupons ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def add_to_wishlist(user_id: str, product_name: str, product_price: int):
    with _get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO wishlist (user_id, product_name, product_price) VALUES (?, ?, ?)",
            (user_id, product_name, product_price),
        )
        conn.commit()


def remove_from_wishlist(user_id: str, product_name: str):
    with _get_connection() as conn:
        conn.execute("DELETE FROM wishlist WHERE user_id=? AND product_name=?", (user_id, product_name))
        conn.commit()


def get_wishlist(user_id: str) -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT product_name, product_price FROM wishlist WHERE user_id=? ORDER BY id DESC", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def add_review(user_id: str, user_name: str, product_name: str, rating: int, review_text: str):
    with _get_connection() as conn:
        conn.execute(
            """INSERT INTO reviews (user_id, user_name, product_name, rating, review_text, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, user_name, product_name, rating, review_text, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def get_reviews(product_name: str) -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT user_name, rating, review_text, created_at FROM reviews WHERE product_name=? ORDER BY created_at DESC",
            (product_name,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_loyalty_points(user_id: str) -> int:
    with _get_connection() as conn:
        row = conn.execute("SELECT points FROM loyalty_points WHERE user_id=?", (user_id,)).fetchone()
        return row["points"] if row else 0


def add_loyalty_points(user_id: str, points: int):
    with _get_connection() as conn:
        conn.execute(
            """INSERT INTO loyalty_points (user_id, points, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET points=points+excluded.points, updated_at=excluded.updated_at""",
            (user_id, points, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def use_loyalty_points(user_id: str, points: int):
    with _get_connection() as conn:
        conn.execute(
            "UPDATE loyalty_points SET points=MAX(0,points-?), updated_at=? WHERE user_id=?",
            (points, datetime.now(timezone.utc).isoformat(), user_id),
        )
        conn.commit()


def get_or_create_referral_code(user_id: str, display_name: str) -> str:
    import random
    import string as _string
    with _get_connection() as conn:
        row = conn.execute("SELECT code FROM referral_codes WHERE user_id=?", (user_id,)).fetchone()
        if row:
            return row["code"]
        name_part = "".join(c for c in display_name.upper() if c.isalpha())[:4] or "USER"
        for _ in range(10):
            rand = "".join(random.choices(_string.ascii_uppercase + _string.digits, k=5))
            code = f"MT{name_part}{rand}"
            if not conn.execute("SELECT 1 FROM referral_codes WHERE code=?", (code,)).fetchone():
                break
        conn.execute(
            "INSERT OR IGNORE INTO referral_codes (code, user_id, created_at) VALUES (?, ?, ?)",
            (code, user_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return code


def get_referral_stats(user_id: str) -> dict:
    with _get_connection() as conn:
        row = conn.execute("SELECT code FROM referral_codes WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            return {"code": None, "uses": 0}
        code = row["code"]
        cnt = conn.execute("SELECT COUNT(*) AS c FROM referral_uses WHERE code=?", (code,)).fetchone()
        return {"code": code, "uses": cnt["c"] if cnt else 0}


def validate_referral_code(code: str, new_user_id: str = "") -> dict:
    with _get_connection() as conn:
        row = conn.execute("SELECT * FROM referral_codes WHERE code=?", (code.upper(),)).fetchone()
        if not row:
            return {"valid": False, "error": "not_found"}
        r = dict(row)
        if new_user_id and r["user_id"] == new_user_id:
            return {"valid": False, "error": "self"}
        if new_user_id:
            used = conn.execute("SELECT 1 FROM referral_uses WHERE new_user_id=?", (new_user_id,)).fetchone()
            if used:
                return {"valid": False, "error": "already_used"}
        return {"valid": True, "referrer_user_id": r["user_id"]}


def record_referral_use(code: str, new_user_id: str, payment_id: str = ""):
    with _get_connection() as conn:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO referral_uses (code, new_user_id, payment_id, created_at) VALUES (?, ?, ?, ?)",
                (code.upper(), new_user_id, payment_id, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        except Exception:
            pass
