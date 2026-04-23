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
            CREATE TABLE IF NOT EXISTS return_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                order_id INTEGER NOT NULL,
                item_name TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS site_config (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS product_qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                question TEXT NOT NULL,
                asked_by TEXT NOT NULL,
                answer TEXT DEFAULT '',
                answered_at TEXT DEFAULT '',
                created_at TEXT NOT NULL
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gift_cards (
                code TEXT PRIMARY KEY,
                buyer_user_id TEXT NOT NULL,
                buyer_email TEXT DEFAULT '',
                buyer_name TEXT DEFAULT '',
                recipient_name TEXT DEFAULT '',
                recipient_email TEXT DEFAULT '',
                message TEXT DEFAULT '',
                value INTEGER NOT NULL,
                balance INTEGER NOT NULL,
                payment_id TEXT UNIQUE,
                redeemed_by TEXT DEFAULT '',
                redeemed_at TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gift_cards_buyer ON gift_cards(buyer_user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gift_cards_redeemed ON gift_cards(redeemed_by)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS package_enquiries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT '',
                user_name TEXT DEFAULT '',
                user_email TEXT DEFAULT '',
                phone TEXT DEFAULT '',
                event_date TEXT DEFAULT '',
                services TEXT NOT NULL,
                total INTEGER DEFAULT 0,
                discount INTEGER DEFAULT 0,
                message TEXT DEFAULT '',
                status TEXT DEFAULT 'new',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS b2b_enquiries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name TEXT NOT NULL,
                contact_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                email TEXT DEFAULT '',
                event_type TEXT DEFAULT '',
                event_date TEXT DEFAULT '',
                quantity INTEGER DEFAULT 0,
                location TEXT DEFAULT '',
                message TEXT DEFAULT '',
                status TEXT DEFAULT 'new',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wallet (
                user_id TEXT PRIMARY KEY,
                balance INTEGER DEFAULT 0,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wallet_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                amount INTEGER NOT NULL,
                type TEXT NOT NULL,
                description TEXT DEFAULT '',
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


def get_order_by_id(order_id: int) -> dict:
    with _get_connection() as conn:
        row = conn.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
        if not row:
            return {}
        d = dict(row)
        d["items"] = json.loads(d.get("items", "[]"))
        return d


def cancel_order(order_id: int, user_id: str) -> dict:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE id=? AND user_id=?", (order_id, user_id)
        ).fetchone()
        if not row:
            return {"ok": False, "error": "not_found"}
        if dict(row)["status"] not in ("confirmed",):
            return {"ok": False, "error": "not_cancellable"}
        conn.execute(
            "UPDATE orders SET status='cancelled' WHERE id=?", (order_id,)
        )
        conn.commit()
        return {"ok": True}


def save_return_request(user_id: str, order_id: int, item_name: str, reason: str) -> int:
    with _get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO return_requests (user_id, order_id, item_name, reason, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, order_id, item_name, reason, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cur.lastrowid


def get_return_requests(user_id: str) -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM return_requests WHERE user_id=? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_returns(limit: int = 200) -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM return_requests ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def update_return_status(return_id: int, status: str):
    with _get_connection() as conn:
        conn.execute("UPDATE return_requests SET status=? WHERE id=?", (status, return_id))
        conn.commit()


def get_analytics() -> dict:
    with _get_connection() as conn:
        total_orders = conn.execute("SELECT COUNT(*) AS c FROM orders").fetchone()["c"]
        total_revenue = conn.execute(
            "SELECT COALESCE(SUM(total),0) AS s FROM orders WHERE status != 'cancelled'"
        ).fetchone()["s"]
        total_customers = conn.execute(
            "SELECT COUNT(DISTINCT user_id) AS c FROM orders"
        ).fetchone()["c"]
        # last 30 days daily revenue
        daily = conn.execute(
            """SELECT substr(created_at,1,10) AS day, SUM(total) AS rev
               FROM orders WHERE status != 'cancelled'
               AND created_at >= date('now','-30 days')
               GROUP BY day ORDER BY day"""
        ).fetchall()
        # top products by quantity
        rows = conn.execute(
            "SELECT items FROM orders WHERE status != 'cancelled'"
        ).fetchall()
        product_counts: dict = {}
        for r in rows:
            try:
                items = json.loads(r["items"])
                for it in items:
                    name = it.get("name", "")
                    qty = it.get("qty", 1)
                    product_counts[name] = product_counts.get(name, 0) + qty
            except Exception:
                pass
        top_products = sorted(product_counts.items(), key=lambda x: -x[1])[:10]
        return {
            "total_orders": total_orders,
            "total_revenue": total_revenue,
            "total_customers": total_customers,
            "daily_revenue": [dict(d) for d in daily],
            "top_products": [{"name": k, "qty": v} for k, v in top_products],
        }


def get_customers(limit: int = 500) -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            """SELECT user_id, user_email, user_name,
                      COUNT(*) AS order_count, SUM(total) AS total_spent,
                      MAX(created_at) AS last_order
               FROM orders GROUP BY user_id ORDER BY last_order DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_site_config(key: str, default: str = "") -> str:
    with _get_connection() as conn:
        row = conn.execute("SELECT value FROM site_config WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def set_site_config(key: str, value: str):
    with _get_connection() as conn:
        conn.execute(
            """INSERT INTO site_config (key, value, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (key, value, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def add_product_qa(product_name: str, question: str, asked_by: str) -> int:
    with _get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO product_qa (product_name, question, asked_by, created_at)
               VALUES (?, ?, ?, ?)""",
            (product_name, question, asked_by, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cur.lastrowid


def answer_product_qa(qa_id: int, answer: str):
    with _get_connection() as conn:
        conn.execute(
            "UPDATE product_qa SET answer=?, answered_at=? WHERE id=?",
            (answer, datetime.now(timezone.utc).isoformat(), qa_id),
        )
        conn.commit()


def get_product_qa(product_name: str) -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM product_qa WHERE product_name=? ORDER BY created_at DESC",
            (product_name,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Gift Cards ──────────────────────────────────────────────────────────────────

def create_gift_card(buyer_user_id: str, value: int, recipient_name: str, recipient_email: str,
                     message: str, payment_id: str, buyer_email: str = "", buyer_name: str = "") -> str:
    """Create a gift card and return the unique code"""
    import random
    import string as _string
    with _get_connection() as conn:
        for _ in range(10):
            code = "".join(random.choices(_string.ascii_uppercase + _string.digits, k=8))
            code = f"GC{code}"
            if not conn.execute("SELECT 1 FROM gift_cards WHERE code=?", (code,)).fetchone():
                break
        conn.execute(
            """INSERT INTO gift_cards (code, buyer_user_id, buyer_email, buyer_name,
                                       recipient_name, recipient_email, message,
                                       value, balance, payment_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (code, buyer_user_id, buyer_email, buyer_name, recipient_name, recipient_email,
             message, value, value, payment_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return code


def get_gift_cards_for_user(user_id: str, limit: int = 50) -> list:
    """Get gift cards bought by a user"""
    with _get_connection() as conn:
        rows = conn.execute(
            """SELECT code, recipient_name, recipient_email, value, balance, created_at
               FROM gift_cards WHERE buyer_user_id=?
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_gift_cards(limit: int = 500) -> list:
    """Get all gift cards (for admin)"""
    with _get_connection() as conn:
        rows = conn.execute(
            """SELECT code, buyer_name, buyer_email, recipient_name, recipient_email,
                      value, balance, redeemed_by, redeemed_at, created_at
               FROM gift_cards ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_gift_card_by_code(code: str) -> dict:
    """Get a gift card by its code"""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM gift_cards WHERE code=?", (code.upper(),)
        ).fetchone()
        return dict(row) if row else {}


def redeem_gift_card(code: str, user_id: str, amount: int = None) -> dict:
    """Use/redeem a gift card for an order"""
    with _get_connection() as conn:
        card = conn.execute(
            "SELECT * FROM gift_cards WHERE code=?", (code.upper(),)
        ).fetchone()
        if not card:
            return {"success": False, "error": "Card not found"}
        card_dict = dict(card)
        if card_dict["balance"] <= 0:
            return {"success": False, "error": "Card has no balance"}
        if amount is None:
            amount = card_dict["balance"]
        else:
            amount = min(amount, card_dict["balance"])
        new_balance = card_dict["balance"] - amount
        conn.execute(
            "UPDATE gift_cards SET balance=?, redeemed_by=?, redeemed_at=? WHERE code=?",
            (new_balance, user_id, datetime.now(timezone.utc).isoformat(), code.upper()),
        )
        conn.commit()
        return {"success": True, "amount_redeemed": amount, "new_balance": new_balance}


# ── Loyalty tiers ──────────────────────────────────────────────────────────────

TIERS = [
    {"name": "Bronze",   "icon": "🥉", "min": 0,    "max": 499,   "multiplier": 1.0, "color": "#cd7f32"},
    {"name": "Silver",   "icon": "🥈", "min": 500,  "max": 1499,  "multiplier": 1.2, "color": "#9ca3af"},
    {"name": "Gold",     "icon": "🥇", "min": 1500, "max": 4999,  "multiplier": 1.5, "color": "#BF8522"},
    {"name": "Platinum", "icon": "💎", "min": 5000, "max": 999999, "multiplier": 2.0, "color": "#6B3820"},
]

def get_tier(points: int) -> dict:
    for t in reversed(TIERS):
        if points >= t["min"]:
            return t
    return TIERS[0]


# ── Package enquiries ──────────────────────────────────────────────────────────

def save_package_enquiry(user_id: str, user_name: str, user_email: str, phone: str,
                          event_date: str, services: list, total: int, discount: int,
                          message: str = "") -> int:
    with _get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO package_enquiries
               (user_id, user_name, user_email, phone, event_date, services, total, discount, message, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, user_name, user_email, phone, event_date, json.dumps(services),
             total, discount, message, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cur.lastrowid


def get_all_package_enquiries(limit: int = 300) -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM package_enquiries ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["services"] = json.loads(d.get("services", "[]"))
            result.append(d)
        return result


def update_package_enquiry_status(enquiry_id: int, status: str):
    with _get_connection() as conn:
        conn.execute("UPDATE package_enquiries SET status=? WHERE id=?", (status, enquiry_id))
        conn.commit()


# ── B2B enquiries ──────────────────────────────────────────────────────────────

def save_b2b_enquiry(company_name: str, contact_name: str, phone: str, email: str,
                     event_type: str, event_date: str, quantity: int,
                     location: str, message: str) -> int:
    with _get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO b2b_enquiries
               (company_name, contact_name, phone, email, event_type, event_date,
                quantity, location, message, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (company_name, contact_name, phone, email, event_type, event_date,
             quantity, location, message, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cur.lastrowid


def get_all_b2b_enquiries(limit: int = 300) -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM b2b_enquiries ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ── Wallet ──────────────────────────────────────────────────────────────────────

def get_wallet_balance(user_id: str) -> int:
    with _get_connection() as conn:
        row = conn.execute("SELECT balance FROM wallet WHERE user_id=?", (user_id,)).fetchone()
        return row["balance"] if row else 0


def get_wallet_transactions(user_id: str, limit: int = 30) -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT id, amount, type, description, created_at FROM wallet_transactions WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def add_wallet_credit(user_id: str, amount: int, description: str = "") -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        conn.execute(
            """INSERT INTO wallet (user_id, balance, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET balance=balance+excluded.balance, updated_at=excluded.updated_at""",
            (user_id, amount, now),
        )
        conn.execute(
            "INSERT INTO wallet_transactions (user_id, amount, type, description, created_at) VALUES (?, ?, 'credit', ?, ?)",
            (user_id, amount, description, now),
        )
        conn.commit()
        row = conn.execute("SELECT balance FROM wallet WHERE user_id=?", (user_id,)).fetchone()
        return row["balance"] if row else 0


def deduct_wallet(user_id: str, amount: int, description: str = "") -> dict:
    now = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        balance = get_wallet_balance(user_id)
        if balance < amount:
            return {"success": False, "error": "Insufficient wallet balance"}
        conn.execute(
            "UPDATE wallet SET balance=balance-?, updated_at=? WHERE user_id=?",
            (amount, now, user_id),
        )
        conn.execute(
            "INSERT INTO wallet_transactions (user_id, amount, type, description, created_at) VALUES (?, ?, 'debit', ?, ?)",
            (user_id, amount, description, now),
        )
        conn.commit()
        new_balance = get_wallet_balance(user_id)
        return {"success": True, "new_balance": new_balance}


def update_order_payment(order_id: int, user_id: str, new_payment_id: str) -> dict:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE id=? AND user_id=?", (order_id, user_id)
        ).fetchone()
        if not row:
            return {"ok": False, "error": "not_found"}
        conn.execute(
            "UPDATE orders SET payment_id=? WHERE id=?", (new_payment_id, order_id)
        )
        conn.commit()
        return {"ok": True}


def update_b2b_status(enquiry_id: int, status: str):
    with _get_connection() as conn:
        conn.execute("UPDATE b2b_enquiries SET status=? WHERE id=?", (status, enquiry_id))
        conn.commit()

