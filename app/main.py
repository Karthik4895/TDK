import base64
import csv
import hmac
import hashlib
import io
import os
import time
import logging
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from app.models import QueryRequest, QueryResponse, HealthResponse
from app.graph import graph
from app.memory import (
    init_db, save_conversation, get_conversations, clear_conversations,
    save_order, get_orders, get_all_orders, update_order_status,
    get_order_by_id, cancel_order,
    validate_coupon, use_coupon, create_coupon, list_coupons,
    add_to_wishlist, remove_from_wishlist, get_wishlist,
    add_review, get_reviews,
    get_loyalty_points, add_loyalty_points, use_loyalty_points,
    get_or_create_referral_code, get_referral_stats,
    validate_referral_code, record_referral_use,
    save_return_request, get_return_requests, get_all_returns, update_return_status,
    get_analytics, get_customers,
    get_site_config, set_site_config,
    add_product_qa, answer_product_qa, get_product_qa,
    get_tier,
    save_package_enquiry, get_all_package_enquiries, update_package_enquiry_status,
    save_b2b_enquiry, get_all_b2b_enquiries, update_b2b_status,
    create_gift_card, get_gift_cards_for_user, get_all_gift_cards, get_gift_card_by_code, redeem_gift_card,
    get_wallet_balance, get_wallet_transactions, add_wallet_credit, deduct_wallet, update_order_payment,
)
from app.config import APP_ENV, RESEND_API_KEY, ADMIN_EMAIL, ADMIN_SECRET, FIREBASE_API_KEY

try:
    import razorpay as _razorpay
except ImportError:
    _razorpay = None

logger = logging.getLogger("tdk.main")

APP_VERSION = "1.0.0"


# ── Admin auth ─────────────────────────────────────────────────────────────────

async def _verify_firebase_id_token(id_token: str) -> dict | None:
    """Call Firebase Auth REST API to verify an ID token and return the user record."""
    if not FIREBASE_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_API_KEY}",
                json={"idToken": id_token},
            )
            if resp.status_code != 200:
                return None
            users = resp.json().get("users", [])
            return users[0] if users else None
    except Exception:
        return None


def _make_admin_token() -> str:
    """Create a signed, time-limited admin session token (8 h)."""
    exp = int(time.time()) + 8 * 3600
    payload = f"{ADMIN_EMAIL}|{exp}"
    sig = hmac.new(ADMIN_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.b64encode(f"{payload}|{sig}".encode()).decode()


def _verify_admin_token(token: str) -> bool:
    """Verify an admin session token.  Returns False if expired or tampered."""
    if not ADMIN_SECRET or not token:
        return False
    try:
        decoded = base64.b64decode(token.encode()).decode()
        email, exp_str, sig = decoded.rsplit("|", 2)
        if email != ADMIN_EMAIL:
            return False
        if int(exp_str) < int(time.time()):
            return False
        expected = hmac.new(
            ADMIN_SECRET.encode(), f"{email}|{exp_str}".encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, sig)
    except Exception:
        return False


def _require_admin(x_admin_token: str = Header(default="")):
    """FastAPI dependency: reject requests that don't carry a valid admin token."""
    if not _verify_admin_token(x_admin_token):
        raise HTTPException(status_code=403, detail="Admin access required")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    logger.info("TDK API v%s started in %s mode", APP_VERSION, APP_ENV)
    yield
    logger.info("TDK API shutting down")


app = FastAPI(
    title="TDK Mehendi Assistant",
    description="AI-powered mehendi service assistant with agentic RAG pipeline",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("app/static/index.html")


@app.get("/sw.js", include_in_schema=False)
async def service_worker():
    return FileResponse(
        "app/static/sw.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )


@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
async def apple_touch_icon():
    png_path = "app/static/apple-touch-icon.png"
    if os.path.exists(png_path):
        return FileResponse(png_path, media_type="image/png")
    # Fallback: serve SVG with image/png content-type is wrong,
    # so return the SVG as-is (iOS 15.4+ handles it)
    return FileResponse("app/static/icon-512.svg", media_type="image/svg+xml")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return {"status": "ok", "version": APP_VERSION, "environment": APP_ENV}


@app.get("/config", include_in_schema=False)
async def frontend_config():
    """Serve Firebase and public config to the frontend from environment variables."""
    return JSONResponse({
        "firebase": {
            "apiKey":            os.getenv("FIREBASE_API_KEY", ""),
            "authDomain":        os.getenv("FIREBASE_AUTH_DOMAIN", ""),
            "projectId":         os.getenv("FIREBASE_PROJECT_ID", ""),
            "storageBucket":     os.getenv("FIREBASE_STORAGE_BUCKET", ""),
            "messagingSenderId": os.getenv("FIREBASE_MSG_SENDER_ID", ""),
            "appId":             os.getenv("FIREBASE_APP_ID", ""),
        },
        "razorpayKeyId": os.getenv("RAZORPAY_KEY_ID", ""),
    })


@app.post("/ask", response_model=QueryResponse, tags=["chat"])
async def ask(req: QueryRequest):
    if not req.user_id.strip():
        raise HTTPException(status_code=422, detail="user_id must not be empty")
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")

    start = time.perf_counter()
    logger.info("Query from '%s': %s", req.user_id, req.query[:100])

    try:
        result = graph.invoke({
            "user_id": req.user_id.strip(),
            "query": req.query.strip(),
            "refined_query": "",
            "context": "",
            "draft": "",
            "review": "",
            "score": 0,
            "iteration": 0,
            "memory": [],
        })
    except Exception as exc:
        logger.error("Graph error for user '%s': %s", req.user_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process your query. Please try again.")

    elapsed = time.perf_counter() - start
    logger.info("Done in %.2fs — score=%d iterations=%d", elapsed, result["score"], result["iteration"])

    save_conversation(req.user_id, req.query, result["draft"], result["score"], result["iteration"])
    return {"answer": result["draft"], "score": result["score"], "iterations": result["iteration"]}


# ── Payment ────────────────────────────────────────────────────────────────────

class OrderRequest(BaseModel):
    amount: int
    currency: str = "INR"
    receipt: str = "order_receipt"

class VerifyRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


@app.get("/payment-status", tags=["payment"])
async def payment_status():
    key_id = os.getenv("RAZORPAY_KEY_ID", "")
    secret  = os.getenv("RAZORPAY_KEY_SECRET", "")
    return {
        "key_id_set": bool(key_id),
        "key_id_prefix": key_id[:8] if key_id else "NOT SET",
        "secret_set": bool(secret),
        "razorpay_pkg": _razorpay is not None,
    }


@app.post("/create-order", tags=["payment"])
async def create_order(req: OrderRequest):
    key_id = os.getenv("RAZORPAY_KEY_ID", "")
    secret  = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not key_id or not secret:
        raise HTTPException(status_code=503, detail="Payment not configured")
    if _razorpay is None:
        raise HTTPException(status_code=503, detail="Razorpay package not available")
    try:
        client = _razorpay.Client(auth=(key_id, secret))
        order = client.order.create({
            "amount": req.amount,
            "currency": req.currency,
            "receipt": req.receipt,
            "payment_capture": 1,
        })
        return {"order_id": order["id"], "key_id": key_id, "amount": req.amount}
    except Exception as exc:
        logger.error("Razorpay order creation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/verify-payment", tags=["payment"])
async def verify_payment(req: VerifyRequest):
    secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not secret:
        raise HTTPException(status_code=503, detail="Payment not configured")
    body = f"{req.razorpay_order_id}|{req.razorpay_payment_id}"
    expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, req.razorpay_signature):
        raise HTTPException(status_code=400, detail="Payment verification failed")
    logger.info("Payment verified: %s", req.razorpay_payment_id)
    return {"success": True, "payment_id": req.razorpay_payment_id}


# ── Orders ─────────────────────────────────────────────────────────────────────

class OrderSaveRequest(BaseModel):
    user_id: str
    user_email: str = ""
    user_name: str = ""
    items: list
    total: int
    payment_id: str
    address: str = ""
    discount: int = 0
    loyalty_used: int = 0
    coupon_code: str = ""
    referral_code: str = ""
    referrer_user_id: str = ""


@app.post("/orders", tags=["orders"])
async def create_order_record(req: OrderSaveRequest):
    if not req.user_id.strip():
        raise HTTPException(status_code=422, detail="user_id must not be empty")
    save_order(
        req.user_id.strip(), req.items, req.total, req.payment_id,
        address=req.address, discount=req.discount, coupon_code=req.coupon_code,
        user_email=req.user_email, user_name=req.user_name,
    )
    earned = max(0, req.total // 10)
    if earned:
        add_loyalty_points(req.user_id.strip(), earned)
    if req.loyalty_used:
        use_loyalty_points(req.user_id.strip(), req.loyalty_used)
    if req.coupon_code:
        use_coupon(req.coupon_code)
    if req.referral_code and req.user_id:
        record_referral_use(req.referral_code, req.user_id.strip(), req.payment_id)
        if req.referrer_user_id:
            add_loyalty_points(req.referrer_user_id, 100)
            logger.info("Referrer %s earned 100pts via code %s", req.referrer_user_id, req.referral_code)
    if req.user_email:
        await _send_order_email(req.user_email, req.user_name, req.items, req.total, req.payment_id, req.address)
    return {"saved": True, "loyalty_earned": earned}


@app.get("/orders/{user_id}", tags=["orders"])
async def fetch_orders(user_id: str, limit: int = 20):
    return get_orders(user_id.strip(), min(limit, 50))


class CancelOrderRequest(BaseModel):
    user_id: str


@app.post("/orders/{order_id}/cancel", tags=["orders"])
async def cancel_order_endpoint(order_id: int, req: CancelOrderRequest):
    result = cancel_order(order_id, req.user_id.strip())
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Cannot cancel"))
    return {"cancelled": True}


# ── COD Order ─────────────────────────────────────────────────────────────────

class CODOrderRequest(BaseModel):
    user_id: str
    user_email: str = ""
    user_name: str = ""
    items: list
    total: int
    address: str = ""
    discount: int = 0
    loyalty_used: int = 0
    coupon_code: str = ""
    referral_code: str = ""
    referrer_user_id: str = ""


@app.post("/cod-order", tags=["orders"])
async def cod_order(req: CODOrderRequest):
    if not req.user_id.strip():
        raise HTTPException(status_code=422, detail="user_id required")
    payment_id = f"COD-{int(time.time())}-{req.user_id[:6]}"
    save_order(
        req.user_id.strip(), req.items, req.total, payment_id,
        address=req.address, discount=req.discount, coupon_code=req.coupon_code,
        user_email=req.user_email, user_name=req.user_name,
    )
    earned = max(0, req.total // 10)
    if earned:
        add_loyalty_points(req.user_id.strip(), earned)
    if req.loyalty_used:
        use_loyalty_points(req.user_id.strip(), req.loyalty_used)
    if req.coupon_code:
        use_coupon(req.coupon_code)
    if req.referral_code and req.user_id:
        record_referral_use(req.referral_code, req.user_id.strip(), payment_id)
        if req.referrer_user_id:
            add_loyalty_points(req.referrer_user_id, 100)
    if req.user_email:
        await _send_order_email(req.user_email, req.user_name, req.items, req.total, payment_id, req.address)
    return {"saved": True, "payment_id": payment_id, "loyalty_earned": earned}


class OrderStatusRequest(BaseModel):
    status: str


@app.put("/orders/{order_id}/status", tags=["orders"])
async def set_order_status(
    order_id: int,
    req: OrderStatusRequest,
    _: None = Depends(_require_admin),
):
    if req.status not in ("confirmed", "packed", "shipped", "delivered"):
        raise HTTPException(status_code=422, detail="Invalid status")
    order = update_order_status(order_id, req.status)
    if req.status in ("packed", "shipped", "delivered") and order.get("user_email"):
        await _send_status_email(
            order["user_email"], order.get("user_name", ""), req.status, order.get("payment_id", "")
        )
    return {"updated": True}


# ── Admin ──────────────────────────────────────────────────────────────────────

@app.post("/admin/session", tags=["admin"])
async def admin_create_session(authorization: str = Header(default="")):
    """Exchange a valid Firebase ID token for a signed 8-hour admin session token."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Firebase ID token")
    id_token = authorization[7:]
    user_info = await _verify_firebase_id_token(id_token)
    if not user_info or user_info.get("email") != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorised as admin")
    if not ADMIN_SECRET:
        raise HTTPException(status_code=503, detail="Admin secret not configured")
    return {"token": _make_admin_token()}


@app.get("/admin/orders", tags=["admin"])
async def admin_orders(_: None = Depends(_require_admin)):
    return get_all_orders()


@app.get("/admin/coupons", tags=["admin"])
async def admin_coupons(_: None = Depends(_require_admin)):
    return list_coupons()


class CouponCreateRequest(BaseModel):
    code: str
    discount_type: str
    discount_value: int
    max_usage: int = 0


@app.post("/admin/coupons", tags=["admin"])
async def admin_create_coupon(req: CouponCreateRequest, _: None = Depends(_require_admin)):
    if req.discount_type not in ("percent", "flat"):
        raise HTTPException(status_code=422, detail="discount_type must be 'percent' or 'flat'")
    create_coupon(req.code, req.discount_type, req.discount_value, req.max_usage)
    return {"created": True}


@app.get("/admin/analytics", tags=["admin"])
async def admin_analytics(_: None = Depends(_require_admin)):
    return get_analytics()


@app.get("/admin/customers", tags=["admin"])
async def admin_customers(_: None = Depends(_require_admin)):
    return get_customers()


@app.get("/admin/export/orders", tags=["admin"])
async def admin_export_orders(_: None = Depends(_require_admin)):
    orders = get_all_orders(limit=5000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "User ID", "Name", "Email", "Items", "Total", "Discount",
                     "Coupon", "Payment ID", "Address", "Status", "Created At"])
    for o in orders:
        items_str = "; ".join(f"{i['name']} x{i['qty']} @₹{i['price']}" for i in o.get("items", []))
        writer.writerow([
            o.get("id"), o.get("user_id"), o.get("user_name"), o.get("user_email"),
            items_str, o.get("total"), o.get("discount"), o.get("coupon_code"),
            o.get("payment_id"), o.get("address"), o.get("status"), o.get("created_at"),
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=orders.csv"},
    )


@app.get("/admin/returns", tags=["admin"])
async def admin_returns(_: None = Depends(_require_admin)):
    return get_all_returns()


class ReturnStatusRequest(BaseModel):
    status: str


@app.put("/admin/returns/{return_id}", tags=["admin"])
async def admin_update_return(return_id: int, req: ReturnStatusRequest, _: None = Depends(_require_admin)):
    if req.status not in ("pending", "approved", "rejected", "completed"):
        raise HTTPException(status_code=422, detail="Invalid status")
    update_return_status(return_id, req.status)
    return {"updated": True}


class SiteConfigRequest(BaseModel):
    key: str
    value: str


@app.get("/admin/config", tags=["admin"])
async def admin_get_config(key: str, _: None = Depends(_require_admin)):
    return {"key": key, "value": get_site_config(key)}


@app.post("/admin/config", tags=["admin"])
async def admin_set_config(req: SiteConfigRequest, _: None = Depends(_require_admin)):
    set_site_config(req.key, req.value)
    return {"saved": True}


# ── Returns ────────────────────────────────────────────────────────────────────

class ReturnRequest(BaseModel):
    user_id: str
    order_id: int
    item_name: str
    reason: str


@app.post("/returns", tags=["returns"])
async def submit_return(req: ReturnRequest):
    if not req.user_id.strip() or not req.reason.strip():
        raise HTTPException(status_code=422, detail="Missing required fields")
    return_id = save_return_request(req.user_id.strip(), req.order_id, req.item_name, req.reason)
    return {"submitted": True, "return_id": return_id}


@app.get("/returns/{user_id}", tags=["returns"])
async def fetch_returns(user_id: str):
    return get_return_requests(user_id.strip())


# ── Coupons ────────────────────────────────────────────────────────────────────

class CouponValidateRequest(BaseModel):
    code: str
    user_id: str = ""


@app.post("/coupons/validate", tags=["coupons"])
async def validate_coupon_endpoint(req: CouponValidateRequest):
    # Check regular coupons first
    coupon = validate_coupon(req.code)
    if coupon:
        disc = f"{coupon['discount_value']}% off" if coupon["discount_type"] == "percent" else f"₹{coupon['discount_value']} off"
        return {"valid": True, "type": "coupon", "discount_type": coupon["discount_type"],
                "discount_value": coupon["discount_value"], "message": disc}
    # Check referral codes
    ref = validate_referral_code(req.code, req.user_id)
    if ref.get("valid"):
        return {"valid": True, "type": "referral", "discount_type": "flat", "discount_value": 50,
                "referrer_user_id": ref["referrer_user_id"], "message": "₹50 off (referral gift)"}
    if ref.get("error") == "self":
        return {"valid": False, "message": "You can't use your own referral code"}
    if ref.get("error") == "already_used":
        return {"valid": False, "message": "You've already used a referral code"}
    return {"valid": False, "message": "Invalid or expired code"}


# ── Wishlist ───────────────────────────────────────────────────────────────────

class WishlistRequest(BaseModel):
    user_id: str
    product_name: str
    product_price: int = 0


@app.post("/wishlist/add", tags=["wishlist"])
async def wishlist_add(req: WishlistRequest):
    add_to_wishlist(req.user_id, req.product_name, req.product_price)
    return {"added": True}


@app.post("/wishlist/remove", tags=["wishlist"])
async def wishlist_remove(req: WishlistRequest):
    remove_from_wishlist(req.user_id, req.product_name)
    return {"removed": True}


@app.get("/wishlist/{user_id}", tags=["wishlist"])
async def wishlist_get(user_id: str):
    return get_wishlist(user_id.strip())


# ── Reviews ────────────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    user_id: str
    user_name: str
    product_name: str
    rating: int
    review_text: str = ""


@app.post("/reviews", tags=["reviews"])
async def submit_review(req: ReviewRequest):
    if not 1 <= req.rating <= 5:
        raise HTTPException(status_code=422, detail="Rating must be 1–5")
    add_review(req.user_id, req.user_name, req.product_name, req.rating, req.review_text)
    return {"submitted": True}


@app.get("/reviews/{product_name}", tags=["reviews"])
async def fetch_reviews(product_name: str):
    return get_reviews(product_name)


# ── Loyalty ────────────────────────────────────────────────────────────────────

@app.get("/loyalty/{user_id}", tags=["loyalty"])
async def fetch_loyalty(user_id: str):
    return {"points": get_loyalty_points(user_id.strip())}


# ── Chat history ───────────────────────────────────────────────────────────────

@app.get("/history/{user_id}", tags=["chat"])
async def history(user_id: str, limit: int = 20):
    return get_conversations(user_id.strip(), min(limit, 50))


@app.delete("/history/{user_id}", tags=["chat"])
async def delete_history(user_id: str):
    clear_conversations(user_id.strip())
    return {"message": "Conversation history cleared"}


# ── Referral ───────────────────────────────────────────────────────────────────

@app.get("/referral/{user_id}", tags=["referral"])
async def get_referral(user_id: str, display_name: str = ""):
    code = get_or_create_referral_code(user_id.strip(), display_name or "User")
    stats = get_referral_stats(user_id.strip())
    return {"code": code, "uses": stats["uses"]}


# ── Gift Cards ──────────────────────────────────────────────────────────────────

class GiftCardBuyRequest(BaseModel):
    user_id: str
    user_email: str = ""
    user_name: str = ""
    amount: int
    recipient_name: str
    recipient_email: str
    message: str = ""
    payment_id: str = ""


@app.post("/gift-cards/buy", tags=["gift-cards"])
async def buy_gift_card(req: GiftCardBuyRequest):
    """Create a gift card after payment"""
    if not req.user_id.strip() or not req.recipient_email.strip():
        raise HTTPException(status_code=422, detail="Missing required fields")
    if req.amount < 100:
        raise HTTPException(status_code=422, detail="Minimum gift card value is ₹100")
    
    code = create_gift_card(
        req.user_id.strip(),
        req.amount,
        req.recipient_name,
        req.recipient_email,
        req.message,
        req.payment_id,
        buyer_email=req.user_email,
        buyer_name=req.user_name,
    )
    logger.info("Gift card created: %s for %s", code, req.recipient_email)
    
    # Send email to recipient with gift card
    await _send_gift_card_email(req.recipient_email, req.recipient_name, code, req.amount, req.message)
    
    return {"code": code, "amount": req.amount, "recipient": req.recipient_email}


@app.get("/gift-cards/{user_id}", tags=["gift-cards"])
async def fetch_gift_cards(user_id: str):
    """Get gift cards bought by a user"""
    return get_gift_cards_for_user(user_id.strip())


@app.post("/gift-cards/validate", tags=["gift-cards"])
async def validate_gift_card(req: dict):
    """Validate and check balance of a gift card"""
    code = req.get("code", "").strip()
    if not code:
        return {"valid": False, "message": "No code provided"}
    
    card = get_gift_card_by_code(code)
    if not card:
        return {"valid": False, "message": "Gift card not found"}
    
    if card["balance"] <= 0:
        return {"valid": False, "message": "Gift card has no balance"}
    
    return {"valid": True, "balance": card["balance"], "code": code}


@app.post("/gift-cards/redeem", tags=["gift-cards"])
async def redeem_gift_card_endpoint(req: dict):
    """Redeem/use a gift card for an order"""
    code = req.get("code", "").strip()
    user_id = req.get("user_id", "").strip()
    amount = req.get("amount")
    
    if not code or not user_id:
        raise HTTPException(status_code=422, detail="Code and user_id required")
    
    result = redeem_gift_card(code, user_id, amount)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Redemption failed"))
    
    return result


@app.get("/admin/gift-cards", tags=["admin"])
async def admin_gift_cards(_: None = Depends(_require_admin)):
    return get_all_gift_cards()


# ── Wallet ──────────────────────────────────────────────────────────────────────

@app.get("/wallet/{user_id}", tags=["wallet"])
async def fetch_wallet(user_id: str):
    uid = user_id.strip()
    return {
        "balance": get_wallet_balance(uid),
        "transactions": get_wallet_transactions(uid),
    }


class WalletTopUpRequest(BaseModel):
    user_id: str
    amount: int
    payment_id: str = ""


@app.post("/wallet/top-up", tags=["wallet"])
async def wallet_top_up(req: WalletTopUpRequest):
    if not req.user_id.strip():
        raise HTTPException(status_code=422, detail="user_id required")
    if req.amount < 50:
        raise HTTPException(status_code=422, detail="Minimum top-up is ₹50")
    new_balance = add_wallet_credit(
        req.user_id.strip(), req.amount,
        f"Top-up via Razorpay ({req.payment_id})" if req.payment_id else "Manual top-up",
    )
    return {"success": True, "new_balance": new_balance}


class WalletUseRequest(BaseModel):
    user_id: str
    amount: int
    description: str = "Order payment"


@app.post("/wallet/use", tags=["wallet"])
async def wallet_use(req: WalletUseRequest):
    if not req.user_id.strip():
        raise HTTPException(status_code=422, detail="user_id required")
    result = deduct_wallet(req.user_id.strip(), req.amount, req.description)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed"))
    return result


class CODPayRequest(BaseModel):
    user_id: str
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


@app.post("/orders/{order_id}/pay-cod", tags=["orders"])
async def pay_cod_order(order_id: int, req: CODPayRequest):
    """Convert a COD order to a paid order after Razorpay payment."""
    secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not secret:
        raise HTTPException(status_code=503, detail="Payment not configured")
    body = f"{req.razorpay_order_id}|{req.razorpay_payment_id}"
    expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, req.razorpay_signature):
        raise HTTPException(status_code=400, detail="Payment verification failed")
    result = update_order_payment(order_id, req.user_id.strip(), req.razorpay_payment_id)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed"))
    return {"success": True, "payment_id": req.razorpay_payment_id}


# ── Contact ─────────────────────────────────────────────────────────────────────

class ContactRequest(BaseModel):
    name: str
    email: str
    message: str


@app.post("/contact", tags=["contact"])
async def contact_form(req: ContactRequest):
    if not req.name.strip() or not req.email.strip() or not req.message.strip():
        raise HTTPException(status_code=422, detail="All fields are required")
    await _send_contact_email(req.name.strip(), req.email.strip(), req.message.strip())
    return {"sent": True}


# ── Loyalty tier ───────────────────────────────────────────────────────────────

@app.get("/loyalty/{user_id}/tier", tags=["loyalty"])
async def fetch_loyalty_tier(user_id: str):
    pts = get_loyalty_points(user_id.strip())
    tier = get_tier(pts)
    return {"points": pts, "tier": tier}


# ── Package Enquiries ───────────────────────────────────────────────────────────

class PackageEnquiryRequest(BaseModel):
    user_id: str = ""
    user_name: str = ""
    user_email: str = ""
    phone: str = ""
    event_date: str = ""
    services: list
    total: int
    discount: int = 0
    message: str = ""


@app.post("/package-enquiry", tags=["packages"])
async def submit_package_enquiry(req: PackageEnquiryRequest):
    if not req.services:
        raise HTTPException(status_code=422, detail="No services selected")
    if not req.phone.strip():
        raise HTTPException(status_code=422, detail="Phone number required")
    eid = save_package_enquiry(
        req.user_id, req.user_name, req.user_email, req.phone.strip(),
        req.event_date, req.services, req.total, req.discount, req.message,
    )
    await _send_package_email(req)
    return {"submitted": True, "id": eid}


@app.get("/admin/package-enquiries", tags=["admin"])
async def admin_package_enquiries(_: None = Depends(_require_admin)):
    return get_all_package_enquiries()


class EnquiryStatusReq(BaseModel):
    status: str


@app.put("/admin/package-enquiries/{eid}", tags=["admin"])
async def update_pkg_status(eid: int, req: EnquiryStatusReq, _: None = Depends(_require_admin)):
    update_package_enquiry_status(eid, req.status)
    return {"updated": True}


# ── B2B Enquiries ───────────────────────────────────────────────────────────────

class B2BEnquiryRequest(BaseModel):
    # Field names match what the frontend sends
    company: str
    contact: str
    phone: str
    email: str = ""
    event_type: str = ""
    event_date: str = ""
    quantity: str = ""
    location: str = ""
    message: str = ""


@app.post("/b2b-enquiry", tags=["b2b"])
async def submit_b2b_enquiry(req: B2BEnquiryRequest):
    if not req.company.strip() or not req.phone.strip():
        raise HTTPException(status_code=422, detail="Company name and phone required")
    eid = save_b2b_enquiry(
        req.company.strip(), req.contact.strip(), req.phone.strip(),
        req.email.strip(), req.event_type, req.event_date,
        req.quantity, req.location.strip(), req.message.strip(),
    )
    await _send_b2b_email(req)
    return {"submitted": True, "id": eid}


@app.get("/admin/b2b-enquiries", tags=["admin"])
async def admin_b2b_enquiries(_: None = Depends(_require_admin)):
    return get_all_b2b_enquiries()


@app.put("/admin/b2b-enquiries/{eid}", tags=["admin"])
async def update_b2b_enquiry_status(eid: int, req: EnquiryStatusReq, _: None = Depends(_require_admin)):
    update_b2b_status(eid, req.status)
    return {"updated": True}


# ── Email helper ───────────────────────────────────────────────────────────────

async def _send_order_email(to_email: str, name: str, items: list, total: int, payment_id: str, address: str):
    if not RESEND_API_KEY:
        return
    try:
        import json as _json
        addr_obj = _json.loads(address) if address else {}
        addr_str = f"{addr_obj.get('line1','')}, {addr_obj.get('city','')}, {addr_obj.get('pincode','')}".strip(", ")
        rows = "".join(
            f"<tr><td style='padding:6px 12px;border-bottom:1px solid #eee'>{i['name']}</td>"
            f"<td style='padding:6px 12px;border-bottom:1px solid #eee;text-align:center'>×{i['qty']}</td>"
            f"<td style='padding:6px 12px;border-bottom:1px solid #eee;text-align:right'>₹{i['price']*i['qty']:,}</td></tr>"
            for i in items
        )
        html = f"""
        <div style="font-family:Georgia,serif;max-width:520px;margin:auto;color:#3a1f10">
          <div style="background:#3a1f10;padding:24px;text-align:center">
            <h1 style="color:#d4a84a;margin:0;font-size:22px">Mehandi Tales By Divya</h1>
            <p style="color:#d4a84a;margin:6px 0 0;font-size:13px">Order Confirmed ✅</p>
          </div>
          <div style="padding:24px;background:#fffdf8">
            <p>Hi <strong>{name or 'there'}</strong>,</p>
            <p>Thank you for your order! Divya will confirm your delivery details via WhatsApp shortly.</p>
            <table style="width:100%;border-collapse:collapse;margin:16px 0">{rows}
              <tr><td colspan="2" style="padding:10px 12px;font-weight:700">Total Paid</td>
              <td style="padding:10px 12px;text-align:right;font-weight:700;color:#bf8522">₹{total:,}</td></tr>
            </table>
            {f'<p style="color:#888;font-size:13px">📍 Delivery to: {addr_str}</p>' if addr_str else ''}
            <p style="font-size:12px;color:#bbb">Payment ID: {payment_id}</p>
          </div>
          <div style="background:#f5efe6;padding:14px;text-align:center;font-size:12px;color:#999">
            © 2025 Mehandi Tales By Divya · Chennai
          </div>
        </div>"""
        async with httpx.AsyncClient() as client:
            await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                json={
                    "from": "Mehandi Tales By Divya <onboarding@resend.dev>",
                    "to": [to_email],
                    "bcc": [ADMIN_EMAIL],
                    "subject": "Your order is confirmed — Mehandi Tales By Divya 🌿",
                    "html": html,
                },
                timeout=10,
            )
    except Exception as exc:
        logger.warning("Email send failed: %s", exc)


async def _send_status_email(to_email: str, name: str, status: str, payment_id: str):
    if not RESEND_API_KEY or not to_email:
        return
    msgs = {
        "packed":    ("📦 Order Packed!", "Your order has been carefully packed and is heading out soon."),
        "shipped":   ("🚚 Your Order is On Its Way!", "Great news — your order has been shipped and is en route to you."),
        "delivered": ("✅ Order Delivered!", "Your order has been delivered. We hope you love it! 🌿"),
    }
    title, body = msgs.get(status, ("Order Update", f"Your order status: {status}"))
    try:
        html = f"""
        <div style="font-family:Georgia,serif;max-width:520px;margin:auto;color:#3a1f10">
          <div style="background:#3a1f10;padding:24px;text-align:center">
            <h1 style="color:#d4a84a;margin:0;font-size:22px">Mehandi Tales By Divya</h1>
            <p style="color:#d4a84a;margin:6px 0 0;font-size:14px">{title}</p>
          </div>
          <div style="padding:28px;background:#fffdf8">
            <p>Hi <strong>{name or 'there'}</strong>,</p>
            <p style="font-size:15px">{body}</p>
            <p>If you have any questions, WhatsApp us at
               <a href="https://wa.me/917550084434" style="color:#bf8522">+91 75500 84434</a>.</p>
            <p style="font-size:12px;color:#bbb;margin-top:20px">Order ref: {payment_id}</p>
          </div>
          <div style="background:#f5efe6;padding:14px;text-align:center;font-size:12px;color:#999">
            © 2025 Mehandi Tales By Divya · Chennai
          </div>
        </div>"""
        async with httpx.AsyncClient() as client:
            await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                json={
                    "from": "Mehandi Tales By Divya <onboarding@resend.dev>",
                    "to": [to_email],
                    "subject": f"{title} — Mehandi Tales By Divya",
                    "html": html,
                },
                timeout=10,
            )
    except Exception as exc:
        logger.warning("Status email failed: %s", exc)


async def _send_package_email(req):
    if not RESEND_API_KEY:
        return
    try:
        rows = "".join(
            f"<tr><td style='padding:6px 10px;border-bottom:1px solid #eee'>{s['name']}</td>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #eee;text-align:center'>×{s['qty']}</td>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #eee;text-align:right'>₹{s['price']*s['qty']:,}</td></tr>"
            for s in req.services
        )
        html = f"""
        <div style="font-family:Georgia,serif;max-width:520px;margin:auto;color:#3a1f10">
          <div style="background:#3a1f10;padding:20px;text-align:center">
            <h2 style="color:#d4a84a;margin:0">💍 New Wedding Package Enquiry</h2>
          </div>
          <div style="padding:24px;background:#fffdf8">
            <p><strong>Name:</strong> {req.user_name or '—'}</p>
            <p><strong>Phone:</strong> {req.phone}</p>
            <p><strong>Email:</strong> {req.user_email or '—'}</p>
            <p><strong>Event Date:</strong> {req.event_date or '—'}</p>
            <table style="width:100%;border-collapse:collapse;margin:16px 0">{rows}
              <tr><td colspan="2" style="padding:10px;font-weight:700">Package Total</td>
              <td style="padding:10px;text-align:right;font-weight:700;color:#bf8522">₹{req.total:,}</td></tr>
              {f'<tr><td colspan="2" style="padding:4px 10px;color:#888">Discount</td><td style="padding:4px 10px;text-align:right;color:#059669">-₹{req.discount:,}</td></tr>' if req.discount else ''}
            </table>
            {f'<p><strong>Message:</strong> {req.message}</p>' if req.message else ''}
          </div>
        </div>"""
        async with httpx.AsyncClient() as client:
            await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                json={"from": "Mehandi Tales <onboarding@resend.dev>", "to": [ADMIN_EMAIL],
                      "subject": f"💍 Wedding Package Enquiry — {req.user_name or req.phone}", "html": html},
                timeout=10,
            )
    except Exception as exc:
        logger.warning("Package email failed: %s", exc)


async def _send_b2b_email(req):
    if not RESEND_API_KEY:
        return
    try:
        html = f"""
        <div style="font-family:Georgia,serif;max-width:520px;margin:auto;color:#3a1f10">
          <div style="background:#3a1f10;padding:20px;text-align:center">
            <h2 style="color:#d4a84a;margin:0">🏢 New B2B Enquiry</h2>
          </div>
          <div style="padding:24px;background:#fffdf8">
            <p><strong>Company:</strong> {req.company}</p>
            <p><strong>Contact:</strong> {req.contact}</p>
            <p><strong>Phone:</strong> {req.phone}</p>
            <p><strong>Email:</strong> {req.email or '—'}</p>
            <p><strong>Event Type:</strong> {req.event_type or '—'}</p>
            <p><strong>Event Date:</strong> {req.event_date or '—'}</p>
            <p><strong>Quantity:</strong> {req.quantity} persons</p>
            <p><strong>Location:</strong> {req.location or '—'}</p>
            {f'<hr style="border:none;border-top:1px solid #eee;margin:16px 0"><p style="white-space:pre-wrap">{req.message}</p>' if req.message else ''}
          </div>
        </div>"""
        async with httpx.AsyncClient() as client:
            await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                json={"from": "Mehandi Tales <onboarding@resend.dev>", "to": [ADMIN_EMAIL],
                      "reply_to": req.email or ADMIN_EMAIL,
                      "subject": f"🏢 B2B Enquiry — {req.company}", "html": html},
                timeout=10,
            )
    except Exception as exc:
        logger.warning("B2B email failed: %s", exc)


async def _send_contact_email(name: str, email: str, message: str):
    if not RESEND_API_KEY:
        return
    try:
        html = f"""
        <div style="font-family:Georgia,serif;max-width:520px;margin:auto;color:#3a1f10">
          <div style="background:#3a1f10;padding:20px;text-align:center">
            <h2 style="color:#d4a84a;margin:0">New Message — Mehandi Tales</h2>
          </div>
          <div style="padding:24px;background:#fffdf8">
            <p><strong>From:</strong> {name}</p>
            <p><strong>Email:</strong> <a href="mailto:{email}" style="color:#bf8522">{email}</a></p>
            <hr style="border:none;border-top:1px solid #eee;margin:16px 0">
            <p style="white-space:pre-wrap">{message}</p>
          </div>
        </div>"""
        async with httpx.AsyncClient() as client:
            await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                json={
                    "from": "Mehandi Tales Website <onboarding@resend.dev>",
                    "to": [ADMIN_EMAIL],
                    "reply_to": email,
                    "subject": f"New message from {name} — Mehandi Tales",
                    "html": html,
                },
                timeout=10,
            )
    except Exception as exc:
        logger.warning("Contact email failed: %s", exc)


async def _send_gift_card_email(recipient_email: str, recipient_name: str, code: str, amount: int, message: str):
    """Send gift card notification email to recipient"""
    if not RESEND_API_KEY:
        return
    try:
        message_html = f"<p><em>{message}</em></p>" if message else ""
        html = f"""
        <div style="font-family:Georgia,serif;max-width:580px;margin:auto;color:#3a1f10">
          <div style="background:linear-gradient(135deg,#3A1F10,#6B3820);padding:32px;text-align:center">
            <h1 style="color:#d4a84a;margin:0;font-size:28px">🎁 Gift Card Received!</h1>
            <p style="color:#d4a84a;margin:8px 0 0;font-size:14px">From Mehandi Tales By Divya</p>
          </div>
          <div style="padding:32px;background:#fffdf8;border-bottom:3px solid #bf8522">
            <p>Hi <strong>{recipient_name or 'there'}</strong>,</p>
            <p>You've been gifted a <strong style="color:#bf8522">₹{amount}</strong> gift card from Mehandi Tales By Divya!</p>
            {message_html}
            <div style="background:#f5efe6;padding:20px;border-radius:12px;margin:24px 0;text-align:center;border:2px dashed #bf8522">
              <p style="margin:0 0 12px 0;font-size:12px;color:#999">Your Gift Card Code</p>
              <p style="margin:0;font-family:monospace;font-size:24px;font-weight:700;color:#3a1f10;letter-spacing:4px">{code}</p>
            </div>
            <p style="font-size:14px;margin:16px 0">You can use this gift card to shop for organic henna cones, bridal mehendi services, and more!</p>
            <div style="text-align:center;margin:24px 0">
              <a href="https://www.mehanditalesbydivya.com" style="display:inline-block;padding:12px 32px;background:linear-gradient(135deg,#BF8522,#D4A84A);color:#fff;text-decoration:none;border-radius:50px;font-weight:700">Shop Now →</a>
            </div>
            <p style="font-size:13px;color:#999;margin:16px 0 0">Questions? WhatsApp us at <a href="https://wa.me/917550084434" style="color:#bf8522;text-decoration:none">+91 75500 84434</a></p>
          </div>
          <div style="background:#f5efe6;padding:14px;text-align:center;font-size:12px;color:#999">
            © 2025 Mehandi Tales By Divya · Chennai
          </div>
        </div>"""
        async with httpx.AsyncClient() as client:
            await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                json={
                    "from": "Mehandi Tales By Divya <onboarding@resend.dev>",
                    "to": [recipient_email],
                    "bcc": [ADMIN_EMAIL],
                    "subject": f"🎁 Your ₹{amount} Gift Card from Mehandi Tales By Divya",
                    "html": html,
                },
                timeout=10,
            )
    except Exception as exc:
        logger.warning("Gift card email failed: %s", exc)
