import csv
import hmac
import hashlib
import io
import os
import time
import logging
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
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
)
from app.config import APP_ENV, RESEND_API_KEY, ADMIN_EMAIL

try:
    import razorpay as _razorpay
except ImportError:
    _razorpay = None

logger = logging.getLogger("tdk.main")

APP_VERSION = "1.0.0"


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


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return {"status": "ok", "version": APP_VERSION, "environment": APP_ENV}


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
    key_id = os.getenv("RAZORPAY_KEY_ID", "RAZORPAY_KEY_ID_REMOVED")
    secret  = os.getenv("RAZORPAY_KEY_SECRET", "RAZORPAY_SECRET_REMOVED")
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
    secret = os.getenv("RAZORPAY_KEY_SECRET", "RAZORPAY_SECRET_REMOVED")
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


@app.put("/orders/{order_id}/status", tags=["orders"])
async def set_order_status(order_id: int, req: dict):
    if req.get("admin_email") != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Forbidden")
    status = req.get("status", "")
    if status not in ("confirmed", "packed", "shipped", "delivered"):
        raise HTTPException(status_code=422, detail="Invalid status")
    order = update_order_status(order_id, status)
    if status in ("packed", "shipped", "delivered") and order.get("user_email"):
        await _send_status_email(
            order["user_email"], order.get("user_name", ""), status, order.get("payment_id", "")
        )
    return {"updated": True}


# ── Admin ──────────────────────────────────────────────────────────────────────

def _check_admin(email: str):
    if email != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.get("/admin/orders", tags=["admin"])
async def admin_orders(admin_email: str = ""):
    _check_admin(admin_email)
    return get_all_orders()


@app.get("/admin/coupons", tags=["admin"])
async def admin_coupons(admin_email: str = ""):
    _check_admin(admin_email)
    return list_coupons()


class CouponCreateRequest(BaseModel):
    code: str
    discount_type: str
    discount_value: int
    max_usage: int = 0
    admin_email: str = ""


@app.post("/admin/coupons", tags=["admin"])
async def admin_create_coupon(req: CouponCreateRequest):
    _check_admin(req.admin_email)
    if req.discount_type not in ("percent", "flat"):
        raise HTTPException(status_code=422, detail="discount_type must be 'percent' or 'flat'")
    create_coupon(req.code, req.discount_type, req.discount_value, req.max_usage)
    return {"created": True}


@app.get("/admin/analytics", tags=["admin"])
async def admin_analytics(admin_email: str = ""):
    _check_admin(admin_email)
    return get_analytics()


@app.get("/admin/customers", tags=["admin"])
async def admin_customers(admin_email: str = ""):
    _check_admin(admin_email)
    return get_customers()


@app.get("/admin/export/orders", tags=["admin"])
async def admin_export_orders(admin_email: str = ""):
    _check_admin(admin_email)
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
async def admin_returns(admin_email: str = ""):
    _check_admin(admin_email)
    return get_all_returns()


class ReturnStatusRequest(BaseModel):
    status: str
    admin_email: str = ""


@app.put("/admin/returns/{return_id}", tags=["admin"])
async def admin_update_return(return_id: int, req: ReturnStatusRequest):
    _check_admin(req.admin_email)
    if req.status not in ("pending", "approved", "rejected", "completed"):
        raise HTTPException(status_code=422, detail="Invalid status")
    update_return_status(return_id, req.status)
    return {"updated": True}


class SiteConfigRequest(BaseModel):
    key: str
    value: str
    admin_email: str = ""


@app.get("/admin/config", tags=["admin"])
async def admin_get_config(key: str, admin_email: str = ""):
    _check_admin(admin_email)
    return {"key": key, "value": get_site_config(key)}


@app.post("/admin/config", tags=["admin"])
async def admin_set_config(req: SiteConfigRequest):
    _check_admin(req.admin_email)
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
