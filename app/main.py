import hmac
import hashlib
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.models import QueryRequest, QueryResponse, HealthResponse
from app.graph import graph
from app.memory import init_db, save_conversation, get_conversations, clear_conversations
from app.config import APP_ENV, RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET

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
    logger.info(
        "Done in %.2fs — score=%d iterations=%d",
        elapsed, result["score"], result["iteration"],
    )

    save_conversation(req.user_id, req.query, result["draft"], result["score"], result["iteration"])

    return {"answer": result["draft"], "score": result["score"], "iterations": result["iteration"]}


class OrderRequest(BaseModel):
    amount: int        # in paise (INR × 100)
    currency: str = "INR"
    receipt: str = "order_receipt"

class VerifyRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


@app.get("/payment-status", tags=["payment"])
async def payment_status():
    return {
        "key_id_set": bool(RAZORPAY_KEY_ID),
        "key_id_prefix": RAZORPAY_KEY_ID[:8] if RAZORPAY_KEY_ID else "NOT SET",
        "secret_set": bool(RAZORPAY_KEY_SECRET),
        "razorpay_pkg": _razorpay is not None,
    }


@app.post("/create-order", tags=["payment"])
async def create_order(req: OrderRequest):
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=503, detail="Payment not configured")
    if _razorpay is None:
        raise HTTPException(status_code=503, detail="Razorpay package not available")
    try:
        client = _razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        order = client.order.create({
            "amount": req.amount,
            "currency": req.currency,
            "receipt": req.receipt,
            "payment_capture": 1,
        })
        return {"order_id": order["id"], "key_id": RAZORPAY_KEY_ID, "amount": req.amount}
    except Exception as exc:
        logger.error("Razorpay order creation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/verify-payment", tags=["payment"])
async def verify_payment(req: VerifyRequest):
    if not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=503, detail="Payment not configured")
    body = f"{req.razorpay_order_id}|{req.razorpay_payment_id}"
    expected = hmac.new(
        RAZORPAY_KEY_SECRET.encode(),
        body.encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, req.razorpay_signature):
        raise HTTPException(status_code=400, detail="Payment verification failed")
    logger.info("Payment verified: %s", req.razorpay_payment_id)
    return {"success": True, "payment_id": req.razorpay_payment_id}


@app.get("/history/{user_id}", tags=["chat"])
async def history(user_id: str, limit: int = 20):
    return get_conversations(user_id.strip(), min(limit, 50))


@app.delete("/history/{user_id}", tags=["chat"])
async def delete_history(user_id: str):
    clear_conversations(user_id.strip())
    return {"message": "Conversation history cleared"}
