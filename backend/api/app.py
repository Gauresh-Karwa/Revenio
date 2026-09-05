"""Revenio's HTTP and WebSocket API.

The API serves both the React dashboard and Razorpay's server-to-server
webhooks.  It does not expose database credentials or allow the browser to
call a payment provider directly.
"""

from __future__ import annotations

import base64
import asyncio
import hashlib
import hmac
import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import load_environment
from backend.api.runtime import DemoRuntime
from backend.integrations.channels import fetch_received_email
from backend.integrations.razorpay_gateway import RazorpayGateway

load_environment()
app = FastAPI(title="Revenio API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("REVENIO_CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
runtime = DemoRuntime()


class SimulationRequest(BaseModel):
    scenario: str = Field(default="random", pattern="^(random|payment_failure|checkout_abandonment|overdue_invoice|mandate_failure)$")
    count: int = Field(default=1, ge=1, le=100)


class CustomSimulationRequest(BaseModel):
    domain_type: str = Field(pattern="^(subscription|checkout_abandonment|b2b_receivables|mandate_retry)$")
    customer_name: str = Field(min_length=2, max_length=80)
    customer_id: str | None = Field(default=None, max_length=120)
    customer_email: str | None = Field(default=None, max_length=254)
    customer_phone: str | None = Field(default=None, max_length=32)
    amount: float = Field(gt=0, le=1_000_000)
    failure_code: str | None = Field(default=None, max_length=80)
    response: str = Field(pattern="^(recovered|lost|paid|promise|no_response|needs_human|hardship)$")
    opt_in: bool = True
    days_overdue: int = Field(default=30, ge=1, le=365)
    await_razorpay_confirmation: bool = True


class ReviewRequest(BaseModel):
    approved: bool


class EmailTestRequest(BaseModel):
    recipient: str = Field(min_length=3, max_length=254)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "interactive-demo"}


@app.get("/api/integrations/status")
def integration_status() -> dict[str, Any]:
    """Expose readiness only—never credentials—to make delivery states clear."""
    def configured(name: str) -> bool:
        value = os.environ.get(name, "")
        return bool(value) and "replace_me" not in value and "your-" not in value

    channel_mode = os.environ.get("REVENIO_CHANNEL_MODE", "sandbox").lower()
    email_ready = configured("RESEND_API_KEY") and configured("REVENIO_EMAIL_FROM")
    phone_ready = all(configured(name) for name in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER"))
    razorpay_ready = all(configured(name) for name in ("RAZORPAY_KEY_ID", "RAZORPAY_KEY_SECRET", "RAZORPAY_WEBHOOK_SECRET"))
    return {
        "channel_mode": channel_mode,
        "live_delivery_acknowledged": os.environ.get("REVENIO_LIVE_DELIVERY_ACK") == "I_HAVE_CONSENT",
        "email": {"ready": email_ready, "sender": os.environ.get("REVENIO_EMAIL_FROM", "") if email_ready else None},
        "sms_voice": {"ready": phone_ready},
        "razorpay": {"ready": razorpay_ready, "mode": "test" if os.environ.get("RAZORPAY_KEY_ID", "").startswith("rzp_test_") else "live" if razorpay_ready else None},
    }


@app.post("/api/integrations/email-test")
def email_test(request: EmailTestRequest) -> dict[str, Any]:
    """Submit one operator-requested email and expose Resend's actual receipt."""
    if "@" not in request.recipient:
        raise HTTPException(status_code=422, detail="Enter a valid email address")
    return runtime.send_email_test(request.recipient)


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    return runtime.dashboard()


@app.get("/api/cases")
def cases(domain_type: str | None = None, status: str | None = None) -> dict[str, Any]:
    return {"items": runtime.list_cases(domain_type=domain_type, status=status)}


@app.get("/api/cases/{case_id}")
def case_detail(case_id: str) -> dict[str, Any]:
    try:
        return runtime.case_detail(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Case not found") from exc


@app.post("/api/simulations")
def simulate(request: SimulationRequest) -> dict[str, Any]:
    return {"items": runtime.simulate(request.scenario, request.count)}


@app.post("/api/simulations/custom")
def simulate_custom(request: CustomSimulationRequest) -> dict[str, Any]:
    try:
        return runtime.simulate_custom(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/reviews")
def reviews() -> dict[str, Any]:
    return {"items": runtime.reviews()}


@app.post("/api/reviews/{case_id}")
def resolve_review(case_id: str, request: ReviewRequest) -> dict[str, Any]:
    try:
        return runtime.resolve_review(case_id, request.approved)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Review case not found") from exc


@app.post("/api/cases/{case_id}/payment-link")
def create_payment_link(case_id: str) -> dict[str, Any]:
    """Create a real Razorpay test/live payment link only after an explicit UI action."""
    try:
        detail = runtime.case_detail(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Case not found") from exc
    case = detail["case"]
    amount = float(case.get("amount", case.get("invoice_amount", 0)))
    if amount <= 0:
        raise HTTPException(status_code=422, detail="Case has no recoverable payment amount")
    try:
        link = RazorpayGateway.from_environment().create_recovery_payment_link(
            case_id=case_id, amount_inr=amount, customer_id=case.get("customer_id")
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    link["_revenio_mode"] = "test" if os.environ.get("RAZORPAY_KEY_ID", "").startswith("rzp_test_") else "live"
    recorded = runtime.record_payment_link(case_id, link)
    return {
        "id": link.get("id"),
        "short_url": link.get("short_url"),
        "status": link.get("status"),
        "case": recorded,
    }


@app.websocket("/ws/events")
async def event_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = runtime.subscribe()
    try:
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        runtime.unsubscribe(queue)


def _verify_razorpay_webhook(raw_body: bytes, received_signature: str | None) -> None:
    if not received_signature:
        raise HTTPException(status_code=401, detail="Missing Razorpay signature")
    try:
        RazorpayGateway.from_environment().verify_webhook(raw_body, received_signature)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid Razorpay signature") from exc


def _verify_resend_webhook(request: Request, raw_body: bytes) -> None:
    """Verify Resend's Svix signature against its endpoint-specific secret.

    Verification uses the untouched HTTP body.  This prevents an arbitrary
    caller from creating communication or human-review events in a case.
    """
    secret = os.environ.get("RESEND_WEBHOOK_SECRET", "")
    message_id = request.headers.get("svix-id")
    timestamp = request.headers.get("svix-timestamp")
    signatures = request.headers.get("svix-signature", "")
    if not all((secret, message_id, timestamp, signatures)):
        raise HTTPException(status_code=401, detail="Missing Resend webhook verification data")
    try:
        timestamp_value = int(timestamp)
        if abs(time.time() - timestamp_value) > 300:
            raise ValueError("Webhook timestamp is outside the allowed window")
        encoded_secret = secret.removeprefix("whsec_")
        padded_secret = encoded_secret + "=" * (-len(encoded_secret) % 4)
        # Svix/Resend secrets are URL-safe base64 and may contain '-' or '_'.
        key = base64.urlsafe_b64decode(padded_secret)
        signed = b".".join((message_id.encode(), timestamp.encode(), raw_body))
        expected = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode()
        supplied = [part.split(",", 1)[1].strip() for part in signatures.split(" ") if part.startswith("v1,")]
        if not any(hmac.compare_digest(expected, candidate) for candidate in supplied):
            raise ValueError("Signature mismatch")
    except (ValueError, TypeError, base64.binascii.Error) as exc:
        raise HTTPException(status_code=401, detail="Invalid Resend webhook signature") from exc


def _case_id_from_resend_recipients(recipients: object) -> str | None:
    if not isinstance(recipients, list):
        return None
    for recipient in recipients:
        local = str(recipient).split("@", 1)[0].lower()
        if local.startswith("case-"):
            requested = local.removeprefix("case-")
            for case_id in runtime._cases:  # local runtime lookup; no user data is guessed
                if case_id.lower() == requested:
                    return case_id
    return None


async def _process_resend_reply(case_id: str, sender: str, email_id: str) -> None:
    """Fetch the potentially slow inbound body after the webhook is acknowledged."""
    last_error: Exception | None = None
    for delay in (0, 1, 3):
        if delay:
            await asyncio.sleep(delay)
        try:
            # urllib is synchronous.  Keep it off FastAPI's event loop so a
            # slow provider retrieval cannot delay the 202 webhook response
            # or make the dashboard appear offline.
            received = await asyncio.to_thread(fetch_received_email, email_id)
            content = received.get("data", received)
            text = str(content.get("text") or content.get("html") or "").strip()
            if text:
                runtime.record_email_reply(case_id, sender, text, email_id)
            else:
                state = runtime.store.derive_state(case_id)
                runtime.store.append(
                    case_id,
                    state["domain_type"],
                    "customer_reply",
                    "InboundEmailProcessingFailed",
                    {"email_id": email_id, "message": "Resend returned an inbound email with no readable text."},
                    customer_id=runtime._cases[case_id].get("customer_id"),
                )
            return
        except (HTTPError, URLError, RuntimeError, ValueError) as exc:
            last_error = exc
    # The reply remains stored in Resend, and this diagnostic makes a missing
    # Full-access API permission visible in the local server log.
    print(f"Resend inbound email {email_id} could not be retrieved: {last_error}")
    state = runtime.store.derive_state(case_id)
    runtime.store.append(
        case_id,
        state["domain_type"],
        "customer_reply",
        "InboundEmailProcessingFailed",
        {"email_id": email_id, "message": f"Could not retrieve reply text from Resend: {last_error}"},
        customer_id=runtime._cases[case_id].get("customer_id"),
    )


@app.post("/webhooks/resend", status_code=202)
async def resend_webhook(request: Request, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Acknowledge a signed inbound reply immediately, then retrieve its text."""
    raw_body = await request.body()
    _verify_resend_webhook(request, raw_body)
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid Resend webhook JSON") from exc
    if payload.get("type") != "email.received":
        return {"status": "ignored"}
    data = payload.get("data", {})
    case_id = _case_id_from_resend_recipients(data.get("to"))
    email_id = data.get("email_id")
    if not case_id or not email_id:
        return {"status": "unmatched"}
    background_tasks.add_task(
        _process_resend_reply,
        case_id,
        str(data.get("from", "unknown sender")),
        str(email_id),
    )
    return {"status": "accepted"}


@app.post("/webhooks/razorpay", status_code=202)
async def razorpay_webhook(request: Request) -> dict[str, str]:
    """Verify first, then hand a payment failure to the asynchronous pipeline.

    This endpoint intentionally refuses unverified browser-originated data.
    The next integration pass persists an outbox row before publishing the
    task; this route already provides the critical trust boundary and maps
    the external event into a domain-neutral case payload.
    """
    raw_body = await request.body()
    _verify_razorpay_webhook(raw_body, request.headers.get("x-razorpay-signature"))
    event_id = request.headers.get("x-razorpay-event-id")
    if not event_id:
        raise HTTPException(status_code=400, detail="Missing Razorpay event id")
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    entities = payload.get("payload", {})
    payment = entities.get("payment", {}).get("entity", {})
    payment_link = entities.get("payment_link", {}).get("entity", {})
    order = entities.get("order", {}).get("entity", {})
    payment_id = payment.get("id")
    case_id = (
        payment.get("notes", {}).get("revenio_case_id")
        or payment_link.get("reference_id")
        or payment_link.get("notes", {}).get("revenio_case_id")
        or order.get("notes", {}).get("revenio_case_id")
    )
    if case_id:
        try:
            runtime.record_razorpay_payment_event(
                case_id, payload.get("event", "unknown"), payment or payment_link
            )
            return {"status": "recorded"}
        except KeyError:
            pass

    if payload.get("event") != "payment.failed":
        return {"status": "ignored"}
    if not payment_id:
        raise HTTPException(status_code=400, detail="Payment id is missing")

    # Razorpay's error codes are preserved for the audit record.  If they do
    # not map to the ISO taxonomy, the existing module safely routes the case
    # to human review instead of inventing a decline reason.
    case = {
        "customer_id": payment.get("email") or payment.get("contact") or payment_id,
        "decline_code": str(payment.get("error_code") or "UNKNOWN"),
        "amount": float(payment.get("amount", 0)) / 100,
        "attempt_number": 1,
        "razorpay_payment_id": payment_id,
        "razorpay_order_id": payment.get("order_id"),
        "razorpay_event_id": event_id,
    }
    # The production consumer is deliberately asynchronous.  We import here
    # so the dashboard's local demo remains runnable without Redis/Celery.
    from backend.queue.webhook_ingestion import enqueue_webhook_event

    queued = enqueue_webhook_event(event_id, f"razorpay-{payment_id}", "subscription", case)
    return {"status": "queued" if queued else "duplicate"}
