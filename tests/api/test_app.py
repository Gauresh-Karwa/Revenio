import base64
import hashlib
import hmac
import json
import time

from fastapi.testclient import TestClient

from backend.api.app import app, runtime


def test_simulation_populates_merchant_and_audit_api():
    client = TestClient(app)

    created = client.post("/api/simulations", json={"scenario": "payment_failure", "count": 1})
    assert created.status_code == 200

    cases = client.get("/api/cases")
    assert cases.status_code == 200
    case_id = cases.json()["items"][0]["case_id"]

    detail = client.get(f"/api/cases/{case_id}")
    assert detail.status_code == 200
    assert any(event["event_type"] == "Diagnosis" for event in detail.json()["events"])


def test_custom_invoice_simulation_records_channel_receipts_and_review():
    client = TestClient(app)
    response = client.post(
        "/api/simulations/custom",
        json={
            "domain_type": "b2b_receivables",
            "customer_name": "Aarav Sharma",
            "customer_id": "merchant-customer-test",
            "amount": 45000,
            "response": "no_response",
            "opt_in": True,
            "days_overdue": 30,
        },
    )

    assert response.status_code == 200
    events = response.json()["events"]
    receipts = [event["payload"]["details"] for event in events if event["event_type"] == "ExecutionResult"]
    assert [receipt["channel"] for receipt in receipts] == ["email", "sms"]
    assert all(receipt["mode"] == "sandbox_simulation" for receipt in receipts)
    assert any(event["event_type"] == "PendingHumanReview" for event in events)

    approved = client.post(f"/api/reviews/{response.json()['state']['case_id']}", json={"approved": True})
    assert approved.status_code == 200
    voice_receipts = [
        event["payload"]["details"]
        for event in approved.json()["events"]
        if event["event_type"] == "ExecutionResult" and event["payload"].get("details", {}).get("channel") == "voice"
    ]
    assert len(voice_receipts) == 1
    assert voice_receipts[0]["provider"] == "Revenio Hinglish Voice Sandbox"


def test_custom_hardship_response_routes_payment_to_human_review():
    client = TestClient(app)
    response = client.post(
        "/api/simulations/custom",
        json={
            "domain_type": "subscription",
            "customer_name": "Aarav Sharma",
            "amount": 1499,
            "failure_code": "51",
            "response": "hardship",
            "opt_in": True,
            "days_overdue": 30,
        },
    )
    assert response.status_code == 200
    assert any(event["event_type"] == "PendingHumanReview" for event in response.json()["events"])


def test_operator_case_never_books_revenue_before_razorpay_confirmation():
    client = TestClient(app)
    response = client.post(
        "/api/simulations/custom",
        json={
            "domain_type": "b2b_receivables",
            "customer_name": "Aarav Sharma",
            "amount": 1499,
            "response": "paid",
            "opt_in": True,
            "days_overdue": 30,
        },
    )
    assert response.status_code == 200
    outcomes = [event["payload"] for event in response.json()["events"] if event["event_type"] == "Outcome"]
    assert outcomes[-1] == {"status": "PENDING", "details": {}, "amount_recovered": 0.0}


def test_email_connection_check_returns_the_provider_receipt(monkeypatch):
    expected = {
        "provider": "Resend",
        "mode": "live",
        "channel": "email",
        "recipient": "owner@example.com",
        "response": "Email accepted by provider.",
        "effect": "delivery_submitted",
    }
    monkeypatch.setattr(runtime, "send_email_test", lambda recipient: expected)

    response = TestClient(app).post(
        "/api/integrations/email-test", json={"recipient": "owner@example.com"}
    )

    assert response.status_code == 200
    assert response.json() == expected


def test_signed_resend_reply_is_recorded_as_a_signal_not_revenue(monkeypatch):
    client = TestClient(app)
    created = client.post(
        "/api/simulations/custom",
        json={
            "domain_type": "subscription",
            "customer_name": "Nila Kapoor",
            "amount": 1499,
            "failure_code": "51",
            "response": "no_response",
            "opt_in": True,
            "days_overdue": 30,
        },
    ).json()
    case_id = created["state"]["case_id"]
    secret_bytes = b"resend-test-signing-secret-123456"
    secret = "whsec_" + base64.b64encode(secret_bytes).decode()
    message_id, timestamp = "msg_reply_test", str(int(time.time()))
    payload = {
        "type": "email.received",
        "data": {
            "email_id": "received_test_123",
            "from": "customer@example.com",
            "to": [f"case-{case_id.lower()}@team.resend.app"],
        },
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    signature = base64.b64encode(
        hmac.new(secret_bytes, b".".join((message_id.encode(), timestamp.encode(), raw)), hashlib.sha256).digest()
    ).decode()
    monkeypatch.setenv("RESEND_WEBHOOK_SECRET", secret)
    monkeypatch.setattr("backend.api.app.fetch_received_email", lambda _: {"text": "Please send a payment link."})

    response = client.post(
        "/webhooks/resend",
        content=raw,
        headers={"svix-id": message_id, "svix-timestamp": timestamp, "svix-signature": f"v1,{signature}"},
    )

    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}
    detail = client.get(f"/api/cases/{case_id}").json()
    reply = next(event for event in detail["events"] if event["event_type"] == "CustomerEmailReply")
    assert reply["payload"]["intent"] == "payment_link_requested"
    assert not any(event["event_type"] == "Outcome" and event["payload"].get("status") == "RECOVERED" for event in detail["events"])


def test_razorpay_webhook_rejects_an_invalid_signature(monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_placeholder")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "placeholder-secret")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "test-secret")
    client = TestClient(app)

    response = client.post(
        "/webhooks/razorpay",
        content=b'{"event":"payment.failed"}',
        headers={"x-razorpay-event-id": "evt_test", "x-razorpay-signature": "bad"},
    )

    assert response.status_code == 401
