"""Delivery adapters for recovery channels.

The project deliberately keeps SMS and voice delivery behind this small
interface: Razorpay is the payments integration, not an SMS or telephony
provider.  The demo adapter below is dynamic and records a realistic receipt;
it does not send a message or place a phone call.  A live provider must be
configured with approved credentials and consented recipient data.
"""

from __future__ import annotations

import base64
import json
import os
import random
from html import escape
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class DemoChannelGateway:
    """Produces auditable, non-delivery receipts for the interactive simulator.

    This adapter is intentionally a sandbox: it never sends an external SMS,
    email or voice call.  The returned receipt makes that visible to the UI
    while exercising the exact module execution boundary a live provider
    implementation would use.
    """

    _SMS_RESPONSES = (
        "Customer replied: payment will be cleared today.",
        "Customer replied: please share a payment link by SMS.",
        "No SMS reply received within the demonstration window.",
    )
    _VOICE_RESPONSES = (
        "Hinglish call response: Haan, invoice mil gaya. Main aaj payment kar dunga.",
        "Hinglish call response: Abhi cash-flow issue hai, kal tak payment link bhej dijiye.",
        "Hinglish call response: Customer requested a human collections specialist.",
    )

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def dispatch(self, case: dict[str, Any], decision: Any) -> dict[str, Any]:
        channel = decision.action_params.get("channel") or self._channel_for(decision)
        selected = case.get("simulated_contact_response")
        response, effect = self._response(selected, channel)
        return {
            "provider": self._provider_for(channel),
            "mode": "sandbox_simulation",
            "channel": channel,
            "locale": decision.action_params.get("locale"),
            "recipient": case.get("customer_id"),
            "message": self._message_for(channel, case),
            "response": response,
            "effect": effect,
            "delivered_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _channel_for(decision: Any) -> str:
        action = getattr(decision.action_type, "value", str(decision.action_type))
        return "payment_retry" if action == "RETRY" else "email"

    @staticmethod
    def _provider_for(channel: str) -> str:
        return {
            "email": "Revenio Email Sandbox",
            "sms": "Revenio SMS Sandbox",
            "voice": "Revenio Hinglish Voice Sandbox",
            "payment_retry": "Razorpay test-mode recovery simulator",
        }.get(channel, "Revenio channel sandbox")

    @staticmethod
    def _message_for(channel: str, case: dict[str, Any]) -> str:
        amount = case.get("invoice_amount", case.get("amount"))
        if channel == "voice":
            return "Namaste, payment reminder ke liye call kiya hai. Kya aap aaj payment complete kar sakte hain?"
        if channel == "sms":
            return f"Payment reminder: INR {amount}. Reply for a secure payment link."
        if channel == "payment_retry":
            return "A compliant payment retry was scheduled through the configured payment rail."
        return f"Invoice/payment reminder for INR {amount}; a secure payment option is available."

    def _response(self, selected: str | None, channel: str) -> tuple[str, str]:
        selected_responses = {
            "paid": ("Customer confirmed that payment was completed.", "payment_recovered"),
            "recovered": ("Customer completed payment after the recovery action.", "payment_recovered"),
            "promise": ("Customer promised to pay by the agreed date.", "promise_to_pay"),
            "no_response": ("No customer response received in the simulation window.", "await_next_step"),
            "lost": ("Customer declined or did not complete the payment.", "recovery_not_completed"),
            "needs_human": ("Customer requested a human collections specialist.", "human_review_required"),
            "hardship": ("Customer reported a financial hardship and requested human help.", "human_review_required"),
        }
        if selected in selected_responses:
            return selected_responses[selected]
        if channel == "sms":
            return self._rng.choice(self._SMS_RESPONSES), "await_next_step"
        if channel == "voice":
            return self._rng.choice(self._VOICE_RESPONSES), "await_next_step"
        return "Delivery recorded in the sandbox; awaiting customer response.", "await_next_step"


class LiveChannelGateway:
    """Opt-in production adapter: Resend email and Twilio SMS/voice REST APIs.

    It sends only when `REVENIO_CHANNEL_MODE=live` and the explicit consent
    acknowledgement is set.  Missing recipient or provider configuration is
    returned as an auditable blocked receipt, never silently simulated.
    """

    def dispatch(self, case: dict[str, Any], decision: Any) -> dict[str, Any]:
        channel = decision.action_params.get("channel") or DemoChannelGateway._channel_for(decision)
        message = DemoChannelGateway._message_for(channel, case)
        receipt = {
            "provider": "unconfigured live channel",
            "mode": "live",
            "channel": channel,
            "locale": decision.action_params.get("locale"),
            "recipient": None,
            "message": message,
            "response": "No customer response available yet.",
            "effect": "delivery_blocked",
            "delivered_at": datetime.now(timezone.utc).isoformat(),
        }
        if os.environ.get("REVENIO_LIVE_DELIVERY_ACK") != "I_HAVE_CONSENT":
            receipt["response"] = "Live delivery blocked until consent acknowledgement is configured."
            return receipt
        try:
            if channel == "email":
                return self._email(case, receipt)
            if channel in {"sms", "voice"}:
                return self._twilio(case, receipt)
            receipt["response"] = "Use the explicit Razorpay payment-link action for payment collection."
            return receipt
        except (HTTPError, URLError, ValueError) as exc:
            receipt["response"] = f"Provider rejected delivery: {exc}"
            return receipt

    def send_test_email(self, recipient: str) -> dict[str, Any]:
        """Send an explicitly requested provider-connection email.

        This is deliberately separate from recovery execution: it lets an
        operator validate Resend before a customer case is contacted.  The
        browser receives the provider result, never credentials.
        """
        receipt: dict[str, Any] = {
            "provider": "Resend",
            "mode": "live",
            "channel": "email",
            "recipient": recipient,
            "message": "Revenio email connection check. This message confirms that your recovery sender can submit email through Resend.",
            "response": "",
            "effect": "delivery_blocked",
            "delivered_at": datetime.now(timezone.utc).isoformat(),
        }
        if os.environ.get("REVENIO_LIVE_DELIVERY_ACK") != "I_HAVE_CONSENT":
            receipt["response"] = "Live delivery blocked until consent acknowledgement is configured."
            return receipt
        try:
            return self._email({"customer_email": recipient}, receipt)
        except (HTTPError, URLError, ValueError) as exc:
            receipt["response"] = f"Provider rejected delivery: {exc}"
            return receipt

    @staticmethod
    def _email(case: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
        recipient = case.get("customer_email")
        key, sender = os.environ.get("RESEND_API_KEY"), os.environ.get("REVENIO_EMAIL_FROM")
        if not (recipient and key and sender):
            receipt["response"] = "Email blocked: customer email, RESEND_API_KEY, or REVENIO_EMAIL_FROM is missing."
            return receipt
        body_payload: dict[str, Any] = {"from": sender, "to": [recipient], "subject": "Payment reminder from Revenio", "text": receipt["message"]}
        reply_to = _case_reply_address(case.get("recovery_case_id"))
        if reply_to:
            body_payload["reply_to"] = [reply_to]
            receipt["reply_to"] = reply_to
        body = json.dumps(body_payload).encode()
        result = _post_json(
            "https://api.resend.com/emails",
            body,
            {
                "Authorization": f"Bearer {key}",
                # Resend rejects direct HTTP clients without this header with
                # 403 / error 1010 before it evaluates the sender or key.
                "User-Agent": "revenio-recovery/0.1",
            },
        )
        receipt.update({"provider": "Resend", "recipient": recipient, "provider_message_id": result.get("id"), "response": "Email accepted by provider.", "effect": "delivery_submitted"})
        return receipt

    @staticmethod
    def _twilio(case: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
        recipient = case.get("customer_phone")
        sid, token, sender = os.environ.get("TWILIO_ACCOUNT_SID"), os.environ.get("TWILIO_AUTH_TOKEN"), os.environ.get("TWILIO_FROM_NUMBER")
        if not (recipient and sid and token and sender):
            receipt["response"] = "Phone delivery blocked: customer phone or Twilio configuration is missing."
            return receipt
        payload: dict[str, str] = {"To": recipient, "From": sender}
        if receipt["channel"] == "sms":
            payload["Body"] = str(receipt["message"])
            resource, provider = "Messages", "Twilio SMS"
        else:
            payload["Twiml"] = f"<Response><Say voice=\"alice\">{escape(str(receipt['message']))}</Say></Response>"
            resource, provider = "Calls", "Twilio Voice"
        auth = base64.b64encode(f"{sid}:{token}".encode()).decode()
        result = _post_json(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/{resource}.json", urlencode(payload).encode(), {"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"})
        receipt.update({"provider": provider, "recipient": recipient, "provider_message_id": result.get("sid"), "response": "Phone delivery accepted by provider.", "effect": "delivery_submitted"})
        return receipt


def _post_json(url: str, body: bytes, headers: dict[str, str]) -> dict[str, Any]:
    request = Request(url, data=body, headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urlopen(request, timeout=12) as response:  # nosec B310 - URL is provider constant
            return json.loads(response.read().decode())
    except HTTPError as exc:
        # Providers place the actionable cause (for example, an unverified
        # Resend sender) in the JSON response body.  Preserve that concise
        # diagnostic for the operator, without ever returning authentication.
        raw = exc.read().decode(errors="replace")[:800]
        try:
            parsed = json.loads(raw)
            message = parsed.get("message") or parsed.get("error") or raw
        except json.JSONDecodeError:
            message = raw
        raise ValueError(f"HTTP {exc.code}: {message}") from exc


def _case_reply_address(case_id: Any) -> str | None:
    """Route each reply to a case-specific address on a Resend inbound domain."""
    base = os.environ.get("REVENIO_EMAIL_REPLY_TO", "").strip()
    if not base or "@" not in base or not case_id:
        return None
    _, domain = base.rsplit("@", 1)
    safe_case = "".join(char.lower() for char in str(case_id) if char.isalnum() or char == "-")
    return f"case-{safe_case}@{domain}" if safe_case else None


def fetch_received_email(email_id: str) -> dict[str, Any]:
    """Retrieve inbound text after Resend sends a signed metadata webhook."""
    key = os.environ.get("RESEND_API_KEY")
    if not key:
        raise RuntimeError("RESEND_API_KEY is missing")
    request = Request(
        f"https://api.resend.com/emails/receiving/{email_id}",
        headers={"Authorization": f"Bearer {key}", "User-Agent": "revenio-recovery/0.1"},
    )
    with urlopen(request, timeout=12) as response:  # nosec B310 - provider URL
        return json.loads(response.read().decode())


class ConfiguredChannelGateway:
    """Select live delivery only for an operator-entered, addressable case.

    Portfolio runs deliberately contain no contact details, so they retain
    sandbox receipts even when a developer has enabled live delivery locally.
    This prevents a bulk click from becoming an outbound messaging action.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._sandbox = DemoChannelGateway(rng)
        self._live = LiveChannelGateway()

    def dispatch(self, case: dict[str, Any], decision: Any) -> dict[str, Any]:
        can_contact = bool(case.get("customer_email") or case.get("customer_phone"))
        if os.environ.get("REVENIO_CHANNEL_MODE", "sandbox").lower() == "live" and can_contact:
            return self._live.dispatch(case, decision)
        return self._sandbox.dispatch(case, decision)

    def send_test_email(self, recipient: str) -> dict[str, Any]:
        if os.environ.get("REVENIO_CHANNEL_MODE", "sandbox").lower() != "live":
            return {
                "provider": "Resend",
                "mode": "sandbox_simulation",
                "channel": "email",
                "recipient": recipient,
                "message": "Revenio email connection check.",
                "response": "Email check blocked: set REVENIO_CHANNEL_MODE=live before sending.",
                "effect": "delivery_blocked",
                "delivered_at": datetime.now(timezone.utc).isoformat(),
            }
        return self._live.send_test_email(recipient)


def configured_channel_gateway(rng: random.Random | None = None) -> ConfiguredChannelGateway:
    return ConfiguredChannelGateway(rng)
