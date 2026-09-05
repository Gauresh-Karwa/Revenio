"""Server-side Razorpay SDK boundary.

Only FastAPI and background workers may construct this client.  Browser code
receives neither the API secret nor a way to call Razorpay's API directly.
"""

from __future__ import annotations

import os
from typing import Any


class RazorpayGateway:
    def __init__(self, key_id: str, key_secret: str, webhook_secret: str) -> None:
        import razorpay

        self._client = razorpay.Client(auth=(key_id, key_secret))
        self._webhook_secret = webhook_secret

    @classmethod
    def from_environment(cls) -> "RazorpayGateway":
        key_id = os.environ.get("RAZORPAY_KEY_ID")
        key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
        webhook_secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET")
        if not all((key_id, key_secret, webhook_secret)):
            raise RuntimeError("Razorpay test-mode credentials are not fully configured")
        return cls(key_id, key_secret, webhook_secret)

    def verify_webhook(self, raw_body: bytes, signature: str) -> None:
        # Razorpay's SDK performs the raw-body HMAC verification documented
        # for X-Razorpay-Signature.  Decoding preserves the original bytes
        # exactly for UTF-8 JSON webhook payloads; we never parse/re-serialize
        # the body before verification.
        self._client.utility.verify_webhook_signature(
            raw_body.decode("utf-8"), signature, self._webhook_secret
        )

    def create_recovery_payment_link(
        self, *, case_id: str, amount_inr: float, customer_id: str | None
    ) -> dict[str, Any]:
        """Create a Razorpay test/live payment link after a human-approved recovery action."""
        return self._client.payment_link.create({
            "amount": round(amount_inr * 100),
            "currency": "INR",
            "reference_id": case_id,
            "description": f"Revenio recovery for {case_id}",
            "notes": {"revenio_case_id": case_id, "customer_reference": customer_id or ""},
        })
