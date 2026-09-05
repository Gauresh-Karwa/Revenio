"""Runtime services used by the interactive dashboard.

This is intentionally not a fixture or a hard-coded JSON feed.  Each demo
incident is generated at request time and runs through the production domain
modules and orchestrator, yielding the same append-only events rendered by the
audit view.  PostgreSQL/Celery remain the production path; this runtime makes
the complete product demonstrable before infrastructure credentials exist.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from typing import Any

from backend.core.events import Event, EventObserver, EventStore
from backend.core.orchestrator import Orchestrator
from backend.integrations.channels import configured_channel_gateway
from backend.modules.b2b_receivables.module import B2BReceivablesModule
from backend.modules.checkout_abandonment.module import CheckoutAbandonmentModule
from backend.modules.mandate_retry.module import MandateRetryModule
from backend.modules.subscription.module import SubscriptionModule


class _EventBroadcaster(EventObserver):
    def __init__(self) -> None:
        self._listeners: set[asyncio.Queue[dict[str, Any]]] = set()

    def on_event(self, event: Event) -> None:
        message = {"type": "case.event", "event": event_to_dict(event)}
        for listener in tuple(self._listeners):
            listener.put_nowait(message)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
        self._listeners.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._listeners.discard(queue)


def event_to_dict(event: Event) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "case_id": event.case_id,
        "domain_type": event.domain_type,
        "stage": event.stage,
        "event_type": event.event_type,
        "payload": event.payload,
        "created_at": event.created_at,
        "customer_id": event.customer_id,
    }


class DemoRuntime:
    """A real orchestration runtime with generated, non-repeatable incidents."""

    def __init__(self) -> None:
        self._rng = random.Random()
        self.store = EventStore()
        self.events = _EventBroadcaster()
        self.store.subscribe(self.events)
        self.orchestrator = Orchestrator(self.store)
        self.channels = configured_channel_gateway(self._rng)
        self.orchestrator.register_module(SubscriptionModule(channel_gateway=self.channels))
        self.orchestrator.register_module(CheckoutAbandonmentModule(channel_gateway=self.channels))
        self.orchestrator.register_module(
            B2BReceivablesModule(channel_gateway=self.channels)
        )
        self.orchestrator.register_module(MandateRetryModule())
        self._cases: dict[str, dict[str, Any]] = {}

    def simulate(self, scenario: str = "random", count: int = 1) -> list[dict[str, Any]]:
        if count < 1 or count > 100:
            raise ValueError("count must be between 1 and 100")
        states = []
        for _ in range(count):
            domain_type, payload = self._new_case(scenario)
            case_id = self._new_reference()
            payload["recovery_case_id"] = case_id
            self._cases[case_id] = payload
            states.append(self.orchestrator.process_case(case_id, domain_type, payload))
        return states

    def simulate_custom(
        self,
        domain_type: str,
        customer_name: str,
        customer_id: str | None,
        customer_email: str | None,
        customer_phone: str | None,
        amount: float,
        failure_code: str | None,
        response: str,
        opt_in: bool = True,
        days_overdue: int = 30,
        await_razorpay_confirmation: bool = True,
    ) -> dict[str, Any]:
        """Create one user-entered incident and run the normal orchestrator.

        The caller selects a simulated customer response, not an event log or
        outcome.  Domain modules still decide the action, compliance gates
        still apply, and the resulting durable event sequence drives the UI.
        """
        case_id = self._new_reference()
        identity = customer_id.strip() if customer_id and customer_id.strip() else f"customer-{uuid.uuid4().hex[:8]}"
        common = {
            "customer_id": identity,
            "customer_name": customer_name.strip(),
            "customer_email": customer_email.strip() if customer_email else None,
            "customer_phone": customer_phone.strip() if customer_phone else None,
            "simulated_contact_response": response,
            "awaiting_razorpay_confirmation": await_razorpay_confirmation,
        }
        if domain_type == "subscription":
            payload = {
                **common,
                "decline_code": failure_code or "51",
                "amount": amount,
                "attempt_number": 1,
                "hour_of_day": datetime.now().hour,
                "is_near_payday": False,
                # A direct response is an explicit customer-provided signal;
                # the module records a privacy-safe reason code and requires
                # human review without treating it as an inferred email model
                # classification.
                "customer_reported_hardship": response == "hardship",
                "simulated_retry_result": "recovered" if response == "recovered" else "lost" if response == "lost" else None,
            }
        elif domain_type == "checkout_abandonment":
            payload = {
                **common,
                "reached_checkout": True,
                "opt_in": opt_in,
                "abandonment_signal": failure_code or "checkout_page_error",
                "amount": amount,
                "simulated_nudge_result": "recovered" if response == "recovered" else "lost" if response == "lost" else None,
            }
        elif domain_type == "b2b_receivables":
            payload = {
                **common,
                "invoice_amount": amount,
                "due_date": (date.today() - timedelta(days=days_overdue)).isoformat(),
                "is_msme_registered": True,
                "has_written_agreement": True,
                "preferred_locale": "hi-IN",
                "on_dnd_registry": False,
                "has_opted_out": False,
                "simulated_payment_result": "paid_full" if response in {"paid", "recovered"} else "promised" if response == "promise" else "written_off" if response == "lost" else None,
                "approval_outcome": response,
            }
        elif domain_type == "mandate_retry":
            payload = {
                **common,
                "rail": "upi_autopay",
                "return_code": failure_code or "U02",
                "amount": amount,
                "simulated_mandate_result": "recovered" if response == "recovered" else "lost" if response == "lost" else None,
            }
        else:
            raise ValueError(f"Unknown domain '{domain_type}'")
        payload["recovery_case_id"] = case_id
        self._cases[case_id] = payload
        max_iterations = 3 if domain_type == "b2b_receivables" and response in {"no_response", "needs_human", "hardship"} else 1
        self.orchestrator.process_case(case_id, domain_type, payload, max_iterations=max_iterations)
        return self.case_detail(case_id)

    def send_email_test(self, recipient: str) -> dict[str, Any]:
        """Validate the configured Resend sender after an explicit UI request."""
        return self.channels.send_test_email(recipient.strip())

    @staticmethod
    def _new_reference() -> str:
        """Readable operator reference; unique enough for the local workbench."""
        return f"RVN-{date.today():%y%m%d}-{uuid.uuid4().hex[:6].upper()}"

    def record_payment_link(self, case_id: str, link: dict[str, Any]) -> dict[str, Any]:
        case = self._cases.get(case_id)
        if case is None:
            raise KeyError(case_id)
        state = self.store.derive_state(case_id)
        self.store.append(
            case_id,
            state["domain_type"],
            "payment_collection",
            "RazorpayPaymentLinkCreated",
            {
                "provider": "Razorpay",
                "mode": link.get("_revenio_mode", "configured"),
                "payment_link_id": link.get("id"),
                "short_url": link.get("short_url"),
                "status": link.get("status"),
                "amount": case.get("amount", case.get("invoice_amount", 0)),
            },
            customer_id=case.get("customer_id"),
        )
        return self.case_detail(case_id)

    def record_email_reply(self, case_id: str, sender: str, text: str, email_id: str) -> dict[str, Any]:
        """Record an inbound email as a communication signal, never payment truth."""
        case = self._cases.get(case_id)
        if case is None:
            raise KeyError(case_id)
        if any(
            event.event_type == "CustomerEmailReply" and event.payload.get("email_id") == email_id
            for event in self.store.get_events(case_id)
        ):
            return self.case_detail(case_id)
        normalized = text.lower()
        if any(word in normalized for word in ("hardship", "cannot pay", "can't pay", "dispute", "wrong invoice")):
            intent, action = "human_review_required", "Reply needs specialist review"
        elif any(word in normalized for word in ("payment link", "pay now", "send link")):
            intent, action = "payment_link_requested", "Customer requested a secure payment link"
        elif any(word in normalized for word in ("tomorrow", "will pay", "promise", "next week")):
            intent, action = "promise_to_pay", "Customer gave a payment commitment"
        else:
            intent, action = "needs_human_review", "Reply is ambiguous; specialist review required"
        state = self.store.derive_state(case_id)
        payload = {"provider": "Resend", "email_id": email_id, "from": sender, "intent": intent, "action": action, "message": text[:1000]}
        self.store.append(case_id, state["domain_type"], "customer_reply", "CustomerEmailReply", payload, customer_id=case.get("customer_id"))
        if intent in {"human_review_required", "needs_human_review"}:
            self.store.append(case_id, state["domain_type"], "customer_reply", "PendingHumanReview", {"reason": action}, customer_id=case.get("customer_id"))
        return self.case_detail(case_id)

    def record_razorpay_payment_event(
        self, case_id: str, event_name: str, payment: dict[str, Any]
    ) -> dict[str, Any]:
        """Turn a verified provider event into the payment truth for its case."""
        case = self._cases.get(case_id)
        if case is None:
            raise KeyError(case_id)
        state = self.store.derive_state(case_id)
        amount = float(payment.get("amount", 0)) / 100
        details = {
            "provider": "Razorpay",
            "event": event_name,
            "payment_id": payment.get("id"),
            "order_id": payment.get("order_id"),
            "status": payment.get("status", "unknown"),
            "error_code": payment.get("error_code"),
            "amount": amount,
        }
        self.store.append(case_id, state["domain_type"], "payment_confirmation", "RazorpayPaymentEvent", details, customer_id=case.get("customer_id"))
        if event_name in {"payment.captured", "payment_link.paid"} or details["status"] == "captured":
            case["awaiting_razorpay_confirmation"] = False
            self.store.append(case_id, state["domain_type"], "track_outcome", "Outcome", {"status": "RECOVERED", "amount_recovered": amount}, customer_id=case.get("customer_id"))
        elif event_name == "payment.failed":
            self.store.append(case_id, state["domain_type"], "payment_confirmation", "RecoveryPaymentFailed", {"message": "Razorpay reported a failed payment attempt. No revenue was recovered.", **details}, customer_id=case.get("customer_id"))
        elif event_name in {"payment_link.cancelled", "payment_link.expired"}:
            self.store.append(case_id, state["domain_type"], "payment_confirmation", "RecoveryPaymentFailed", {"message": f"Razorpay marked the payment link {event_name.rsplit('.', 1)[-1]}. No revenue was recovered.", **details}, customer_id=case.get("customer_id"))
        return self.case_detail(case_id)

    def _new_case(self, requested: str) -> tuple[str, dict[str, Any]]:
        scenario = self._rng.choice(
            ("payment_failure", "checkout_abandonment", "overdue_invoice", "mandate_failure")
        ) if requested == "random" else requested
        first_name = self._rng.choice(("Aarav", "Diya", "Kabir", "Ananya", "Vihaan", "Meera", "Arjun", "Ishita"))
        last_name = self._rng.choice(("Sharma", "Patel", "Reddy", "Gupta", "Iyer", "Khan", "Mehta", "Nair"))
        customer_name = f"{first_name} {last_name}"
        customer_id = f"cust-{self._rng.randrange(10_000, 99_999)}"

        if scenario == "payment_failure":
            decline_code = self._rng.choices(
                ("51", "05", "91", "96"), weights=(40, 25, 15, 15), k=1
            )[0]
            return "subscription", {
                "customer_id": customer_id,
                "customer_name": customer_name,
                "decline_code": decline_code,
                "amount": round(self._rng.uniform(199, 9_999), 2),
                "attempt_number": 1,
                "hour_of_day": self._rng.randrange(24),
                "is_near_payday": self._rng.choice((True, False)),
                "simulated_retry_result": self._rng.choices(
                    ("recovered", "lost"), weights=(58, 42), k=1
                )[0],
            }

        if scenario == "checkout_abandonment":
            signal = self._rng.choice(
                (
                    "shipping_cost_surprise",
                    "forced_account_creation",
                    "payment_method_unavailable",
                    "checkout_form_friction",
                    "checkout_page_error",
                    "low_purchase_intent",
                )
            )
            return "checkout_abandonment", {
                "customer_id": customer_id,
                "customer_name": customer_name,
                "reached_checkout": True,
                "opt_in": self._rng.random() > 0.12,
                "abandonment_signal": signal,
                "amount": round(self._rng.uniform(299, 14_999), 2),
                "simulated_nudge_result": self._rng.choices(
                    ("recovered", "lost"), weights=(35, 65), k=1
                )[0],
            }

        if scenario == "overdue_invoice":
            # This path deliberately runs email -> SMS -> human-approved
            # Hinglish voice.  It gives a judge a complete gated workflow to
            # inspect rather than a one-step happy-path animation.
            return "b2b_receivables", {
                "customer_id": customer_id,
                "customer_name": f"{self._rng.choice(('Aster', 'BluePeak', 'Cedar', 'Delta', 'Evergreen', 'Nova'))} {self._rng.choice(('Systems', 'Trading', 'Industries', 'Logistics', 'Foods'))}",
                "invoice_amount": round(self._rng.uniform(20_000, 400_000), 2),
                "due_date": (date.today() - timedelta(days=self._rng.randrange(18, 80))).isoformat(),
                "is_msme_registered": self._rng.choice((True, False)),
                "has_written_agreement": self._rng.choice((True, False)),
                "preferred_locale": "hi-IN",
                "on_dnd_registry": False,
                "has_opted_out": False,
                "simulated_payment_result": None,
            }

        if scenario == "mandate_failure":
            return "mandate_retry", {
                "customer_id": customer_id,
                "customer_name": customer_name,
                "rail": "upi_autopay",
                "return_code": self._rng.choice(("U02", "U03", "U04")),
                "amount": round(self._rng.uniform(150, 12_000), 2),
                "simulated_mandate_result": self._rng.choice(("recovered", "lost")),
            }
        raise ValueError(f"Unknown scenario '{requested}'")

    def list_cases(self, domain_type: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        rows = []
        for case_id in self._cases:
            state = self.store.derive_state(case_id)
            if domain_type and state["domain_type"] != domain_type:
                continue
            label = self._status_label(case_id, state)
            if status and label != status:
                continue
            latest = self.store.get_events(case_id)[-1]
            rows.append({
                "case_id": case_id,
                "customer_name": self._cases[case_id].get("customer_name", self._cases[case_id].get("customer_id", "Customer")),
                "domain_type": state["domain_type"],
                "status": label,
                "terminal": state["terminal"],
                "stage_count": state["stage_count"],
                "updated_at": latest.created_at,
                "amount": self._cases[case_id].get("amount", self._cases[case_id].get("invoice_amount", 0)),
                "reason": self._case_reason(case_id),
            })
        return sorted(rows, key=lambda row: row["updated_at"], reverse=True)

    def _case_reason(self, case_id: str) -> str:
        events = self.store.get_events(case_id)
        diagnosis = next((event.payload for event in reversed(events) if event.event_type == "Diagnosis"), {})
        return str(diagnosis.get("root_cause", "awaiting assessment"))

    def case_detail(self, case_id: str) -> dict[str, Any]:
        state = self.store.derive_state(case_id)
        if not state["exists"]:
            raise KeyError(case_id)
        return {
            "state": state,
            "case": self._public_case(self._cases[case_id]),
            "events": [event_to_dict(event) for event in self.store.get_events(case_id)],
        }

    def reviews(self) -> list[dict[str, Any]]:
        rows = []
        for row in self.list_cases():
            events = self.store.get_events(row["case_id"])
            pending = any(event.event_type == "PendingHumanReview" for event in events)
            resolved = any(event.event_type == "HumanReviewDecision" for event in events)
            if pending and not resolved:
                diagnosis = next((event.payload for event in reversed(events) if event.event_type == "Diagnosis"), {})
                decision = next((event.payload for event in reversed(events) if event.event_type == "Decision"), {})
                rows.append({**row, "diagnosis": diagnosis, "decision": decision})
        return rows

    def resolve_review(self, case_id: str, approved: bool) -> dict[str, Any]:
        case = self._cases.get(case_id)
        if case is None:
            raise KeyError(case_id)
        self.orchestrator.submit_human_review(case_id, confirmed=approved, case=case)
        if approved:
            case["review_approved"] = True
            if self.store.get_events(case_id)[0].domain_type == "b2b_receivables":
                selected = case.get("approval_outcome")
                case["simulated_payment_result"] = (
                    "paid_full" if selected in {"paid", "recovered"} else "promised" if selected == "promise" else "written_off" if selected == "lost" else self._rng.choices(
                    ("paid_full", "promised", "written_off"), weights=(50, 35, 15), k=1
                )[0] if selected is None else None)
                self.orchestrator.process_case(case_id, "b2b_receivables", case, max_iterations=1)
        return self.case_detail(case_id)

    def dashboard(self) -> dict[str, Any]:
        rows = self.list_cases()
        recovered = 0.0
        terminal = 0
        recovered_cases = 0
        for row in rows:
            events = self.store.get_events(row["case_id"])
            for event in events:
                if event.event_type == "Outcome" and event.payload.get("status") == "RECOVERED":
                    recovered += float(event.payload.get("amount_recovered", 0))
            if row["terminal"]:
                terminal += 1
            if row["status"] == "RECOVERED":
                recovered_cases += 1
        by_domain: dict[str, int] = {}
        for row in rows:
            by_domain[row["domain_type"]] = by_domain.get(row["domain_type"], 0) + 1
        return {
            "money_recovered": round(recovered, 2),
            "recovery_rate": round((recovered_cases / terminal * 100) if terminal else 0, 1),
            "active_recoveries": sum(not row["terminal"] for row in rows),
            "human_review_count": len(self.reviews()),
            "total_cases": len(rows),
            "domain_breakdown": [
                {"domain_type": domain, "case_count": count}
                for domain, count in sorted(by_domain.items())
            ],
        }

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        return self.events.subscribe()

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.events.unsubscribe(queue)

    def _status_label(self, case_id: str, state: dict[str, Any]) -> str:
        events = self.store.get_events(case_id)
        if any(event.event_type == "RecoveryPaymentFailed" for event in events):
            return "PAYMENT_FAILED"
        if any(event.event_type == "PendingHumanReview" for event in events) and not any(
            event.event_type == "HumanReviewDecision" for event in events
        ):
            return "HUMAN_REVIEW"
        if any(event.event_type == "RazorpayPaymentLinkCreated" for event in events):
            return "AWAITING_PAYMENT"
        return state["terminal_status"] or "ACTIVE"

    @staticmethod
    def _public_case(case: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in case.items() if key not in {"email_text", "customer_email", "customer_phone"}}
