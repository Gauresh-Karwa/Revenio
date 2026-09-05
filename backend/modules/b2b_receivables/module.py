"""
B2B receivables chaser. Architecture doc: "richest compliance/escalation
story, weakest ground truth" — built third, once the orchestrator and
learning core are already proven on two simpler domains.

TWO REAL COMPLIANCE SOURCES, DELIBERATELY NOT CONFLATED:

1. Section 43B(h), Income Tax Act (Finance Act 2023, effective 1 Apr 2024),
   read with Section 15, MSMED Act 2006. A Udyam-registered Micro or Small
   Enterprise must be paid within 45 days if a written agreement exists,
   15 days if not — clock starts from acceptance of goods/services, not
   invoice date. Missing the deadline disallows the BUYER's tax deduction
   for that expense until the year actual payment is made. This is a real,
   current rule (verified July 2026) — but it is a TAX CONSEQUENCE FOR THE
   BUYER, not a rule about whether we're allowed to contact anyone. It
   shapes urgency (raw_signal), never check_stop. Conflating it with a
   contact-permission rule would be a real compliance-logic error, not
   just an imprecise comment.

2. DND / National Customer Preference Register (NCPR), under TCCCPR 2010
   as amended (most recently TCCCPR Second Amendment, 2025). Applies to
   the SUBSCRIBER — the person whose name the phone connection is in —
   with NO blanket B2B exemption; a proprietor's personal mobile carries
   personal DND preferences regardless of the business reason for calling
   it. A real transactional/service-message exemption path exists (via a
   correctly DLT-registered template and number series), but using it
   correctly requires infrastructure (DLT registration, template
   categorization) this system does not have. DELIBERATE, DOCUMENTED
   CHOICE: treat any DND/opt-out signal as blocking ALL channels — safety-
   first, not an attempt to exploit an exemption we can't actually verify.

CHANNEL + LOCALE, per architecture doc 1.1's own scoping decision:
"Hinglish voice recovery... one possible value of the shared
SWITCH_CHANNEL action (a channel + locale combination), not a separate
domain." Implemented here exactly that way — voice contact carries a
`locale` in action_params, defaulting to "hi-IN" (Hindi), with Hinglish
understood as a code-mixed SPOKEN STYLE within that locale rather than a
separate ISO/BCP-47 tag (no such standard tag exists) — a real execution
detail of the voice channel, not a fourth channel type.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from backend.core.contract import (
    ActionType,
    Decision,
    Diagnosis,
    ExecutionResult,
    Outcome,
    OutcomeStatus,
    PromiseOutcome,
    StopDecision,
    StopReason,
)

# --- Section 43B(h) / MSMED Act Section 15 — sourced, verified July 2026 ---
MSME_PAYMENT_DEADLINE_DAYS_WITH_AGREEMENT = 45
MSME_PAYMENT_DEADLINE_DAYS_NO_AGREEMENT = 15

# --- Channel escalation. NOT a bandit arm menu by default (that wiring is
# the same optional learning_core pattern already used by the other two
# modules — see __init__) — a fixed order is the safe, explainable default.
CHANNEL_ESCALATION = ["email", "sms", "voice"]
DEFAULT_VOICE_LOCALE = "hi-IN"  # see module docstring's Hinglish note

# --- Judgment-call thresholds, flagged as estimated, not sourced — same
# discipline as MAX_NUDGES in checkout_abandonment. B2B collections
# realistically continues longer than a single subscription retry or
# abandonment nudge, given typical invoice values are larger.
MAX_CONTACT_ATTEMPTS = 5
# Docx section 3.6: "a broken promise re-enters the loop at check_stop...
# exactly the kind of event that can trigger DIMINISHING_RETURNS." How many
# broken promises before that fires is explicitly an open item in the
# architecture doc — this is the estimated default, flagged.
MAX_BROKEN_PROMISES = 2


def _days_overdue(case: dict[str, Any], as_of: date | None = None) -> int:
    as_of = as_of or datetime.now(timezone.utc).date()
    due_date = case.get("due_date")
    if due_date is None:
        return 0
    if isinstance(due_date, str):
        due_date = date.fromisoformat(due_date)
    return max(0, (as_of - due_date).days)


def _msme_deadline_days(case: dict[str, Any]) -> int | None:
    if not case.get("is_msme_registered", False):
        return None  # Section 43B(h) simply doesn't apply — not a real deadline to track
    return (
        MSME_PAYMENT_DEADLINE_DAYS_WITH_AGREEMENT
        if case.get("has_written_agreement", False)
        else MSME_PAYMENT_DEADLINE_DAYS_NO_AGREEMENT
    )


def _count_broken_promises(history: list[dict[str, Any]]) -> int:
    return sum(
        1 for h in history
        if h.get("_event_type") == "PromiseOutcome" and h.get("kept") is False
    )


class B2BReceivablesModule:
    domain_type = "b2b_receivables"

    def __init__(self, learning_core: Any = None, channel_gateway: Any = None) -> None:
        """
        learning_core: optional backend.core.learning_core.LearningCore,
        same pattern as SubscriptionModule/CheckoutAbandonmentModule. When
        provided and has a policy for "b2b_receivables" (3 arms, matching
        CHANNEL_ESCALATION), decide() asks it which channel arm to pull
        instead of the fixed email->sms->voice order. None (default)
        preserves the fixed schedule.
        """
        self._learning_core = learning_core
        # A gateway is injected instead of baked into the module so its
        # policy remains independently testable.  Production can supply an
        # approved SMS/voice provider; the interactive demo supplies a
        # runtime simulator that returns a generated customer response.
        self._channel_gateway = channel_gateway

    def check_stop(
        self, case: dict[str, Any], history: list[dict[str, Any]]
    ) -> StopDecision:
        # DND / consent gate — see module docstring for why this is
        # deliberately conservative (blocks ALL channels, not just
        # promotional ones, rather than relying on an exemption path this
        # system can't correctly verify).
        if case.get("on_dnd_registry", False) or case.get("has_opted_out", False):
            return StopDecision(should_stop=True, stop_reason=StopReason.OPT_OUT)

        # A disputed invoice needs human/legal handling, not more automated
        # contact — continuing the automated loop here has real legal and
        # reputational cost that outweighs any benefit. Flagged as a
        # judgment call: COST_THRESHOLD is the closest fit among the
        # existing shared StopReason vocabulary, not a perfect label.
        if case.get("is_disputed", False):
            return StopDecision(should_stop=True, stop_reason=StopReason.COST_THRESHOLD)

        # Broken-promise-triggered diminishing returns — docx section 3.6,
        # explicitly named as the canonical example of this stop reason.
        if _count_broken_promises(history) >= MAX_BROKEN_PROMISES:
            return StopDecision(should_stop=True, stop_reason=StopReason.DIMINISHING_RETURNS)

        contact_count = sum(1 for h in history if h.get("_event_type") == "ExecutionResult")

        if (
            self._learning_core is not None
            and self._learning_core.has_policy(self.domain_type)
            and contact_count >= 1
        ):
            arms = self._learning_core.snapshot()[self.domain_type]["arms"]
            if all(a["pull_count"] >= 20 for a in arms):
                best_estimate = max((a.get("mean_estimate", a.get("mean_reward")) or 0.0) for a in arms)
                if best_estimate < 0.10:
                    return StopDecision(should_stop=True, stop_reason=StopReason.DIMINISHING_RETURNS)

        if contact_count >= MAX_CONTACT_ATTEMPTS:
            return StopDecision(should_stop=True, stop_reason=StopReason.DIMINISHING_RETURNS)

        return StopDecision(should_stop=False)

    def diagnose(
        self, case: dict[str, Any], customer_history: list[dict[str, Any]] | None = None
    ) -> Diagnosis:
        invoice_amount = case.get("invoice_amount")
        if invoice_amount is None or invoice_amount <= 0:
            return Diagnosis(
                root_cause="unmapped_invoice_data",
                is_recoverable=False,
                confidence=0.2,
                raw_signal={"source": "missing_or_invalid_invoice_amount"},
            )

        days_overdue = _days_overdue(case)
        msme_deadline = _msme_deadline_days(case)

        raw_signal: dict[str, Any] = {
            "days_overdue": days_overdue,
            "is_msme_registered": case.get("is_msme_registered", False),
            "msme_payment_deadline_days": msme_deadline,
        }
        # Section 43B(h) urgency signal — a tax-deduction consequence for
        # our merchant client, not a contact-permission rule (see module
        # docstring). Purely informational here: it can motivate PRIORITY
        # in a real merchant-facing view, never gates check_stop/execute.
        if msme_deadline is not None:
            raw_signal["days_until_43bh_deadline"] = msme_deadline - days_overdue

        return Diagnosis(
            root_cause="overdue_invoice",
            is_recoverable=True,
            confidence=0.9,
            raw_signal=raw_signal,
        )

    def decide(
        self, case: dict[str, Any], diagnosis: Diagnosis, history: list[dict[str, Any]]
    ) -> Decision:
        if diagnosis.confidence < 0.5:
            return Decision(
                action_type=ActionType.ESCALATE,
                reasoning="Invoice data incomplete or invalid — cannot proceed automatically.",
                requires_human_review=True,
            )

        # An active, not-yet-due promise-to-pay means WAIT, not re-contact.
        active_promise = case.get("active_promise_date")
        if active_promise is not None:
            return Decision(
                action_type=ActionType.WAIT,
                action_params={"promised_date": active_promise},
                reasoning=f"Customer has an active promise to pay by {active_promise} — waiting, not re-contacting.",
                requires_human_review=False,
            )

        contact_count = sum(1 for h in history if h.get("_event_type") == "ExecutionResult")

        action_params: dict[str, Any] = {}
        if self._learning_core is not None and self._learning_core.has_policy(self.domain_type):
            arm = self._learning_core.select_arm(self.domain_type)
            channel = CHANNEL_ESCALATION[arm]
            action_params["bandit_arm"] = arm
        else:
            channel_index = min(contact_count, len(CHANNEL_ESCALATION) - 1)
            channel = CHANNEL_ESCALATION[channel_index]

        action_params["channel"] = channel
        if channel == "voice":
            action_params["locale"] = case.get("preferred_locale", DEFAULT_VOICE_LOCALE)

        # Docx: "the domain that actually exercises the human-review-queue
        # path most fully." Reaching the final channel tier (voice) without
        # resolution is a real, deliberate trigger for that — a large
        # overdue B2B invoice reaching the most escalated automated channel
        # is exactly the kind of case that should get a human's eyes,
        # not one more automated attempt indistinguishable from the rest.
        requires_review = (
            channel == "voice"
            and contact_count >= len(CHANNEL_ESCALATION) - 1
            and not case.get("review_approved", False)
        )

        return Decision(
            action_type=ActionType.SWITCH_CHANNEL,
            action_params=action_params,
            reasoning=(
                f"Overdue invoice ({diagnosis.raw_signal.get('days_overdue', 0)} days), "
                f"contact #{contact_count + 1} via {channel}"
            ),
            requires_human_review=requires_review,
        )

    def execute(self, case: dict[str, Any], decision: Decision) -> ExecutionResult:
        # Double-enforcement, matching checkout_abandonment's own pattern:
        # DND is checked again here, not just trusted from check_stop.
        dnd_blocked = case.get("on_dnd_registry", False) or case.get("has_opted_out", False)
        details: dict[str, Any] = {}
        if not dnd_blocked and self._channel_gateway is not None:
            details = self._channel_gateway.dispatch(case, decision)
        return ExecutionResult(
            success=not dnd_blocked,
            compliance_check_passed=not dnd_blocked,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details,
        )

    def track_outcome(self, case: dict[str, Any]) -> Outcome:
        if case.get("awaiting_razorpay_confirmation"):
            return Outcome(status=OutcomeStatus.PENDING, amount_recovered=0.0)
        simulated = case.get("simulated_payment_result")
        if simulated == "paid_full":
            return Outcome(
                status=OutcomeStatus.RECOVERED,
                amount_recovered=case.get("invoice_amount", 0.0),
            )
        if simulated == "promised":
            promised_date = case.get("promised_date")
            return Outcome(
                status=OutcomeStatus.PROMISED,
                amount_recovered=0.0,
                details={"promised_date": promised_date},
            )
        if simulated == "written_off":
            return Outcome(status=OutcomeStatus.LOST, amount_recovered=0.0)
        return Outcome(status=OutcomeStatus.PENDING, amount_recovered=0.0)

    def on_promise_due(self, case: dict[str, Any]) -> PromiseOutcome:
        """
        Docx section 3.6: fires when a PROMISED case reaches its promised
        date. kept is determined by the caller-provided simulated signal
        (a real system would check actual payment receipt here) — same
        "simulated_X_result" testing pattern used by every other domain's
        track_outcome.
        """
        kept = case.get("simulated_promise_kept", False)
        return PromiseOutcome(kept=kept)
