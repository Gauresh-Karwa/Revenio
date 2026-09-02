"""
backend/modules/mandate_retry/module.py

Mandate retry sequencer — Step 8 (stretch), per architecture doc 1.1 & 4:
"reuses the subscription domain's shape on a different payment rail, so
it's cheap to add once that module exists." Built exactly that way: the
same diagnose/decide/check_stop/execute/track_outcome shape as
SubscriptionModule, with a rail-specific (UPI Autopay / NACH) taxonomy in
place of ISO 8583 decline codes.

TWO RAILS, TWO REAL COMPLIANCE SOURCES, DELIBERATELY NOT CONFLATED — same
discipline as B2BReceivablesModule's two-sources docstring:

1. UPI AUTOPAY (NPCI), verified against current 2026 rules:
   - RBI's e-mandate "Additional Factor of Authentication" (AFA) exemption:
     recurring UPI Autopay debits up to Rs 15,000 execute without UPI-PIN
     re-entry; above that threshold, the customer's bank REQUIRES a fresh
     UPI-PIN authentication for that specific debit. This is not a
     probabilistic recovery question the way code-51 amount-dependence is
     for subscriptions — it is a deterministic rule. A case above the
     threshold cannot be silently auto-retried; the customer must be
     pushed to open their UPI app and authenticate. Modeled here as its
     own root_cause / SWITCH_CHANNEL action, not folded into the ordinary
     retry path.
   - NPCI's August-2026 execution-window rule: each Autopay mandate gets
     exactly ONE main execution attempt plus UP TO THREE retries (4 total
     attempts), and mandate executions must run in non-peak windows only.
     This is the authoritative basis for MAX_UPI_AUTOPAY_ATTEMPTS — a real
     network rule, not an invented threshold — the same role Visa's
     retry-eligibility categories play for COMPLIANCE_LIMIT in the
     subscription module.
   - The exact 24h/72h/168h backoff spacing used below is a DIRECTIONALLY
     sourced industry practice (multiple payments-ops writeups recommend
     spaced, non-consecutive retries), NOT an NPCI-mandated number — same
     flagging discipline as subscription's ATTEMPT_DECAY magnitude and
     checkout-abandonment's MAX_NUDGES. Tracked as an open item.

2. NACH (RBI's ECS(Debit) Procedural Guidelines, which NACH operates
   under as NPCI's electronic successor to ECS):
   - Return-reason-codes 1, 2, and 3 are explicitly documented as items
     that must NOT be resent/re-presented "without carrying out the
     necessary corrections" first — i.e. these are not a "try again
     later" situation, they need a human or a data fix before any further
     automated presentation is legitimate. Modeled as its own bucket,
     distinct from both "auto-retryable" and "hard stop": recoverable in
     principle, but requires_human_review=True rather than RETRY.
   - Return-reason-code 8 is explicitly documented as "mandate not
     received" — there is no active mandate to collect against at all.
     Modeled as a hard stop (OPT_OUT-equivalent: no consent instrument
     exists), matching subscription's stop-instruction-code treatment.
   - Return-reason-code 9 is documented as "miscellaneous" — modeled the
     same way subscription treats an unmapped decline code: low
     confidence, escalate, never guessed.
   - The RBI guideline separately states that an item "repeatedly
     presented... for more than three ECS runs" may be refused further
     entertainment by the clearing house — the authoritative basis for
     MAX_NACH_PRESENTATIONS = 3.
   - HONEST GAP, flagged rather than guessed: the RBI ECS(Debit) guideline
     text located and cited above confirms the CATEGORY behavior of
     reason codes 1-3, 8, and 9, but does not give this project a verified
     digit-for-digit mapping of every NACH return code (e.g. which single
     digit means "insufficient funds" specifically) against NPCI's current
     numeric reason-code list. Rather than inventing a plausible-looking
     number, the insufficient-funds case below uses a symbolic code
     ("NACH_INSUFFICIENT_FUNDS") until the exact NPCI numeric mapping is
     verified — tracked as an open item, exactly like checkout-abandonment's
     unsourced per-signal recovery rates.

Interesting emergent finding, not assumed going in: both independently
regulated rails converge on the same retry ceiling — NPCI caps UPI Autopay
at 1 main attempt + 3 retries, and RBI's ECS(Debit) guideline caps NACH
re-presentment at 3 runs before the clearing house may refuse further
entertainment. COMPLIANCE_LIMIT for this module is grounded in real,
rail-specific network rules either way, matching the discipline used for
Visa's retry-eligibility categories in the subscription module.

LEARNING-CORE WIRING, same constructor-injection pattern as
SubscriptionModule and CheckoutAbandonmentModule — a
`learning_core: LearningCore | None = None` argument, `None` preserving
the fixed schedule exactly. Scoped DELIBERATELY to UPI Autopay only:

UPI_RETRY_BACKOFF_HOURS already has 3 real, distinct values (24/72/168h,
directionally sourced) — a genuine multi-arm choice, the same shape as
subscription's 4-arm RETRY_BACKOFF_HOURS bandit. NACH's retry cadence
below is currently a single fixed constant (24h) with no second sourced
alternative to choose between; wiring a bandit over a 1-arm "choice" would
be theater, not a real policy decision. Rather than inventing plausible-
looking NACH cadence options just to give the bandit something to do, this
stays a fixed schedule for NACH and is flagged here as an open item —
exactly the "flag rather than guess" discipline used throughout this
project. `BanditUpdateObserver` (backend/core/bandit_observer.py) needed
ZERO changes to support this: it's already domain-agnostic, keyed only on
the `bandit_arm` key inside `action_params` and the module's own
`domain_type` string.
"""

from __future__ import annotations

from datetime import datetime, timezone
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

# --- UPI Autopay taxonomy ---------------------------------------------------

UPI_AUTOPAY_SOFT_CODES: dict[str, str] = {
    "U01": "insufficient_funds",
    "U02": "issuer_bank_unavailable",
    "U03": "npci_technical_decline",
    "U04": "beneficiary_bank_timeout",
}

UPI_AUTOPAY_STOP_CODES: dict[str, str] = {
    "U_REVOKED": "mandate_revoked_by_customer",
    "U_PAUSED": "mandate_paused_by_customer",
    "U_EXPIRED": "mandate_validity_expired",
}

# Sourced: RBI e-mandate AFA-exemption threshold, current as of 2026 —
# recurring UPI Autopay debits at or below this amount execute without a
# fresh UPI-PIN entry; above it, the bank requires manual re-authentication
# for that specific debit. A deterministic rule, not a probability.
AFA_EXEMPTION_THRESHOLD_INR = 15_000.0

# Sourced: NPCI's 2026 execution-window rule — 1 main attempt + up to 3
# retries per mandate. 4 total attempts, not 3 — matches subscription's
# "the cap counts total attempts" convention (MAX_RETRY_ATTEMPTS there is
# also a total, not an "additional retries" count).
MAX_UPI_AUTOPAY_ATTEMPTS = 4

# Directionally sourced (spaced retries, not same-day bursts), NOT an NPCI-
# mandated schedule — flagged exactly like ATTEMPT_DECAY's magnitude and
# checkout-abandonment's MAX_NUDGES. Open item, not silently asserted. Also
# doubles as the bandit's fixed 3-arm action space when a learning_core is
# wired (see module docstring's LEARNING-CORE WIRING section).
UPI_RETRY_BACKOFF_HOURS = [24, 72, 168]

# --- NACH taxonomy -----------------------------------------------------------

# Return-reason-codes 1-3: RBI ECS(Debit) Procedural Guidelines — must not
# be resent/represented without correcting the underlying account data
# first. Recoverable in principle, but never by blind automated retry.
NACH_CORRECTION_REQUIRED_CODES: dict[str, str] = {
    "1": "account_data_correction_required",
    "2": "account_data_correction_required",
    "3": "account_data_correction_required",
}

# Return-reason-code 8: RBI ECS(Debit) Procedural Guidelines — "ECS mandate
# not received." No active mandate exists; this is a hard stop, not a
# retryable failure.
NACH_MANDATE_NOT_RECEIVED_CODE = "8"

# Return-reason-code 9: RBI ECS(Debit) Procedural Guidelines — "Miscellaneous
# (to be specified)." Treated the same way subscription treats an unmapped
# decline code: never guessed, always low-confidence + escalate.
NACH_MISCELLANEOUS_CODE = "9"

# Symbolic, NOT a verified NPCI numeric code — see module docstring's
# "HONEST GAP" note. Placeholder until the exact digit is confirmed against
# NPCI's current reason-code list.
NACH_INSUFFICIENT_FUNDS_CODE = "NACH_INSUFFICIENT_FUNDS"

# Sourced: RBI ECS(Debit) Procedural Guidelines — an item "repeatedly
# presented... for more than three ECS runs" may be refused further
# entertainment by the clearing house. 3 total presentations, matching
# both rails converging on the same ceiling independently.
MAX_NACH_PRESENTATIONS = 3

# Fixed, unsourced-magnitude NACH re-presentment cadence — see module
# docstring's LEARNING-CORE WIRING note on why this stays fixed (no second
# sourced alternative exists yet to make a bandit arm meaningful).
NACH_RETRY_HOURS = 24

RAILS = {"upi_autopay", "nach"}


def _attempt_count(history: list[dict[str, Any]]) -> int:
    return sum(1 for h in history if h.get("_event_type") == "ExecutionResult")


class MandateRetryModule:
    domain_type = "mandate_retry"

    # Bandit-informed diminishing-returns thresholds — same values and same
    # role as SubscriptionModule's, applied only on the UPI Autopay rail
    # (the only rail with a real bandit arm space; see module docstring).
    DIMINISHING_RETURNS_MIN_PULLS_PER_ARM = 20
    DIMINISHING_RETURNS_PROBABILITY_FLOOR = 0.10
    DIMINISHING_RETURNS_MIN_CASE_RETRIES = 2

    def __init__(self, learning_core: Any = None) -> None:
        """
        learning_core: optional backend.core.learning_core.LearningCore.
        When provided AND it has a policy registered for "mandate_retry"
        (3 arms, matching UPI_RETRY_BACKOFF_HOURS), decide() asks it which
        backoff arm to pull for a UPI Autopay retry instead of the fixed
        escalating schedule. NACH is unaffected either way — see module
        docstring. None (default) preserves the original fixed-schedule
        behavior exactly, same contract as every other module's optional
        learning_core parameter.
        """
        self._learning_core = learning_core

    def check_stop(
        self, case: dict[str, Any], history: list[dict[str, Any]]
    ) -> StopDecision:
        rail = case.get("rail")
        code = case.get("return_code")
        attempt_count = _attempt_count(history)

        if rail == "upi_autopay":
            if code in UPI_AUTOPAY_STOP_CODES:
                return StopDecision(should_stop=True, stop_reason=StopReason.OPT_OUT)

            if (
                self._learning_core is not None
                and self._learning_core.has_policy(self.domain_type)
                and attempt_count >= self.DIMINISHING_RETURNS_MIN_CASE_RETRIES
            ):
                arms = self._learning_core.snapshot()[self.domain_type]["arms"]
                if all(a["pull_count"] >= self.DIMINISHING_RETURNS_MIN_PULLS_PER_ARM for a in arms):
                    best_estimate = max(
                        (a.get("mean_estimate", a.get("mean_reward")) or 0.0) for a in arms
                    )
                    if best_estimate < self.DIMINISHING_RETURNS_PROBABILITY_FLOOR:
                        return StopDecision(should_stop=True, stop_reason=StopReason.DIMINISHING_RETURNS)

            if attempt_count >= MAX_UPI_AUTOPAY_ATTEMPTS:
                return StopDecision(should_stop=True, stop_reason=StopReason.COMPLIANCE_LIMIT)
            return StopDecision(should_stop=False)

        if rail == "nach":
            if code == NACH_MANDATE_NOT_RECEIVED_CODE:
                return StopDecision(should_stop=True, stop_reason=StopReason.OPT_OUT)
            # NACH_CORRECTION_REQUIRED_CODES deliberately does NOT stop here
            # — it is recoverable once corrected, so it goes to decide() and
            # is escalated to human review there, exactly the way an
            # unmapped subscription decline code is escalated rather than
            # silently stopped.
            if attempt_count >= MAX_NACH_PRESENTATIONS:
                return StopDecision(should_stop=True, stop_reason=StopReason.COMPLIANCE_LIMIT)
            return StopDecision(should_stop=False)

        # Unknown/missing rail: never guess, never stop silently — decide()
        # will route this to human review via the standard low-confidence gate.
        return StopDecision(should_stop=False)

    def diagnose(
        self, case: dict[str, Any], customer_history: list[dict[str, Any]] | None = None
    ) -> Diagnosis:
        rail = case.get("rail")
        code = case.get("return_code")
        amount = case.get("amount", 0.0)

        if rail == "upi_autopay":
            if code in UPI_AUTOPAY_STOP_CODES:
                return Diagnosis(
                    root_cause=UPI_AUTOPAY_STOP_CODES[code],
                    is_recoverable=False,
                    confidence=0.95,
                    raw_signal={"rail": rail, "return_code": code, "source": "npci_mandate_status"},
                )

            # Deterministic AFA rule, not a probabilistic diagnosis — checked
            # before the soft-code lookup because it applies REGARDLESS of
            # which soft failure code (if any) was returned.
            if amount > AFA_EXEMPTION_THRESHOLD_INR:
                return Diagnosis(
                    root_cause="afa_reauth_required_above_threshold",
                    is_recoverable=True,
                    confidence=0.95,
                    raw_signal={
                        "rail": rail, "amount": amount,
                        "afa_exemption_threshold_inr": AFA_EXEMPTION_THRESHOLD_INR,
                        "source": "rbi_e_mandate_afa_rule",
                    },
                )

            if code in UPI_AUTOPAY_SOFT_CODES:
                return Diagnosis(
                    root_cause=UPI_AUTOPAY_SOFT_CODES[code],
                    is_recoverable=True,
                    confidence=0.9,
                    raw_signal={"rail": rail, "return_code": code, "source": "npci_soft_lookup"},
                )

            return Diagnosis(
                root_cause="unmapped_upi_autopay_code",
                is_recoverable=False,
                confidence=0.2,
                raw_signal={"rail": rail, "return_code": code, "source": "unmapped"},
            )

        if rail == "nach":
            if code == NACH_MANDATE_NOT_RECEIVED_CODE:
                return Diagnosis(
                    root_cause="mandate_not_received",
                    is_recoverable=False,
                    confidence=0.95,
                    raw_signal={"rail": rail, "return_code": code, "source": "rbi_ecs_debit_guidelines"},
                )

            if code in NACH_CORRECTION_REQUIRED_CODES:
                return Diagnosis(
                    root_cause=NACH_CORRECTION_REQUIRED_CODES[code],
                    is_recoverable=True,
                    confidence=0.9,
                    raw_signal={
                        "rail": rail, "return_code": code,
                        "source": "rbi_ecs_debit_guidelines",
                        "requires_data_correction": True,
                    },
                )

            if code == NACH_INSUFFICIENT_FUNDS_CODE:
                return Diagnosis(
                    root_cause="insufficient_funds",
                    is_recoverable=True,
                    confidence=0.9,
                    raw_signal={"rail": rail, "return_code": code, "source": "rbi_ecs_debit_guidelines"},
                )

            if code == NACH_MISCELLANEOUS_CODE:
                return Diagnosis(
                    root_cause="miscellaneous_return_code",
                    is_recoverable=False,
                    confidence=0.2,
                    raw_signal={"rail": rail, "return_code": code, "source": "rbi_ecs_debit_guidelines"},
                )

            return Diagnosis(
                root_cause="unmapped_nach_return_code",
                is_recoverable=False,
                confidence=0.2,
                raw_signal={"rail": rail, "return_code": code, "source": "unmapped"},
            )

        return Diagnosis(
            root_cause="unmapped_rail",
            is_recoverable=False,
            confidence=0.1,
            raw_signal={"rail": rail, "source": "unmapped"},
        )

    def decide(
        self, case: dict[str, Any], diagnosis: Diagnosis, history: list[dict[str, Any]]
    ) -> Decision:
        rail = case.get("rail")
        attempt_count = _attempt_count(history)

        if diagnosis.confidence < 0.5:
            return Decision(
                action_type=ActionType.ESCALATE,
                reasoning=(
                    f"Rail '{rail}' return code is not in the known taxonomy — "
                    "confidence too low for automated handling."
                ),
                requires_human_review=True,
            )

        if diagnosis.raw_signal.get("requires_data_correction"):
            return Decision(
                action_type=ActionType.ESCALATE,
                reasoning=(
                    "NACH return requires correcting the underlying account "
                    "data (RBI ECS(Debit) reason codes 1-3) before any further "
                    "presentation — not a 'try again later' failure."
                ),
                requires_human_review=True,
            )

        if diagnosis.root_cause == "afa_reauth_required_above_threshold":
            return Decision(
                action_type=ActionType.SWITCH_CHANNEL,
                action_params={
                    "channel": "push_notification",
                    "reason": "manual_upi_pin_reauth_required",
                },
                reasoning=(
                    f"Amount exceeds the Rs {AFA_EXEMPTION_THRESHOLD_INR:,.0f} AFA "
                    "exemption threshold — routine manual re-authentication, not "
                    "an exception, so no human review needed."
                ),
                requires_human_review=False,
            )

        if diagnosis.is_recoverable:
            if rail == "upi_autopay":
                action_params: dict[str, Any] = {"rail": rail}
                if self._learning_core is not None and self._learning_core.has_policy(self.domain_type):
                    # Step 8 bandit wiring: the bandit chooses which backoff
                    # arm to pull for THIS retry, instead of the fixed
                    # escalating schedule — same pattern as subscription's
                    # RETRY_BACKOFF_HOURS bandit selection.
                    arm = self._learning_core.select_arm(self.domain_type)
                    retry_hours = UPI_RETRY_BACKOFF_HOURS[arm]
                    action_params["bandit_arm"] = arm
                else:
                    backoff_index = min(attempt_count, len(UPI_RETRY_BACKOFF_HOURS) - 1)
                    retry_hours = UPI_RETRY_BACKOFF_HOURS[backoff_index]
                action_params["retry_in_hours"] = retry_hours
            else:
                # NACH stays fixed regardless of learning_core — see module
                # docstring's LEARNING-CORE WIRING note.
                action_params = {"rail": rail, "retry_in_hours": NACH_RETRY_HOURS}

            return Decision(
                action_type=ActionType.RETRY,
                action_params=action_params,
                reasoning=f"Recoverable mandate failure ({diagnosis.root_cause}), attempt #{attempt_count + 1}",
                requires_human_review=False,
            )

        return Decision(
            action_type=ActionType.STOP,
            reasoning="Diagnosis marked unrecoverable outside the known stop set.",
            requires_human_review=True,
        )

    def execute(self, case: dict[str, Any], decision: Decision) -> ExecutionResult:
        # Second enforcement point — the same "don't just trust decide()"
        # discipline used in checkout_abandonment.execute() and
        # B2BReceivablesModule.execute(). A silent RETRY above the AFA
        # threshold would violate the RBI e-mandate rule even if it somehow
        # reached execute() — this is the actual last line of defense, not
        # redundant with diagnose()'s branch.
        if (
            decision.action_type == ActionType.RETRY
            and case.get("rail") == "upi_autopay"
            and case.get("amount", 0.0) > AFA_EXEMPTION_THRESHOLD_INR
        ):
            return ExecutionResult(
                success=False,
                compliance_check_passed=False,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        return ExecutionResult(
            success=True,
            compliance_check_passed=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def track_outcome(self, case: dict[str, Any]) -> Outcome:
        simulated = case.get("simulated_mandate_result")
        if simulated == "recovered":
            return Outcome(
                status=OutcomeStatus.RECOVERED, amount_recovered=case.get("amount", 0.0)
            )
        if simulated == "lost":
            return Outcome(status=OutcomeStatus.LOST, amount_recovered=0.0)
        return Outcome(status=OutcomeStatus.PENDING, amount_recovered=0.0)

    def on_promise_due(self, case: dict[str, Any]) -> PromiseOutcome:
        return PromiseOutcome(kept=True)  # mandate retries never produce PROMISED
