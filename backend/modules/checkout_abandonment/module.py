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

# --- Session-telemetry taxonomy (architecture doc 5.2) ---

RECOVERABLE_SIGNALS: dict[str, str] = {
    "shipping_cost_surprise": "shipping_cost_surprise",
    "forced_account_creation": "forced_account_creation",
    "payment_method_unavailable": "payment_method_unavailable",
    "checkout_form_friction": "checkout_form_friction",
    "checkout_page_error": "checkout_page_error",
    "distracted_high_intent": "distracted_high_intent",
}

# Deliberately non-recoverable — sourced stopping rule (arch doc 5.2), not
# a default: chasing low-engagement sessions costs more than it recovers.
NON_RECOVERABLE_SIGNALS = {"low_purchase_intent"}

# Reasonable default, explicitly NOT sourced the way the decline-code caps
# were — no Visa-equivalent number exists for "how many nudges is too many."
# Tracked as an open item (architecture doc section 11), not asserted as fact.
MAX_NUDGES = 3

NUDGE_CHANNEL_ESCALATION = ["email", "sms", "in_app"]


class CheckoutAbandonmentModule:
    domain_type = "checkout_abandonment"

    def check_stop(
        self, case: dict[str, Any], history: list[dict[str, Any]]
    ) -> StopDecision:
        # Enforcement point 1a of 2: checkout-starter filter.
        if not case.get("reached_checkout", False):
            return StopDecision(should_stop=True, stop_reason=StopReason.COST_THRESHOLD)

        # Enforcement point 2a of 2: consent gate, before decide/execute ever run.
        if not case.get("opt_in", False):
            return StopDecision(should_stop=True, stop_reason=StopReason.OPT_OUT)

        signal = case.get("abandonment_signal")
        if signal in NON_RECOVERABLE_SIGNALS:
            return StopDecision(should_stop=True, stop_reason=StopReason.COST_THRESHOLD)

        nudge_count = sum(1 for h in history if "compliance_check_passed" in h)
        if nudge_count >= MAX_NUDGES:
            return StopDecision(should_stop=True, stop_reason=StopReason.DIMINISHING_RETURNS)

        return StopDecision(should_stop=False)

    def diagnose(self, case: dict[str, Any]) -> Diagnosis:
        # Enforcement point 1b of 2: checkout-starter filter, restated at the
        # diagnosis level so is_recoverable is correct even if this method is
        # ever called in isolation (e.g. from a future analytics/reporting
        # path) without going through check_stop first.
        if not case.get("reached_checkout", False):
            return Diagnosis(
                root_cause="never_reached_checkout",
                is_recoverable=False,
                confidence=0.99,
                raw_signal={"reached_checkout": False},
            )

        signal = case.get("abandonment_signal")

        if signal in RECOVERABLE_SIGNALS:
            return Diagnosis(
                root_cause=RECOVERABLE_SIGNALS[signal],
                is_recoverable=True,
                confidence=0.9,
                raw_signal={"abandonment_signal": signal, "source": "baymard_taxonomy"},
            )

        if signal in NON_RECOVERABLE_SIGNALS:
            return Diagnosis(
                root_cause=signal,
                is_recoverable=False,
                confidence=0.9,
                raw_signal={"abandonment_signal": signal, "source": "baymard_taxonomy"},
            )

        # Honest handling of an unmapped signal — never guessed.
        return Diagnosis(
            root_cause="unmapped_abandonment_signal",
            is_recoverable=False,
            confidence=0.2,
            raw_signal={"abandonment_signal": signal, "source": "unmapped"},
        )

    def decide(
        self, case: dict[str, Any], diagnosis: Diagnosis, history: list[dict[str, Any]]
    ) -> Decision:
        if diagnosis.confidence < 0.5:
            return Decision(
                action_type=ActionType.ESCALATE,
                reasoning=(
                    f"Abandonment signal '{case.get('abandonment_signal')}' is not in "
                    "the known taxonomy — confidence too low for automated handling."
                ),
                requires_human_review=True,
            )

        if diagnosis.is_recoverable:
            nudge_count = sum(1 for h in history if "compliance_check_passed" in h)
            channel_index = min(nudge_count, len(NUDGE_CHANNEL_ESCALATION) - 1)
            channel = NUDGE_CHANNEL_ESCALATION[channel_index]
            return Decision(
                action_type=ActionType.SWITCH_CHANNEL,
                action_params={"channel": channel},
                reasoning=f"Recoverable abandonment ({diagnosis.root_cause}), nudge #{nudge_count + 1} via {channel}",
                requires_human_review=False,
            )

        return Decision(
            action_type=ActionType.STOP,
            reasoning="Diagnosis marked unrecoverable outside the known stop set.",
            requires_human_review=True,
        )

    def execute(self, case: dict[str, Any], decision: Decision) -> ExecutionResult:
        # Enforcement point 2b of 2 — THE ACTUAL LAST LINE OF DEFENSE.
        # Even if check_stop's consent gate were somehow bypassed or consent
        # changed mid-loop, execute() itself refuses to send anything without
        # opt_in. This is not redundant with check_stop's gate — it's what
        # makes compliance_check_passed mean something concrete rather than
        # being a field that's always True by construction.
        has_consent = case.get("opt_in", False)

        if not has_consent:
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
        # No real checkout-completion webhook exists yet at this checkpoint —
        # same honest gap as the subscription module. A case can carry
        # 'simulated_nudge_result' to exercise every branch in tests.
        simulated = case.get("simulated_nudge_result")
        if simulated == "recovered":
            return Outcome(
                status=OutcomeStatus.RECOVERED, amount_recovered=case.get("amount", 0.0)
            )
        if simulated == "lost":
            return Outcome(status=OutcomeStatus.LOST, amount_recovered=0.0)
        return Outcome(status=OutcomeStatus.PENDING, amount_recovered=0.0)

    def on_promise_due(self, case: dict[str, Any]) -> PromiseOutcome:
        return PromiseOutcome(kept=True)  # checkout abandonment never produces PROMISED