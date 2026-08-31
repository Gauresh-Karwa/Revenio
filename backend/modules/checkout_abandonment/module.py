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

    def __init__(self, learning_core: Any = None) -> None:
        """
        learning_core: optional backend.core.learning_core.LearningCore.
        When provided AND it has a policy registered for
        "checkout_abandonment", decide() asks it which channel arm to pull
        instead of the fixed email->sms->in_app escalation order.
        """
        self._learning_core = learning_core

    # Bandit-informed early exit, symmetric to SubscriptionModule's. Scaled
    # down for this domain's much smaller MAX_NUDGES=3 ceiling.
    DIMINISHING_RETURNS_MIN_PULLS_PER_ARM = 20
    DIMINISHING_RETURNS_PROBABILITY_FLOOR = 0.10
    DIMINISHING_RETURNS_MIN_CASE_RETRIES = 1

    def check_stop(
        self, case: dict[str, Any], history: list[dict[str, Any]]
    ) -> StopDecision:
        if not case.get("reached_checkout", False):
            return StopDecision(should_stop=True, stop_reason=StopReason.COST_THRESHOLD)

        if not case.get("opt_in", False):
            return StopDecision(should_stop=True, stop_reason=StopReason.OPT_OUT)

        signal = case.get("abandonment_signal")
        if signal in NON_RECOVERABLE_SIGNALS:
            return StopDecision(should_stop=True, stop_reason=StopReason.COST_THRESHOLD)

        nudge_count = sum(1 for h in history if h.get("_event_type") == "ExecutionResult")

        if (
            self._learning_core is not None
            and self._learning_core.has_policy(self.domain_type)
            and nudge_count >= self.DIMINISHING_RETURNS_MIN_CASE_RETRIES
        ):
            arms = self._learning_core.snapshot()[self.domain_type]["arms"]
            if all(a["pull_count"] >= self.DIMINISHING_RETURNS_MIN_PULLS_PER_ARM for a in arms):
                best_estimate = max(
                    (a.get("mean_estimate", a.get("mean_reward")) or 0.0) for a in arms
                )
                if best_estimate < self.DIMINISHING_RETURNS_PROBABILITY_FLOOR:
                    return StopDecision(should_stop=True, stop_reason=StopReason.DIMINISHING_RETURNS)

        if nudge_count >= MAX_NUDGES:
            return StopDecision(should_stop=True, stop_reason=StopReason.DIMINISHING_RETURNS)

        return StopDecision(should_stop=False)

    def diagnose(
        self, case: dict[str, Any], customer_history: list[dict[str, Any]] | None = None
    ) -> Diagnosis:
        # customer_history is accepted (required by the shared contract, see
        # contract.py's DomainModule.diagnose) but deliberately UNUSED here.
        # This is a documented scope decision, not an oversight: no
        # cross-case behavioral signal (e.g. "this customer has abandoned
        # checkout N times recently") has been built or tested for this
        # domain yet — unlike subscription's customer_recent_failure_pressure,
        # which was built AND empirically validated (see README's step 5
        # section) before being wired in. Extending this domain the same
        # way is a real, open follow-up, tracked in README's open items,
        # not assumed to be either useful or not.
        #
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
            nudge_count = sum(1 for h in history if h.get("_event_type") == "ExecutionResult")

            action_params: dict[str, Any] = {}
            if self._learning_core is not None and self._learning_core.has_policy(self.domain_type):
                arm = self._learning_core.select_arm(self.domain_type)
                channel = NUDGE_CHANNEL_ESCALATION[arm]
                action_params["bandit_arm"] = arm
            else:
                channel_index = min(nudge_count, len(NUDGE_CHANNEL_ESCALATION) - 1)
                channel = NUDGE_CHANNEL_ESCALATION[channel_index]

            action_params["channel"] = channel

            return Decision(
                action_type=ActionType.SWITCH_CHANNEL,
                action_params=action_params,
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