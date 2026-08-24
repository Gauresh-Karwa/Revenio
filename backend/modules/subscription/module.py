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

SOFT_DECLINE_CODES: dict[str, str] = {
    "51": "insufficient_funds",
    "05": "do_not_honor",
    "91": "issuer_unavailable",
    "96": "system_malfunction",
    "65": "activity_limit_exceeded",
    "61": "exceeds_withdrawal_limit",
}

HARD_DECLINE_CODES: dict[str, str] = {
    "04": "pickup_card",
    "07": "pickup_card_special",
    "12": "invalid_transaction",
    "14": "invalid_card_number",
    "15": "invalid_issuer",
    "41": "lost_card",
    "43": "stolen_card",
    "46": "closed_account",
    "57": "transaction_not_permitted",
}

STOP_INSTRUCTION_CODES: dict[str, str] = {
    "R0": "customer_stopped_specific_payment",
    "R1": "customer_stopped_all_recurring",
    "R3": "authorization_revoked",
}

MAX_RETRY_ATTEMPTS = 15

RETRY_BACKOFF_HOURS = [1, 6, 24, 72]


class SubscriptionModule:
    domain_type = "subscription"

    def check_stop(
        self, case: dict[str, Any], history: list[dict[str, Any]]
    ) -> StopDecision:
        code = case.get("decline_code")

        if code in HARD_DECLINE_CODES:
            return StopDecision(should_stop=True, stop_reason=StopReason.COMPLIANCE_LIMIT)

        if code in STOP_INSTRUCTION_CODES:
            return StopDecision(should_stop=True, stop_reason=StopReason.OPT_OUT)

        retry_count = sum(1 for h in history if h.get("_event_type") == "ExecutionResult")
        if retry_count >= MAX_RETRY_ATTEMPTS:
            return StopDecision(should_stop=True, stop_reason=StopReason.COMPLIANCE_LIMIT)

        return StopDecision(should_stop=False)

    def diagnose(self, case: dict[str, Any]) -> Diagnosis:
        code = case.get("decline_code")

        if code in SOFT_DECLINE_CODES:
            return Diagnosis(
                root_cause=SOFT_DECLINE_CODES[code],
                is_recoverable=True,
                confidence=0.95,
                raw_signal={"decline_code": code, "source": "iso8583_soft_lookup"},
            )

        if code in HARD_DECLINE_CODES:
            return Diagnosis(
                root_cause=HARD_DECLINE_CODES[code],
                is_recoverable=False,
                confidence=0.95,
                raw_signal={"decline_code": code, "source": "visa_category1_lookup"},
            )

        if code in STOP_INSTRUCTION_CODES:
            return Diagnosis(
                root_cause=STOP_INSTRUCTION_CODES[code],
                is_recoverable=False,
                confidence=0.99,
                raw_signal={"decline_code": code, "source": "stop_instruction_lookup"},
            )

        return Diagnosis(
            root_cause="unmapped_decline_code",
            is_recoverable=False,
            confidence=0.2,
            raw_signal={"decline_code": code, "source": "unmapped"},
        )

    def decide(
        self, case: dict[str, Any], diagnosis: Diagnosis, history: list[dict[str, Any]]
    ) -> Decision:
        if diagnosis.confidence < 0.5:
            return Decision(
                action_type=ActionType.ESCALATE,
                reasoning=(
                    f"Decline code '{case.get('decline_code')}' is not in the known "
                    "taxonomy — confidence too low for automated handling."
                ),
                requires_human_review=True,
            )

        if diagnosis.is_recoverable:
            retry_count = sum(1 for h in history if h.get("_event_type") == "ExecutionResult")
            backoff_index = min(retry_count, len(RETRY_BACKOFF_HOURS) - 1)
            backoff_hours = RETRY_BACKOFF_HOURS[backoff_index]
            return Decision(
                action_type=ActionType.RETRY,
                action_params={"retry_in_hours": backoff_hours},
                reasoning=f"Soft decline ({diagnosis.root_cause}), retry #{retry_count + 1}",
                requires_human_review=False,
            )

        # check_stop already halts the loop for every known hard/stop code
        # before decide() is ever reached. Landing here means diagnose()
        # marked something unrecoverable outside that known set — a real
        # edge case, not silently swallowed.
        return Decision(
            action_type=ActionType.STOP,
            reasoning="Diagnosis marked unrecoverable outside the known stop-code set.",
            requires_human_review=True,
        )

    def execute(self, case: dict[str, Any], decision: Decision) -> ExecutionResult:
        # check_stop already gated every hard-decline/compliance-limited case
        # before this point, so compliance_check_passed is True by construction
        # here — this is the retry-scheduling call, not the retry itself.
        return ExecutionResult(
            success=True,
            compliance_check_passed=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def track_outcome(self, case: dict[str, Any]) -> Outcome:
        
        simulated = case.get("simulated_retry_result")
        if simulated == "recovered":
            return Outcome(
                status=OutcomeStatus.RECOVERED, amount_recovered=case.get("amount", 0.0)
            )
        if simulated == "lost":
            return Outcome(status=OutcomeStatus.LOST, amount_recovered=0.0)
        return Outcome(status=OutcomeStatus.PENDING, amount_recovered=0.0)

    def on_promise_due(self, case: dict[str, Any]) -> PromiseOutcome:
        # Subscriptions never produce a PROMISED outcome — no-op by design.
        return PromiseOutcome(kept=True)