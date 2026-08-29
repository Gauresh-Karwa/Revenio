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


class DummyModule:
    domain_type = "dummy"

    def check_stop(
        self, case: dict[str, Any], history: list[dict[str, Any]]
    ) -> StopDecision:
        # Stop once we've already gone through one full execute+track cycle.
        execute_count = sum(1 for h in history if h.get("_event_type") == "ExecutionResult")
        if execute_count >= 1:
            return StopDecision(should_stop=True, stop_reason=StopReason.RESOLVED)
        return StopDecision(should_stop=False)

    def diagnose(
        self, case: dict[str, Any], customer_history: list[dict[str, Any]] | None = None
    ) -> Diagnosis:
        return Diagnosis(
            root_cause="dummy_fixed_cause",
            is_recoverable=True,
            confidence=0.99,
            raw_signal={"note": "fixed value, no real model"},
        )

    def decide(
        self, case: dict[str, Any], diagnosis: Diagnosis, history: list[dict[str, Any]]
    ) -> Decision:
        return Decision(
            action_type=ActionType.RETRY,
            action_params={"retry_in_seconds": 1},
            reasoning="dummy module always retries once",
            requires_human_review=False,
        )

    def execute(self, case: dict[str, Any], decision: Decision) -> ExecutionResult:
        return ExecutionResult(
            success=True,
            compliance_check_passed=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def track_outcome(self, case: dict[str, Any]) -> Outcome:
        return Outcome(status=OutcomeStatus.PENDING, amount_recovered=0.0)

    def on_promise_due(self, case: dict[str, Any]) -> PromiseOutcome:
        return PromiseOutcome(kept=True)
