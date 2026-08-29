"""
The shared contract every domain module implements.

This file has zero domain-specific logic in it, on purpose (see architecture
doc section 2). It only defines the shapes that flow between the orchestrator
and any module — subscription, checkout-abandonment, B2B, whatever comes later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class ActionType(str, Enum):
    RETRY = "RETRY"
    SWITCH_CHANNEL = "SWITCH_CHANNEL"
    ESCALATE = "ESCALATE"
    WAIT = "WAIT"
    STOP = "STOP"


class StopReason(str, Enum):
    COMPLIANCE_LIMIT = "COMPLIANCE_LIMIT"
    OPT_OUT = "OPT_OUT"
    DIMINISHING_RETURNS = "DIMINISHING_RETURNS"
    COST_THRESHOLD = "COST_THRESHOLD"
    RESOLVED = "RESOLVED"


class OutcomeStatus(str, Enum):
    RECOVERED = "RECOVERED"
    PROMISED = "PROMISED"
    LOST = "LOST"
    PENDING = "PENDING"


@dataclass(frozen=True)
class Diagnosis:
    root_cause: str
    is_recoverable: bool
    confidence: float  # 0.0-1.0; categorization certainty ONLY — see below.
    raw_signal: dict[str, Any] = field(default_factory=dict)

    # NEW. Deliberately a separate field from `confidence`, not a rename of
    # it. `confidence` answers "how sure am I this root_cause label is
    # correct" (still ~0.95 for known codes, ~0.2 for unmapped ones — a
    # lookup-table property, unaffected by whether a trained model exists).
    # `predicted_recovery_probability` answers a different question: "given
    # that label, how likely is this specific case to recover." Conflating
    # the two would silently change what requires_human_review's
    # confidence-threshold gate means (decide() in the subscription module
    # gates on confidence, not on this field). None for any code the model
    # was never trained on (hard/stop/unmapped) — scoring those would be
    # meaningless, not just unavailable.
    predicted_recovery_probability: float | None = None


@dataclass(frozen=True)
class Decision:
    action_type: ActionType
    action_params: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    requires_human_review: bool = False


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    stop_reason: StopReason | None = None


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    compliance_check_passed: bool
    timestamp: str  # ISO 8601, set by the module or orchestrator


@dataclass(frozen=True)
class Outcome:
    status: OutcomeStatus
    amount_recovered: float = 0.0


@dataclass(frozen=True)
class PromiseOutcome:
    kept: bool


class DomainModule(Protocol):
    """
    Every domain module (subscription, checkout-abandonment, B2B, ...)
    implements this. The orchestrator only ever calls these five methods
    (plus on_promise_due) and never reaches into a module's internals.

    NAMING NOTE: decide/check_stop's `history` is THIS case's own past
    events (the within-case retry chain — e.g. "how many times has this
    same decline event already been retried"). diagnose's `customer_history`
    is a DIFFERENT, cross-case concept — this customer's OTHER, past cases
    and their outcomes. Conflating the two would be a real bug (see
    backend/core/events.py's get_customer_case_history vs get_events) —
    deliberately different parameter names, not an oversight.
    """

    domain_type: str

    def diagnose(
        self, case: dict[str, Any], customer_history: list[dict[str, Any]] | None = None
    ) -> Diagnosis: ...

    def decide(
        self, case: dict[str, Any], diagnosis: Diagnosis, history: list[dict[str, Any]]
    ) -> Decision: ...

    def check_stop(
        self, case: dict[str, Any], history: list[dict[str, Any]]
    ) -> StopDecision: ...

    def execute(self, case: dict[str, Any], decision: Decision) -> ExecutionResult: ...

    def track_outcome(self, case: dict[str, Any]) -> Outcome: ...

    def on_promise_due(self, case: dict[str, Any]) -> PromiseOutcome:
        return PromiseOutcome(kept=True)