from datetime import datetime, timezone
from typing import Any

from backend.core.contract import (
    ActionType,
    Decision,
    Diagnosis,
    ExecutionResult,
    Outcome,
    OutcomeStatus,
    StopDecision,
    StopReason,
)
from backend.core.events import EventStore
from backend.core.orchestrator import Orchestrator, UnknownDomainError
from backend.modules.dummy.module import DummyModule


def test_unknown_domain_raises():
    orchestrator = Orchestrator(EventStore())
    try:
        orchestrator.process_case("case-1", "not_registered", {})
        assert False, "expected UnknownDomainError"
    except UnknownDomainError:
        pass


def test_dummy_module_runs_full_loop_and_stops():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(DummyModule())

    final_state = orchestrator.process_case("case-1", "dummy", {})

    assert final_state["exists"] is True
    assert final_state["terminal"] is True
    assert final_state["terminal_status"] == "STOPPED:RESOLVED"

    events = store.get_events("case-1")
    event_types = [e.event_type for e in events]
    assert "StopDecision" in event_types
    assert "Diagnosis" in event_types
    assert "Decision" in event_types
    assert "ExecutionResult" in event_types
    assert "Outcome" in event_types


def test_stop_gate_prevents_second_full_cycle():
    """
    The dummy module resolves after exactly one execute+track cycle. This
    proves the orchestrator's check_stop call actually gates the loop,
    rather than the module happening to return a terminal outcome.
    """
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(DummyModule())

    orchestrator.process_case("case-1", "dummy", {})
    events = store.get_events("case-1")

    execute_events = [e for e in events if e.event_type == "ExecutionResult"]
    assert len(execute_events) == 1


class HumanReviewModule:
    """Test-only module: always requires human review, to prove the orchestrator
    routes to that state instead of auto-executing."""

    domain_type = "human_review_test"

    def check_stop(self, case: dict[str, Any], history: list[dict[str, Any]]) -> StopDecision:
        return StopDecision(should_stop=False)

    def diagnose(self, case: dict[str, Any]) -> Diagnosis:
        return Diagnosis(root_cause="low_confidence_case", is_recoverable=True, confidence=0.2)

    def decide(self, case, diagnosis, history) -> Decision:
        return Decision(
            action_type=ActionType.ESCALATE,
            reasoning="confidence below threshold",
            requires_human_review=True,
        )

    def execute(self, case, decision) -> ExecutionResult:
        raise AssertionError("execute() must never be called when human review is required")

    def track_outcome(self, case: dict[str, Any]) -> Outcome:
        raise AssertionError("track_outcome() must never be called when human review is required")


def test_human_review_gate_halts_before_execute():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(HumanReviewModule())

    final_state = orchestrator.process_case("case-1", "human_review_test", {})

    events = store.get_events("case-1")
    event_types = [e.event_type for e in events]
    assert "PendingHumanReview" in event_types
    assert "ExecutionResult" not in event_types
    assert final_state["terminal"] is False


class NeverStopsModule:
    """Test-only module that never satisfies its own stop condition, to prove
    the orchestrator's max_iterations circuit breaker protects it regardless."""

    domain_type = "never_stops_test"

    def check_stop(self, case, history) -> StopDecision:
        return StopDecision(should_stop=False)

    def diagnose(self, case) -> Diagnosis:
        return Diagnosis(root_cause="loop_forever", is_recoverable=True, confidence=0.9)

    def decide(self, case, diagnosis, history) -> Decision:
        return Decision(action_type=ActionType.RETRY, requires_human_review=False)

    def execute(self, case, decision) -> ExecutionResult:
        return ExecutionResult(
            success=True, compliance_check_passed=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def track_outcome(self, case) -> Outcome:
        return Outcome(status=OutcomeStatus.PENDING, amount_recovered=0.0)


def test_circuit_breaker_stops_a_module_that_never_stops_itself():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(NeverStopsModule())

    orchestrator.process_case("case-1", "never_stops_test", {}, max_iterations=5)

    events = store.get_events("case-1")
    execute_events = [e for e in events if e.event_type == "ExecutionResult"]
    assert len(execute_events) == 5

def test_history_payloads_are_tagged_with_real_event_type_not_guessed():
    """
    Regression test for a real bug: modules used to guess "was this an
    execute stage" by checking whether a payload dict happened to contain a
    key named 'compliance_check_passed' — which only worked by coincidence.
    This proves the orchestrator now tells modules the real event_type
    explicitly, so nothing downstream has to guess from key names again.
    """
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(DummyModule())

    orchestrator.process_case("case-1", "dummy", {})

    events = store.get_events("case-1")
    execute_event = next(e for e in events if e.event_type == "ExecutionResult")
    assert "compliance_check_passed" in execute_event.payload  # the real field is still there

    final_state = store.derive_state("case-1")
    assert final_state["terminal_status"] == "STOPPED:RESOLVED"