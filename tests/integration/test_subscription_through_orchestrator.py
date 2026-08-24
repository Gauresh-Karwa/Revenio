"""
Proves the subscription module works correctly wired into the REAL
orchestrator (not the dummy module) — the actual integration point of step 2.
"""

from backend.core.events import EventStore
from backend.core.orchestrator import Orchestrator
from backend.modules.subscription.module import SubscriptionModule


def test_hard_decline_stops_immediately_no_retry_attempted():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule())

    final_state = orchestrator.process_case(
        "case-1", "subscription", {"decline_code": "43"}  # stolen card
    )

    assert final_state["terminal_status"] == "STOPPED:COMPLIANCE_LIMIT"
    execute_events = [e for e in store.get_events("case-1") if e.event_type == "ExecutionResult"]
    assert len(execute_events) == 0  # never even attempted


def test_stop_instruction_code_halts_with_opt_out():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule())

    final_state = orchestrator.process_case("case-1", "subscription", {"decline_code": "R1"})

    assert final_state["terminal_status"] == "STOPPED:OPT_OUT"


def test_unmapped_code_routes_to_human_review_not_execute():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule())

    final_state = orchestrator.process_case("case-1", "subscription", {"decline_code": "99"})

    events = store.get_events("case-1")
    event_types = [e.event_type for e in events]
    assert "PendingHumanReview" in event_types
    assert "ExecutionResult" not in event_types
    assert final_state["terminal"] is False


def test_soft_decline_recovers_when_simulated_retry_succeeds():
    """
    Soft decline, first pass: RETRY, PENDING (no real gateway yet).
    Since track_outcome for step 2 checks 'simulated_retry_result', the loop
    only resolves once we simulate that the retry itself came back positive
    on a later processing pass.
    """
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule())

    case = {"decline_code": "51", "amount": 499.0}
    orchestrator.process_case("case-1", "subscription", case, max_iterations=1)

    events_after_first = store.get_events("case-1")
    outcome_events = [e for e in events_after_first if e.event_type == "Outcome"]
    assert outcome_events[-1].payload["status"] == "PENDING"

    case["simulated_retry_result"] = "recovered"
    final_state = orchestrator.process_case("case-1", "subscription", case, max_iterations=1)

    assert final_state["terminal_status"] == "RECOVERED"