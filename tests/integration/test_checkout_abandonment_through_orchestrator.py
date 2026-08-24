from backend.core.events import EventStore
from backend.core.orchestrator import Orchestrator
from backend.modules.checkout_abandonment.module import CheckoutAbandonmentModule


def test_non_checkout_starter_never_reaches_execute():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(CheckoutAbandonmentModule())

    final_state = orchestrator.process_case(
        "case-1", "checkout_abandonment", {"reached_checkout": False}
    )

    assert final_state["terminal_status"] == "STOPPED:COST_THRESHOLD"
    execute_events = [e for e in store.get_events("case-1") if e.event_type == "ExecutionResult"]
    assert len(execute_events) == 0


def test_no_consent_never_reaches_execute():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(CheckoutAbandonmentModule())

    final_state = orchestrator.process_case(
        "case-1",
        "checkout_abandonment",
        {"reached_checkout": True, "abandonment_signal": "shipping_cost_surprise", "opt_in": False},
    )

    assert final_state["terminal_status"] == "STOPPED:OPT_OUT"
    execute_events = [e for e in store.get_events("case-1") if e.event_type == "ExecutionResult"]
    assert len(execute_events) == 0


def test_recoverable_case_with_consent_recovers_when_simulated():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(CheckoutAbandonmentModule())

    case = {
        "reached_checkout": True,
        "abandonment_signal": "checkout_form_friction",
        "opt_in": True,
        "amount": 175.0,
    }
    orchestrator.process_case("case-1", "checkout_abandonment", case, max_iterations=1)

    outcome_events = [e for e in store.get_events("case-1") if e.event_type == "Outcome"]
    assert outcome_events[-1].payload["status"] == "PENDING"

    case["simulated_nudge_result"] = "recovered"
    final_state = orchestrator.process_case("case-1", "checkout_abandonment", case, max_iterations=1)

    assert final_state["terminal_status"] == "RECOVERED"