"""
Proves the full, real pipeline: orchestrator -> EventStore.get_customer_case_history
-> SubscriptionModule.diagnose -> customer_recent_failure_pressure.
"""

from backend.core.events import EventStore
from backend.core.orchestrator import Orchestrator
from backend.modules.subscription.module import SubscriptionModule


def _run_one_case(orchestrator, case_id, customer_id, decline_code, simulated_result):
    return orchestrator.process_case(
        case_id, "subscription",
        {
            "decline_code": decline_code,
            "customer_id": customer_id,
            "simulated_retry_result": simulated_result,
        },
    )


def _diagnosis_events(store, case_id):
    return [e for e in store.get_events(case_id) if e.event_type == "Diagnosis"]


def test_first_case_for_a_customer_has_zero_pressure():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule())

    _run_one_case(orchestrator, "case-1", "cust-alpha", "51", "lost")

    diag = _diagnosis_events(store, "case-1")[0]
    assert diag.payload["raw_signal"]["customer_recent_failure_pressure"] == 0.0
    assert diag.payload["raw_signal"]["n_past_cases_considered"] == 0


def test_pressure_from_a_lost_case_carries_into_the_customers_next_case():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule())

    _run_one_case(orchestrator, "case-1", "cust-beta", "51", "lost")
    _run_one_case(orchestrator, "case-2", "cust-beta", "05", "lost")

    diag_2 = _diagnosis_events(store, "case-2")[0]
    assert diag_2.payload["raw_signal"]["n_past_cases_considered"] == 1
    assert diag_2.payload["raw_signal"]["customer_recent_failure_pressure"] > 0.0


def test_different_customers_never_leak_pressure_into_each_other():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule())

    _run_one_case(orchestrator, "case-1", "cust-gamma", "51", "lost")
    _run_one_case(orchestrator, "case-2", "cust-delta", "51", "lost")

    diag_2 = _diagnosis_events(store, "case-2")[0]
    assert diag_2.payload["raw_signal"]["n_past_cases_considered"] == 0
    assert diag_2.payload["raw_signal"]["customer_recent_failure_pressure"] == 0.0


def test_a_recovered_case_does_not_inflate_the_customers_next_pressure():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule())

    _run_one_case(orchestrator, "case-1", "cust-epsilon", "51", "recovered")
    _run_one_case(orchestrator, "case-2", "cust-epsilon", "51", "lost")

    diag_2 = _diagnosis_events(store, "case-2")[0]
    assert diag_2.payload["raw_signal"]["n_past_cases_considered"] == 1
    assert diag_2.payload["raw_signal"]["customer_recent_failure_pressure"] < 0.5


def test_case_with_no_customer_id_still_works_and_stays_neutral():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule())

    final_state = orchestrator.process_case("case-1", "subscription", {"decline_code": "51"})

    assert final_state["exists"] is True
    diag = _diagnosis_events(store, "case-1")[0]
    assert diag.payload["raw_signal"]["customer_recent_failure_pressure"] == 0.0
