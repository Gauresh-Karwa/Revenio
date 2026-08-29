from backend.core.events import EventStore


def test_get_customer_case_history_returns_only_this_customers_events():
    store = EventStore()
    store.append("case-1", "subscription", "diagnose", "Diagnosis", {}, customer_id="cust-1")
    store.append("case-2", "subscription", "diagnose", "Diagnosis", {}, customer_id="cust-2")

    history = store.get_customer_case_history("cust-1")
    assert len(history) == 1
    assert history[0].case_id == "case-1"


def test_get_customer_case_history_excludes_current_case():
    store = EventStore()
    store.append("case-1", "subscription", "diagnose", "Diagnosis", {}, customer_id="cust-1")
    store.append("case-2", "subscription", "diagnose", "Diagnosis", {}, customer_id="cust-1")

    history = store.get_customer_case_history("cust-1", exclude_case_id="case-2")
    assert len(history) == 1
    assert history[0].case_id == "case-1"


def test_get_customer_case_history_spans_multiple_past_cases_in_order():
    store = EventStore()
    store.append("case-1", "subscription", "track_outcome", "Outcome",
                  {"status": "LOST"}, customer_id="cust-1")
    store.append("case-2", "subscription", "track_outcome", "Outcome",
                  {"status": "RECOVERED"}, customer_id="cust-1")

    history = store.get_customer_case_history("cust-1", exclude_case_id="case-3")
    statuses = [e.payload["status"] for e in history]
    assert statuses == ["LOST", "RECOVERED"]


def test_events_without_customer_id_are_never_returned():
    store = EventStore()
    store.append("case-1", "subscription", "diagnose", "Diagnosis", {})

    assert store.get_customer_case_history("cust-1") == []


def test_get_events_unaffected_by_customer_id_addition():
    store = EventStore()
    store.append("case-1", "subscription", "diagnose", "Diagnosis", {}, customer_id="cust-1")
    store.append("case-2", "subscription", "diagnose", "Diagnosis", {}, customer_id="cust-1")

    assert len(store.get_events("case-1")) == 1
