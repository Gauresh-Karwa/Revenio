from backend.core.events import EventStore


def test_append_and_get_events():
    store = EventStore()
    store.append("case-1", "dummy", "diagnose", "Diagnosis", {"root_cause": "x"})
    store.append("case-1", "dummy", "decide", "Decision", {"action_type": "RETRY"})

    events = store.get_events("case-1")
    assert len(events) == 2
    assert events[0].stage == "diagnose"
    assert events[1].stage == "decide"


def test_events_are_scoped_per_case():
    store = EventStore()
    store.append("case-1", "dummy", "diagnose", "Diagnosis", {})
    store.append("case-2", "dummy", "diagnose", "Diagnosis", {})

    assert len(store.get_events("case-1")) == 1
    assert len(store.get_events("case-2")) == 1


def test_derive_state_for_unknown_case_returns_not_exists():
    store = EventStore()
    state = store.derive_state("no-such-case")
    assert state["exists"] is False


def test_derive_state_detects_stop_as_terminal():
    store = EventStore()
    store.append(
        "case-1", "dummy", "check_stop", "StopDecision",
        {"should_stop": True, "stop_reason": "RESOLVED"},
    )
    state = store.derive_state("case-1")
    assert state["terminal"] is True
    assert state["terminal_status"] == "STOPPED:RESOLVED"


def test_derive_state_detects_recovered_outcome_as_terminal():
    store = EventStore()
    store.append(
        "case-1", "dummy", "track_outcome", "Outcome",
        {"status": "RECOVERED", "amount_recovered": 500.0},
    )
    state = store.derive_state("case-1")
    assert state["terminal"] is True
    assert state["terminal_status"] == "RECOVERED"