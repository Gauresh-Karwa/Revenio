from backend.core.events import EventStore
from backend.core.orchestrator import Orchestrator
from backend.modules.b2b_receivables.module import B2BReceivablesModule, MAX_BROKEN_PROMISES


def _make_orchestrator():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(B2BReceivablesModule())
    return store, orchestrator


def test_full_case_first_contact_via_email_reaches_pending():
    store, orchestrator = _make_orchestrator()
    case = {"invoice_amount": 10000, "due_date": "2026-01-01"}
    state = orchestrator.process_case("inv-1", "b2b_receivables", case)

    decisions = [e for e in store.get_events("inv-1") if e.event_type == "Decision"]
    assert decisions[0].payload["action_params"]["channel"] == "email"
    assert state["terminal"] is False  # PENDING is not terminal


def test_dnd_case_never_reaches_execute():
    store, orchestrator = _make_orchestrator()
    case = {"invoice_amount": 10000, "on_dnd_registry": True}
    orchestrator.process_case("inv-1", "b2b_receivables", case)

    executions = [e for e in store.get_events("inv-1") if e.event_type == "ExecutionResult"]
    assert executions == []


def test_full_payment_recovers_the_case():
    store, orchestrator = _make_orchestrator()
    case = {
        "invoice_amount": 10000, "due_date": "2026-01-01",
        "simulated_payment_result": "paid_full",
    }
    state = orchestrator.process_case("inv-1", "b2b_receivables", case)
    assert state["terminal_status"] == "RECOVERED"


def test_promise_due_kept_marks_case_recovered():
    store, orchestrator = _make_orchestrator()
    case = {"invoice_amount": 10000, "due_date": "2026-01-01", "simulated_payment_result": "promised"}
    orchestrator.process_case("inv-1", "b2b_receivables", case)  # case now PROMISED, non-terminal

    kept_case = {**case, "simulated_promise_kept": True}
    state = orchestrator.check_promise_due("inv-1", kept_case)

    assert state["terminal_status"] == "RECOVERED"
    promise_events = [e for e in store.get_events("inv-1") if e.event_type == "PromiseOutcome"]
    assert promise_events[-1].payload == {"kept": True}


def test_promise_due_broken_re_enters_check_stop_and_continues_contacting():
    store, orchestrator = _make_orchestrator()
    case = {"invoice_amount": 10000, "due_date": "2026-01-01", "simulated_payment_result": "promised"}
    orchestrator.process_case("inv-1", "b2b_receivables", case)

    broken_case = {**case, "simulated_promise_kept": False}
    state = orchestrator.check_promise_due("inv-1", broken_case)

    promise_events = [e for e in store.get_events("inv-1") if e.event_type == "PromiseOutcome"]
    assert promise_events[-1].payload == {"kept": False}
    # Only ONE broken promise so far — below MAX_BROKEN_PROMISES, so the
    # case should NOT be stopped; it should have continued contacting.
    assert state["terminal"] is False


def test_repeated_broken_promises_eventually_trigger_diminishing_returns():
    """
    The docx's own canonical example of DIMINISHING_RETURNS, proven
    end-to-end: enough broken promises on the SAME case genuinely halt it.
    """
    store, orchestrator = _make_orchestrator()
    case = {"invoice_amount": 10000, "due_date": "2026-01-01", "simulated_payment_result": "promised"}
    orchestrator.process_case("inv-1", "b2b_receivables", case)

    state = None
    for _ in range(MAX_BROKEN_PROMISES):
        broken_case = {**case, "simulated_promise_kept": False}
        state = orchestrator.check_promise_due("inv-1", broken_case)
        # Re-arm the case as PROMISED again for the next iteration if it
        # wasn't stopped, mirroring a customer making (and breaking)
        # another promise.
        if not state["terminal"]:
            orchestrator.process_case(
                "inv-1", "b2b_receivables",
                {**case, "simulated_payment_result": "promised"},
            )

    assert state["terminal"] is True
    assert state["terminal_status"] == "STOPPED:DIMINISHING_RETURNS"


def test_dummy_and_subscription_style_default_on_promise_due_marks_recovered():
    """
    Domains that never produce PROMISED default on_promise_due to
    kept=True (contract.py's documented default) — if check_promise_due
    were ever called on them, they'd resolve to RECOVERED harmlessly, not
    crash. Not something that happens in practice (nothing reaches PROMISED
    for these domains), but the plumbing shouldn't break if it did.
    """
    from backend.modules.dummy.module import DummyModule

    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(DummyModule())
    orchestrator.process_case("case-1", "dummy", {})

    state = orchestrator.check_promise_due("case-1", {})
    assert state["terminal_status"] == "RECOVERED"


def test_check_promise_due_raises_for_unknown_case():
    import pytest

    store, orchestrator = _make_orchestrator()
    with pytest.raises(ValueError):
        orchestrator.check_promise_due("nonexistent", {})
