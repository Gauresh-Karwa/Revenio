from backend.modules.b2b_receivables.module import (
    CHANNEL_ESCALATION,
    MAX_BROKEN_PROMISES,
    MAX_CONTACT_ATTEMPTS,
    B2BReceivablesModule,
)


def _module():
    return B2BReceivablesModule()


def _history(n_execute_results=0, broken_promises=0):
    h = [{"_event_type": "ExecutionResult"} for _ in range(n_execute_results)]
    h += [{"_event_type": "PromiseOutcome", "kept": False} for _ in range(broken_promises)]
    return h


# --- check_stop ---

def test_check_stop_halts_on_dnd_registry():
    module = _module()
    decision = module.check_stop({"on_dnd_registry": True}, [])
    assert decision.should_stop is True
    assert decision.stop_reason.value == "OPT_OUT"


def test_check_stop_halts_on_opted_out():
    module = _module()
    decision = module.check_stop({"has_opted_out": True}, [])
    assert decision.should_stop is True
    assert decision.stop_reason.value == "OPT_OUT"


def test_check_stop_halts_on_disputed_invoice():
    module = _module()
    decision = module.check_stop({"is_disputed": True}, [])
    assert decision.should_stop is True
    assert decision.stop_reason.value == "COST_THRESHOLD"


def test_check_stop_halts_after_max_broken_promises():
    module = _module()
    history = _history(broken_promises=MAX_BROKEN_PROMISES)
    decision = module.check_stop({}, history)
    assert decision.should_stop is True
    assert decision.stop_reason.value == "DIMINISHING_RETURNS"


def test_check_stop_does_not_halt_below_max_broken_promises():
    module = _module()
    history = _history(broken_promises=MAX_BROKEN_PROMISES - 1)
    decision = module.check_stop({}, history)
    assert decision.should_stop is False


def test_check_stop_halts_after_max_contact_attempts():
    module = _module()
    history = _history(n_execute_results=MAX_CONTACT_ATTEMPTS)
    decision = module.check_stop({}, history)
    assert decision.should_stop is True
    assert decision.stop_reason.value == "DIMINISHING_RETURNS"


def test_check_stop_false_on_fresh_case():
    module = _module()
    decision = module.check_stop({"invoice_amount": 5000}, [])
    assert decision.should_stop is False


def test_dnd_check_wins_over_broken_promises_and_disputes():
    """Compliance gates are checked first, before any other stop condition."""
    module = _module()
    case = {"on_dnd_registry": True, "is_disputed": True}
    decision = module.check_stop(case, _history(broken_promises=5))
    assert decision.stop_reason.value == "OPT_OUT"


# --- diagnose ---

def test_diagnose_missing_invoice_amount_is_low_confidence():
    module = _module()
    diagnosis = module.diagnose({})
    assert diagnosis.confidence < 0.5
    assert diagnosis.is_recoverable is False


def test_diagnose_valid_invoice_is_recoverable_high_confidence():
    module = _module()
    diagnosis = module.diagnose({"invoice_amount": 10000, "due_date": "2026-07-01"})
    assert diagnosis.is_recoverable is True
    assert diagnosis.confidence >= 0.5
    assert diagnosis.root_cause == "overdue_invoice"
    assert "days_overdue" in diagnosis.raw_signal


def test_diagnose_computes_days_overdue():
    from datetime import date, timedelta

    module = _module()
    due = date.today() - timedelta(days=10)
    diagnosis = module.diagnose({"invoice_amount": 5000, "due_date": due.isoformat()})
    assert diagnosis.raw_signal["days_overdue"] == 10


def test_diagnose_no_msme_registration_has_no_43bh_deadline():
    module = _module()
    diagnosis = module.diagnose({"invoice_amount": 5000, "due_date": "2026-01-01", "is_msme_registered": False})
    assert diagnosis.raw_signal["msme_payment_deadline_days"] is None
    assert "days_until_43bh_deadline" not in diagnosis.raw_signal


def test_diagnose_msme_with_written_agreement_gets_45_day_deadline():
    module = _module()
    diagnosis = module.diagnose({
        "invoice_amount": 5000, "due_date": "2026-01-01",
        "is_msme_registered": True, "has_written_agreement": True,
    })
    assert diagnosis.raw_signal["msme_payment_deadline_days"] == 45


def test_diagnose_msme_without_written_agreement_gets_15_day_deadline():
    module = _module()
    diagnosis = module.diagnose({
        "invoice_amount": 5000, "due_date": "2026-01-01",
        "is_msme_registered": True, "has_written_agreement": False,
    })
    assert diagnosis.raw_signal["msme_payment_deadline_days"] == 15


# --- decide ---

def test_decide_low_confidence_escalates_to_human_review():
    module = _module()
    diagnosis = module.diagnose({})
    decision = module.decide({}, diagnosis, [])
    assert decision.requires_human_review is True
    assert decision.action_type.value == "ESCALATE"


def test_decide_active_promise_waits_instead_of_contacting():
    module = _module()
    case = {"invoice_amount": 5000, "active_promise_date": "2026-08-01"}
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, [])
    assert decision.action_type.value == "WAIT"
    assert decision.requires_human_review is False


def test_decide_first_contact_uses_email():
    module = _module()
    case = {"invoice_amount": 5000, "due_date": "2026-01-01"}
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, [])
    assert decision.action_params["channel"] == "email"


def test_decide_escalates_channel_with_contact_count():
    module = _module()
    case = {"invoice_amount": 5000, "due_date": "2026-01-01"}
    diagnosis = module.diagnose(case)

    decision_1 = module.decide(case, diagnosis, _history(n_execute_results=1))
    assert decision_1.action_params["channel"] == "sms"

    decision_2 = module.decide(case, diagnosis, _history(n_execute_results=2))
    assert decision_2.action_params["channel"] == "voice"


def test_decide_voice_channel_carries_default_locale():
    module = _module()
    case = {"invoice_amount": 5000, "due_date": "2026-01-01"}
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, _history(n_execute_results=2))
    assert decision.action_params["locale"] == "hi-IN"


def test_decide_voice_channel_respects_preferred_locale_override():
    module = _module()
    case = {"invoice_amount": 5000, "due_date": "2026-01-01", "preferred_locale": "ta-IN"}
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, _history(n_execute_results=2))
    assert decision.action_params["locale"] == "ta-IN"


def test_decide_reaching_voice_tier_requires_human_review():
    module = _module()
    case = {"invoice_amount": 5000, "due_date": "2026-01-01"}
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, _history(n_execute_results=2))
    assert decision.action_type.value == "SWITCH_CHANNEL"
    assert decision.action_params["channel"] == "voice"
    assert decision.requires_human_review is True


def test_decide_email_and_sms_tiers_do_not_require_review():
    module = _module()
    case = {"invoice_amount": 5000, "due_date": "2026-01-01"}
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, _history(n_execute_results=0))
    assert decision.requires_human_review is False


# --- execute ---

def test_execute_succeeds_when_not_dnd_blocked():
    module = _module()
    case = {"invoice_amount": 5000, "due_date": "2026-01-01"}
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, [])
    result = module.execute(case, decision)
    assert result.success is True
    assert result.compliance_check_passed is True


def test_execute_fails_closed_if_dnd_somehow_reached():
    """Double-enforcement: even if a case with on_dnd_registry somehow
    reaches execute (shouldn't happen given check_stop), execute refuses."""
    module = _module()
    case = {"invoice_amount": 5000, "on_dnd_registry": True}
    result = module.execute(case, None)
    assert result.success is False
    assert result.compliance_check_passed is False


# --- track_outcome ---

def test_track_outcome_recovered_on_full_payment():
    module = _module()
    case = {"invoice_amount": 5000, "simulated_payment_result": "paid_full"}
    outcome = module.track_outcome(case)
    assert outcome.status.value == "RECOVERED"
    assert outcome.amount_recovered == 5000


def test_track_outcome_promised_carries_promised_date_in_details():
    module = _module()
    case = {"simulated_payment_result": "promised", "promised_date": "2026-08-15"}
    outcome = module.track_outcome(case)
    assert outcome.status.value == "PROMISED"
    assert outcome.details["promised_date"] == "2026-08-15"


def test_track_outcome_written_off_is_lost():
    module = _module()
    case = {"simulated_payment_result": "written_off"}
    outcome = module.track_outcome(case)
    assert outcome.status.value == "LOST"


def test_track_outcome_defaults_to_pending():
    module = _module()
    outcome = module.track_outcome({})
    assert outcome.status.value == "PENDING"


# --- on_promise_due ---

def test_on_promise_due_kept():
    module = _module()
    result = module.on_promise_due({"simulated_promise_kept": True})
    assert result.kept is True


def test_on_promise_due_broken():
    module = _module()
    result = module.on_promise_due({"simulated_promise_kept": False})
    assert result.kept is False


def test_on_promise_due_defaults_to_not_kept():
    module = _module()
    result = module.on_promise_due({})
    assert result.kept is False
