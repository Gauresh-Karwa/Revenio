from backend.core.contract import ActionType, OutcomeStatus, StopReason
from backend.modules.checkout_abandonment.module import CheckoutAbandonmentModule


def make_case(**kwargs):
    base = {"reached_checkout": True, "abandonment_signal": "shipping_cost_surprise", "opt_in": True}
    base.update(kwargs)
    return base


def test_diagnose_marks_non_checkout_starter_unrecoverable():
    module = CheckoutAbandonmentModule()
    diagnosis = module.diagnose(make_case(reached_checkout=False))
    assert diagnosis.is_recoverable is False
    assert diagnosis.root_cause == "never_reached_checkout"


def test_check_stop_halts_non_checkout_starter_before_diagnose_matters():
    module = CheckoutAbandonmentModule()
    result = module.check_stop(make_case(reached_checkout=False), history=[])
    assert result.should_stop is True
    assert result.stop_reason == StopReason.COST_THRESHOLD


def test_check_stop_halts_without_consent():
    module = CheckoutAbandonmentModule()
    result = module.check_stop(make_case(opt_in=False), history=[])
    assert result.should_stop is True
    assert result.stop_reason == StopReason.OPT_OUT


def test_execute_refuses_to_act_without_consent_even_if_reached():
    """
    The real point of this test: execute() is called directly, bypassing
    check_stop entirely, to prove it is NOT trusting an upstream gate —
    it enforces consent itself, as the actual last line of defense.
    """
    module = CheckoutAbandonmentModule()
    case = make_case(opt_in=False)
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    result = module.execute(case, decision)
    assert result.success is False
    assert result.compliance_check_passed is False


def test_execute_succeeds_with_consent():
    module = CheckoutAbandonmentModule()
    case = make_case(opt_in=True)
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    result = module.execute(case, decision)
    assert result.success is True
    assert result.compliance_check_passed is True


def test_check_stop_halts_low_purchase_intent():
    module = CheckoutAbandonmentModule()
    result = module.check_stop(make_case(abandonment_signal="low_purchase_intent"), history=[])
    assert result.should_stop is True
    assert result.stop_reason == StopReason.COST_THRESHOLD


def test_diagnose_marks_low_purchase_intent_unrecoverable():
    module = CheckoutAbandonmentModule()
    diagnosis = module.diagnose(make_case(abandonment_signal="low_purchase_intent"))
    assert diagnosis.is_recoverable is False


def test_diagnose_recoverable_signal():
    module = CheckoutAbandonmentModule()
    diagnosis = module.diagnose(make_case(abandonment_signal="forced_account_creation"))
    assert diagnosis.is_recoverable is True
    assert diagnosis.root_cause == "forced_account_creation"


def test_decide_switches_channel_for_recoverable_case():
    module = CheckoutAbandonmentModule()
    case = make_case()
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    assert decision.action_type == ActionType.SWITCH_CHANNEL
    assert decision.action_params["channel"] == "email"
    assert decision.requires_human_review is False


def test_decide_escalates_channel_on_repeat_nudges():
    module = CheckoutAbandonmentModule()
    case = make_case()
    diagnosis = module.diagnose(case)
    history = [{"compliance_check_passed": True}]
    decision = module.decide(case, diagnosis, history)
    assert decision.action_params["channel"] == "sms"


def test_decide_escalates_unmapped_signal_to_human_review():
    module = CheckoutAbandonmentModule()
    case = make_case(abandonment_signal="something_new")
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    assert decision.requires_human_review is True
    assert decision.action_type == ActionType.ESCALATE


def test_check_stop_halts_after_max_nudges():
    module = CheckoutAbandonmentModule()
    history = [{"compliance_check_passed": True} for _ in range(3)]
    result = module.check_stop(make_case(), history)
    assert result.should_stop is True
    assert result.stop_reason == StopReason.DIMINISHING_RETURNS


def test_track_outcome_defaults_to_pending():
    module = CheckoutAbandonmentModule()
    outcome = module.track_outcome(make_case())
    assert outcome.status == OutcomeStatus.PENDING


def test_track_outcome_simulated_recovered():
    module = CheckoutAbandonmentModule()
    outcome = module.track_outcome(make_case(simulated_nudge_result="recovered", amount=250.0))
    assert outcome.status == OutcomeStatus.RECOVERED
    assert outcome.amount_recovered == 250.0