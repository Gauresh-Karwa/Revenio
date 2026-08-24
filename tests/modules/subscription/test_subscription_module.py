from backend.core.contract import ActionType, OutcomeStatus, StopReason
from backend.modules.subscription.module import SubscriptionModule


def make_case(decline_code, **extra):
    return {"decline_code": decline_code, **extra}


def test_diagnose_soft_decline_is_recoverable():
    module = SubscriptionModule()
    diagnosis = module.diagnose(make_case("51"))
    assert diagnosis.root_cause == "insufficient_funds"
    assert diagnosis.is_recoverable is True
    assert diagnosis.confidence >= 0.9


def test_diagnose_hard_decline_is_not_recoverable():
    module = SubscriptionModule()
    diagnosis = module.diagnose(make_case("43"))
    assert diagnosis.root_cause == "stolen_card"
    assert diagnosis.is_recoverable is False
    assert diagnosis.confidence >= 0.9


def test_diagnose_stop_instruction_code():
    module = SubscriptionModule()
    diagnosis = module.diagnose(make_case("R1"))
    assert diagnosis.root_cause == "customer_stopped_all_recurring"
    assert diagnosis.is_recoverable is False


def test_diagnose_unmapped_code_has_low_confidence():
    module = SubscriptionModule()
    diagnosis = module.diagnose(make_case("99"))
    assert diagnosis.root_cause == "unmapped_decline_code"
    assert diagnosis.confidence < 0.5


def test_check_stop_false_on_soft_decline_first_attempt():
    module = SubscriptionModule()
    result = module.check_stop(make_case("51"), history=[])
    assert result.should_stop is False


def test_check_stop_true_on_hard_decline():
    module = SubscriptionModule()
    result = module.check_stop(make_case("43"), history=[])
    assert result.should_stop is True
    assert result.stop_reason == StopReason.COMPLIANCE_LIMIT


def test_check_stop_true_on_explicit_opt_out():
    module = SubscriptionModule()
    result = module.check_stop(make_case("R0"), history=[])
    assert result.should_stop is True
    assert result.stop_reason == StopReason.OPT_OUT


def test_check_stop_true_after_max_retries():
    module = SubscriptionModule()
    history = [{"compliance_check_passed": True} for _ in range(15)]
    result = module.check_stop(make_case("51"), history=history)
    assert result.should_stop is True
    assert result.stop_reason == StopReason.COMPLIANCE_LIMIT


def test_check_stop_false_just_under_max_retries():
    module = SubscriptionModule()
    history = [{"compliance_check_passed": True} for _ in range(14)]
    result = module.check_stop(make_case("51"), history=history)
    assert result.should_stop is False


def test_decide_retries_soft_decline():
    module = SubscriptionModule()
    diagnosis = module.diagnose(make_case("51"))
    decision = module.decide(make_case("51"), diagnosis, history=[])
    assert decision.action_type == ActionType.RETRY
    assert decision.requires_human_review is False


def test_decide_escalates_unmapped_code_to_human_review():
    module = SubscriptionModule()
    case = make_case("99")
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    assert decision.requires_human_review is True
    assert decision.action_type == ActionType.ESCALATE


def test_decide_backoff_increases_with_retry_count():
    module = SubscriptionModule()
    diagnosis = module.diagnose(make_case("51"))
    first = module.decide(make_case("51"), diagnosis, history=[])
    later_history = [{"compliance_check_passed": True} for _ in range(2)]
    third = module.decide(make_case("51"), diagnosis, later_history)
    assert third.action_params["retry_in_hours"] > first.action_params["retry_in_hours"]


def test_execute_reports_compliance_passed():
    module = SubscriptionModule()
    diagnosis = module.diagnose(make_case("51"))
    decision = module.decide(make_case("51"), diagnosis, history=[])
    result = module.execute(make_case("51"), decision)
    assert result.success is True
    assert result.compliance_check_passed is True


def test_track_outcome_defaults_to_pending():
    module = SubscriptionModule()
    outcome = module.track_outcome(make_case("51"))
    assert outcome.status == OutcomeStatus.PENDING


def test_track_outcome_simulated_recovered():
    module = SubscriptionModule()
    outcome = module.track_outcome(
        make_case("51", simulated_retry_result="recovered", amount=999.0)
    )
    assert outcome.status == OutcomeStatus.RECOVERED
    assert outcome.amount_recovered == 999.0


def test_track_outcome_simulated_lost():
    module = SubscriptionModule()
    outcome = module.track_outcome(make_case("51", simulated_retry_result="lost"))
    assert outcome.status == OutcomeStatus.LOST
    assert outcome.amount_recovered == 0.0
