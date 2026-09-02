from backend.core.contract import ActionType, OutcomeStatus, StopReason
from backend.core.learning_core import StationaryThompsonSampling, LearningCore
from backend.modules.mandate_retry.module import (
    AFA_EXEMPTION_THRESHOLD_INR,
    MAX_NACH_PRESENTATIONS,
    MAX_UPI_AUTOPAY_ATTEMPTS,
    UPI_RETRY_BACKOFF_HOURS,
    MandateRetryModule,
)


def _history(n_execute_results):
    return [{"_event_type": "ExecutionResult"} for _ in range(n_execute_results)]


def upi_case(code="U01", amount=500.0, **extra):
    return {"rail": "upi_autopay", "return_code": code, "amount": amount, **extra}


def nach_case(code="NACH_INSUFFICIENT_FUNDS", amount=500.0, **extra):
    return {"rail": "nach", "return_code": code, "amount": amount, **extra}


# --- UPI Autopay: diagnose ---

def test_diagnose_upi_soft_code_is_recoverable():
    module = MandateRetryModule()
    diagnosis = module.diagnose(upi_case("U01"))
    assert diagnosis.root_cause == "insufficient_funds"
    assert diagnosis.is_recoverable is True
    assert diagnosis.confidence >= 0.9


def test_diagnose_upi_stop_code_is_not_recoverable():
    module = MandateRetryModule()
    diagnosis = module.diagnose(upi_case("U_REVOKED"))
    assert diagnosis.root_cause == "mandate_revoked_by_customer"
    assert diagnosis.is_recoverable is False


def test_diagnose_upi_unmapped_code_has_low_confidence():
    module = MandateRetryModule()
    diagnosis = module.diagnose(upi_case("U99"))
    assert diagnosis.root_cause == "unmapped_upi_autopay_code"
    assert diagnosis.confidence < 0.5


def test_diagnose_upi_above_afa_threshold_overrides_soft_code():
    module = MandateRetryModule()
    diagnosis = module.diagnose(upi_case("U01", amount=AFA_EXEMPTION_THRESHOLD_INR + 1))
    assert diagnosis.root_cause == "afa_reauth_required_above_threshold"
    assert diagnosis.is_recoverable is True


def test_diagnose_upi_at_exactly_threshold_does_not_trigger_afa():
    module = MandateRetryModule()
    diagnosis = module.diagnose(upi_case("U01", amount=AFA_EXEMPTION_THRESHOLD_INR))
    assert diagnosis.root_cause == "insufficient_funds"


# --- UPI Autopay: check_stop ---

def test_check_stop_false_on_soft_upi_code_first_attempt():
    module = MandateRetryModule()
    result = module.check_stop(upi_case("U01"), history=[])
    assert result.should_stop is False


def test_check_stop_true_on_upi_stop_code():
    module = MandateRetryModule()
    result = module.check_stop(upi_case("U_PAUSED"), history=[])
    assert result.should_stop is True
    assert result.stop_reason == StopReason.OPT_OUT


def test_check_stop_true_at_max_upi_attempts():
    module = MandateRetryModule()
    result = module.check_stop(upi_case("U01"), _history(MAX_UPI_AUTOPAY_ATTEMPTS))
    assert result.should_stop is True
    assert result.stop_reason == StopReason.COMPLIANCE_LIMIT


def test_check_stop_false_just_under_max_upi_attempts():
    module = MandateRetryModule()
    result = module.check_stop(upi_case("U01"), _history(MAX_UPI_AUTOPAY_ATTEMPTS - 1))
    assert result.should_stop is False


# --- UPI Autopay: decide (fixed schedule, no learning_core) ---

def test_decide_retries_soft_upi_code():
    module = MandateRetryModule()
    case = upi_case("U01")
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    assert decision.action_type == ActionType.RETRY
    assert decision.action_params["rail"] == "upi_autopay"
    assert "bandit_arm" not in decision.action_params
    assert decision.requires_human_review is False


def test_decide_backoff_increases_with_attempt_count():
    module = MandateRetryModule()
    case = upi_case("U01")
    diagnosis = module.diagnose(case)
    first = module.decide(case, diagnosis, history=[])
    later = module.decide(case, diagnosis, _history(2))
    assert later.action_params["retry_in_hours"] > first.action_params["retry_in_hours"]


def test_decide_above_afa_threshold_switches_channel_not_human_review():
    module = MandateRetryModule()
    case = upi_case("U01", amount=AFA_EXEMPTION_THRESHOLD_INR + 5000)
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    assert decision.action_type == ActionType.SWITCH_CHANNEL
    assert decision.action_params["channel"] == "push_notification"
    assert decision.requires_human_review is False


def test_decide_escalates_unmapped_upi_code():
    module = MandateRetryModule()
    case = upi_case("U99")
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    assert decision.action_type == ActionType.ESCALATE
    assert decision.requires_human_review is True


# --- UPI Autopay: bandit wiring ---

def test_decide_uses_bandit_arm_when_learning_core_has_policy():
    core = LearningCore()
    core.register_policy("mandate_retry", StationaryThompsonSampling(n_arms=3, seed=1))
    module = MandateRetryModule(learning_core=core)
    case = upi_case("U01")
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    assert "bandit_arm" in decision.action_params
    assert decision.action_params["retry_in_hours"] in UPI_RETRY_BACKOFF_HOURS


def test_decide_ignores_learning_core_without_registered_policy():
    core = LearningCore()  # no "mandate_retry" policy registered
    module = MandateRetryModule(learning_core=core)
    case = upi_case("U01")
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    assert "bandit_arm" not in decision.action_params


def test_decide_nach_retry_is_unaffected_by_learning_core():
    """NACH deliberately never consults the bandit — see module docstring."""
    core = LearningCore()
    core.register_policy("mandate_retry", StationaryThompsonSampling(n_arms=3, seed=1))
    module = MandateRetryModule(learning_core=core)
    case = nach_case("NACH_INSUFFICIENT_FUNDS")
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    assert "bandit_arm" not in decision.action_params
    assert decision.action_params["retry_in_hours"] == 24


def _hopeless_core():
    core = LearningCore()
    policy = StationaryThompsonSampling(n_arms=3, seed=1)
    for arm in range(3):
        for _ in range(25):
            policy.update(arm, reward=0.0)
    core.register_policy("mandate_retry", policy)
    return core


def _promising_core():
    core = LearningCore()
    policy = StationaryThompsonSampling(n_arms=3, seed=1)
    for arm in range(3):
        for _ in range(25):
            policy.update(arm, reward=1.0 if arm == 0 else 0.0)
    core.register_policy("mandate_retry", policy)
    return core


def test_diminishing_returns_fires_on_upi_when_bandit_is_hopeless():
    module = MandateRetryModule(learning_core=_hopeless_core())
    result = module.check_stop(upi_case("U01"), _history(2))
    assert result.should_stop is True
    assert result.stop_reason == StopReason.DIMINISHING_RETURNS


def test_diminishing_returns_does_not_fire_when_an_arm_looks_promising():
    module = MandateRetryModule(learning_core=_promising_core())
    result = module.check_stop(upi_case("U01"), _history(2))
    assert result.should_stop is False


def test_diminishing_returns_never_fires_on_nach_even_with_hopeless_bandit():
    """The bandit is scoped to UPI only — NACH's check_stop must never consult it."""
    module = MandateRetryModule(learning_core=_hopeless_core())
    result = module.check_stop(nach_case("NACH_INSUFFICIENT_FUNDS"), _history(2))
    assert result.should_stop is False


# --- UPI Autopay: execute (second enforcement point) ---

def test_execute_refuses_silent_retry_above_afa_threshold():
    from backend.core.contract import Decision

    module = MandateRetryModule()
    case = upi_case("U01", amount=AFA_EXEMPTION_THRESHOLD_INR + 1)
    forced_retry = Decision(
        action_type=ActionType.RETRY,
        action_params={"rail": "upi_autopay", "retry_in_hours": 24},
    )
    result = module.execute(case, forced_retry)
    assert result.success is False
    assert result.compliance_check_passed is False


def test_execute_succeeds_for_normal_retry_below_threshold():
    module = MandateRetryModule()
    case = upi_case("U01")
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    result = module.execute(case, decision)
    assert result.success is True
    assert result.compliance_check_passed is True


# --- NACH: diagnose ---

def test_diagnose_nach_insufficient_funds_is_recoverable():
    module = MandateRetryModule()
    diagnosis = module.diagnose(nach_case("NACH_INSUFFICIENT_FUNDS"))
    assert diagnosis.root_cause == "insufficient_funds"
    assert diagnosis.is_recoverable is True


def test_diagnose_nach_correction_required_code_is_recoverable_but_flagged():
    module = MandateRetryModule()
    diagnosis = module.diagnose(nach_case("1"))
    assert diagnosis.root_cause == "account_data_correction_required"
    assert diagnosis.is_recoverable is True
    assert diagnosis.raw_signal["requires_data_correction"] is True


def test_diagnose_nach_mandate_not_received_is_not_recoverable():
    module = MandateRetryModule()
    diagnosis = module.diagnose(nach_case("8"))
    assert diagnosis.root_cause == "mandate_not_received"
    assert diagnosis.is_recoverable is False


def test_diagnose_nach_miscellaneous_has_low_confidence():
    module = MandateRetryModule()
    diagnosis = module.diagnose(nach_case("9"))
    assert diagnosis.confidence < 0.5


# --- NACH: check_stop ---

def test_check_stop_true_on_nach_mandate_not_received():
    module = MandateRetryModule()
    result = module.check_stop(nach_case("8"), history=[])
    assert result.should_stop is True
    assert result.stop_reason == StopReason.OPT_OUT


def test_check_stop_false_on_nach_correction_required_code():
    module = MandateRetryModule()
    result = module.check_stop(nach_case("1"), history=[])
    assert result.should_stop is False


def test_check_stop_true_at_max_nach_presentations():
    module = MandateRetryModule()
    result = module.check_stop(nach_case("NACH_INSUFFICIENT_FUNDS"), _history(MAX_NACH_PRESENTATIONS))
    assert result.should_stop is True
    assert result.stop_reason == StopReason.COMPLIANCE_LIMIT


# --- NACH: decide ---

def test_decide_escalates_nach_correction_required_code():
    module = MandateRetryModule()
    case = nach_case("2")
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    assert decision.action_type == ActionType.ESCALATE
    assert decision.requires_human_review is True
    assert "correct" in decision.reasoning.lower()


def test_decide_retries_nach_insufficient_funds():
    module = MandateRetryModule()
    case = nach_case("NACH_INSUFFICIENT_FUNDS")
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    assert decision.action_type == ActionType.RETRY
    assert decision.action_params["rail"] == "nach"


# --- track_outcome / on_promise_due ---

def test_track_outcome_defaults_to_pending():
    module = MandateRetryModule()
    outcome = module.track_outcome(upi_case())
    assert outcome.status == OutcomeStatus.PENDING


def test_track_outcome_simulated_recovered():
    module = MandateRetryModule()
    outcome = module.track_outcome({**upi_case(), "simulated_mandate_result": "recovered", "amount": 9999.0})
    assert outcome.status == OutcomeStatus.RECOVERED
    assert outcome.amount_recovered == 9999.0


def test_on_promise_due_defaults_kept_true_noop():
    module = MandateRetryModule()
    result = module.on_promise_due(upi_case())
    assert result.kept is True


# --- Unknown rail ---

def test_diagnose_unknown_rail_is_low_confidence():
    module = MandateRetryModule()
    diagnosis = module.diagnose({"rail": "something_new"})
    assert diagnosis.root_cause == "unmapped_rail"
    assert diagnosis.confidence < 0.5


def test_decide_escalates_unknown_rail():
    module = MandateRetryModule()
    case = {"rail": "something_new"}
    diagnosis = module.diagnose(case)
    decision = module.decide(case, diagnosis, history=[])
    assert decision.requires_human_review is True
