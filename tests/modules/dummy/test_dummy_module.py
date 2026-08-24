from backend.core.contract import ActionType, OutcomeStatus, StopReason
from backend.modules.dummy.module import DummyModule


def test_check_stop_false_on_empty_history():
    module = DummyModule()
    result = module.check_stop(case={}, history=[])
    assert result.should_stop is False


def test_check_stop_true_after_one_execute_cycle():
    module = DummyModule()
    history = [{"compliance_check_passed": True}]
    result = module.check_stop(case={}, history=history)
    assert result.should_stop is True
    assert result.stop_reason == StopReason.RESOLVED


def test_diagnose_returns_recoverable_with_confidence():
    module = DummyModule()
    diagnosis = module.diagnose(case={})
    assert diagnosis.is_recoverable is True
    assert 0.0 <= diagnosis.confidence <= 1.0


def test_decide_returns_retry_without_human_review():
    module = DummyModule()
    decision = module.decide(case={}, diagnosis=module.diagnose({}), history=[])
    assert decision.action_type == ActionType.RETRY
    assert decision.requires_human_review is False


def test_execute_reports_success_and_compliance_passed():
    module = DummyModule()
    decision = module.decide(case={}, diagnosis=module.diagnose({}), history=[])
    result = module.execute(case={}, decision=decision)
    assert result.success is True
    assert result.compliance_check_passed is True


def test_track_outcome_is_pending():
    module = DummyModule()
    outcome = module.track_outcome(case={})
    assert outcome.status == OutcomeStatus.PENDING


def test_on_promise_due_default_noop():
    module = DummyModule()
    result = module.on_promise_due(case={})
    assert result.kept is True