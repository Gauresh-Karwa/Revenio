from backend.core.bandit_observer import BanditUpdateObserver
from backend.core.events import EventStore
from backend.core.learning_core import LearningCore, StationaryThompsonSampling
from backend.core.orchestrator import Orchestrator
from backend.modules.mandate_retry.module import AFA_EXEMPTION_THRESHOLD_INR, MandateRetryModule


def _make_orchestrator(learning_core=None):
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(MandateRetryModule(learning_core=learning_core))
    return store, orchestrator


def test_upi_stop_code_halts_immediately_no_retry_attempted():
    store, orchestrator = _make_orchestrator()
    final_state = orchestrator.process_case(
        "case-1", "mandate_retry",
        {"rail": "upi_autopay", "return_code": "U_REVOKED", "amount": 500.0},
    )
    assert final_state["terminal_status"] == "STOPPED:OPT_OUT"
    executions = [e for e in store.get_events("case-1") if e.event_type == "ExecutionResult"]
    assert executions == []


def test_upi_soft_code_recovers_when_simulated():
    store, orchestrator = _make_orchestrator()
    case = {"rail": "upi_autopay", "return_code": "U01", "amount": 500.0}
    orchestrator.process_case("case-1", "mandate_retry", case, max_iterations=1)

    outcome_events = [e for e in store.get_events("case-1") if e.event_type == "Outcome"]
    assert outcome_events[-1].payload["status"] == "PENDING"

    case["simulated_mandate_result"] = "recovered"
    final_state = orchestrator.process_case("case-1", "mandate_retry", case, max_iterations=1)
    assert final_state["terminal_status"] == "RECOVERED"


def test_upi_above_afa_threshold_never_reaches_a_silent_retry_execution():
    store, orchestrator = _make_orchestrator()
    case = {
        "rail": "upi_autopay", "return_code": "U01",
        "amount": AFA_EXEMPTION_THRESHOLD_INR + 20000,
    }
    orchestrator.process_case("case-1", "mandate_retry", case, max_iterations=1)

    decisions = [e for e in store.get_events("case-1") if e.event_type == "Decision"]
    assert decisions[0].payload["action_type"] == "SWITCH_CHANNEL"
    assert decisions[0].payload["action_params"]["channel"] == "push_notification"


def test_nach_correction_required_routes_to_human_review_not_execute():
    store, orchestrator = _make_orchestrator()
    final_state = orchestrator.process_case(
        "case-1", "mandate_retry",
        {"rail": "nach", "return_code": "1", "amount": 3000.0},
    )
    events = store.get_events("case-1")
    event_types = [e.event_type for e in events]
    assert "PendingHumanReview" in event_types
    assert "ExecutionResult" not in event_types
    assert final_state["terminal"] is False


def test_nach_mandate_not_received_stops_with_opt_out():
    store, orchestrator = _make_orchestrator()
    final_state = orchestrator.process_case(
        "case-1", "mandate_retry",
        {"rail": "nach", "return_code": "8", "amount": 1000.0},
    )
    assert final_state["terminal_status"] == "STOPPED:OPT_OUT"


# --- Bandit observer wiring (mirrors tests/integration/test_bandit_observer_wiring.py) ---

def _make_wired_system():
    core = LearningCore()
    core.register_policy("mandate_retry", StationaryThompsonSampling(n_arms=3, seed=1))

    store = EventStore()
    store.subscribe(BanditUpdateObserver(core))
    orchestrator = Orchestrator(store)
    orchestrator.register_module(MandateRetryModule(learning_core=core))
    return core, store, orchestrator


def test_decision_event_carries_bandit_arm_when_learning_core_is_wired():
    core, store, orchestrator = _make_wired_system()
    orchestrator.process_case(
        "case-1", "mandate_retry",
        {"rail": "upi_autopay", "return_code": "U01", "simulated_mandate_result": "recovered"},
    )
    decisions = [e for e in store.get_events("case-1") if e.event_type == "Decision"]
    assert "bandit_arm" in decisions[0].payload["action_params"]


def test_bandit_learns_from_a_recovered_upi_case_with_zero_bandit_observer_changes():
    """
    Proves BanditUpdateObserver needed no code changes to support a new
    domain — it's already keyed purely on the generic 'bandit_arm' key and
    the module's own domain_type string.
    """
    core, store, orchestrator = _make_wired_system()
    before = core.snapshot()["mandate_retry"]["arms"]
    assert all(a["pull_count"] == 0 for a in before)

    orchestrator.process_case(
        "case-1", "mandate_retry",
        {"rail": "upi_autopay", "return_code": "U01", "simulated_mandate_result": "recovered"},
    )

    after = core.snapshot()["mandate_retry"]["arms"]
    assert sum(a["pull_count"] for a in after) == 1


def test_nach_case_never_produces_a_bandit_arm_even_when_wired():
    core, store, orchestrator = _make_wired_system()
    orchestrator.process_case(
        "case-1", "mandate_retry",
        {"rail": "nach", "return_code": "NACH_INSUFFICIENT_FUNDS", "simulated_mandate_result": "recovered"},
    )
    decisions = [e for e in store.get_events("case-1") if e.event_type == "Decision"]
    assert "bandit_arm" not in decisions[0].payload["action_params"]
    # No arm chosen -> nothing for the observer to credit.
    assert sum(a["pull_count"] for a in core.snapshot()["mandate_retry"]["arms"]) == 0
