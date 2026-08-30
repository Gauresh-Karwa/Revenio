"""
Proves the full observer-driven pipeline: EventStore.subscribe ->
BanditUpdateObserver -> LearningCore, through the REAL orchestrator and
modules — not a unit test of BanditUpdateObserver in isolation.
"""

from backend.core.bandit_observer import BanditUpdateObserver
from backend.core.events import EventStore
from backend.core.learning_core import DriftAwareThompsonSampling, LearningCore, StationaryThompsonSampling
from backend.core.orchestrator import Orchestrator
from backend.modules.checkout_abandonment.module import CheckoutAbandonmentModule
from backend.modules.subscription.module import SubscriptionModule


def _make_wired_system(subscription_arms=4, abandonment_arms=3):
    core = LearningCore()
    core.register_policy("subscription", StationaryThompsonSampling(n_arms=subscription_arms, seed=1))
    core.register_policy("checkout_abandonment", StationaryThompsonSampling(n_arms=abandonment_arms, seed=2))

    store = EventStore()
    observer = BanditUpdateObserver(core)
    store.subscribe(observer)

    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule(learning_core=core))
    orchestrator.register_module(CheckoutAbandonmentModule(learning_core=core))

    return core, store, orchestrator


def test_decision_event_carries_bandit_arm_when_learning_core_is_wired():
    core, store, orchestrator = _make_wired_system()
    orchestrator.process_case(
        "case-1", "subscription",
        {"decline_code": "51", "simulated_retry_result": "recovered"},
    )
    decisions = [e for e in store.get_events("case-1") if e.event_type == "Decision"]
    assert len(decisions) >= 1
    assert "bandit_arm" in decisions[0].payload["action_params"]


def test_bandit_learns_from_a_recovered_case():
    core, store, orchestrator = _make_wired_system()
    before = core.snapshot()["subscription"]["arms"]
    assert all(a["pull_count"] == 0 for a in before)

    orchestrator.process_case(
        "case-1", "subscription",
        {"decline_code": "51", "simulated_retry_result": "recovered"},
    )

    after = core.snapshot()["subscription"]["arms"]
    assert sum(a["pull_count"] for a in after) == 1


def test_bandit_learns_from_a_lost_case_too():
    core, store, orchestrator = _make_wired_system()
    orchestrator.process_case(
        "case-1", "subscription",
        {"decline_code": "51", "simulated_retry_result": "lost"},
    )
    after = core.snapshot()["subscription"]["arms"]
    assert sum(a["pull_count"] for a in after) == 1


def test_pending_case_does_not_update_the_bandit_yet():
    """A case that stays PENDING (never resolves) must not be credited/blamed."""
    core, store, orchestrator = _make_wired_system()
    orchestrator.process_case(
        "case-1", "subscription", {"decline_code": "51"}, max_iterations=1,
    )
    after = core.snapshot()["subscription"]["arms"]
    assert sum(a["pull_count"] for a in after) == 0


def test_pooling_subscription_and_abandonment_share_one_learning_core_without_interference():
    core, store, orchestrator = _make_wired_system()

    orchestrator.process_case(
        "sub-case-1", "subscription",
        {"decline_code": "51", "simulated_retry_result": "recovered"},
    )
    orchestrator.process_case(
        "abandon-case-1", "checkout_abandonment",
        {
            "reached_checkout": True,
            "opt_in": True,
            "abandonment_signal": "shipping_cost_surprise",
            "simulated_nudge_result": "recovered",
        },
    )

    snap = core.snapshot()
    assert sum(a["pull_count"] for a in snap["subscription"]["arms"]) == 1
    assert sum(a["pull_count"] for a in snap["checkout_abandonment"]["arms"]) == 1


def test_without_a_learning_core_behavior_is_completely_unaffected():
    """Backward compatibility: no learning_core -> original fixed schedule, no bandit_arm key."""
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule())  # no learning_core

    orchestrator.process_case(
        "case-1", "subscription",
        {"decline_code": "51", "simulated_retry_result": "recovered"},
    )
    decisions = [e for e in store.get_events("case-1") if e.event_type == "Decision"]
    assert "bandit_arm" not in decisions[0].payload["action_params"]
    assert decisions[0].payload["action_params"]["retry_in_hours"] == 1


def test_drift_aware_bandit_can_be_wired_in_place_of_stationary():
    core = LearningCore()
    core.register_policy("subscription", DriftAwareThompsonSampling(n_arms=4, discount_factor=0.9, seed=1))
    store = EventStore()
    store.subscribe(BanditUpdateObserver(core))
    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule(learning_core=core))

    orchestrator.process_case(
        "case-1", "subscription",
        {"decline_code": "51", "simulated_retry_result": "recovered"},
    )
    assert sum(a["pull_count"] for a in core.snapshot()["subscription"]["arms"]) == 1
