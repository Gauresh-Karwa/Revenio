import numpy as np
import pytest

from backend.core.learning_core import (
    DriftAwareThompsonSampling,
    LearningCore,
    StaticHeuristicPolicy,
    StationaryThompsonSampling,
    ThompsonSamplingBandit,
)


def test_static_heuristic_always_selects_fixed_arm_regardless_of_rewards():
    policy = StaticHeuristicPolicy(n_arms=4, fixed_arm=1)
    for _ in range(50):
        assert policy.select_arm() == 1
        policy.update(arm=1, reward=0.0)
    for _ in range(50):
        assert policy.select_arm() == 1


def test_static_heuristic_rejects_invalid_fixed_arm():
    with pytest.raises(ValueError):
        StaticHeuristicPolicy(n_arms=3, fixed_arm=3)


def test_thompson_sampling_rejects_non_positive_priors():
    with pytest.raises(ValueError):
        ThompsonSamplingBandit(n_arms=3, prior_alpha=0.0)
    with pytest.raises(ValueError):
        ThompsonSamplingBandit(n_arms=3, prior_beta=-1.0)


def test_thompson_sampling_rejects_discount_factor_out_of_range():
    with pytest.raises(ValueError):
        ThompsonSamplingBandit(n_arms=3, discount_factor=0.0)
    with pytest.raises(ValueError):
        ThompsonSamplingBandit(n_arms=3, discount_factor=1.5)


def test_thompson_sampling_rejects_invalid_window_size():
    with pytest.raises(ValueError):
        ThompsonSamplingBandit(n_arms=3, window_size=0)


def test_update_rejects_reward_outside_unit_interval():
    policy = StationaryThompsonSampling(n_arms=2, seed=0)
    with pytest.raises(ValueError):
        policy.update(arm=0, reward=1.5)
    with pytest.raises(ValueError):
        policy.update(arm=0, reward=-0.1)


def test_update_rejects_out_of_range_arm():
    policy = StationaryThompsonSampling(n_arms=2, seed=0)
    with pytest.raises(ValueError):
        policy.update(arm=2, reward=0.5)


def test_drift_aware_requires_a_real_drift_mechanism():
    with pytest.raises(ValueError):
        DriftAwareThompsonSampling(n_arms=3)


def test_thompson_sampling_rejects_discount_and_window_combined():
    with pytest.raises(ValueError):
        ThompsonSamplingBandit(n_arms=3, discount_factor=0.9, window_size=10)


def test_drift_aware_accepts_discount_only():
    DriftAwareThompsonSampling(n_arms=3, discount_factor=0.9)


def test_drift_aware_accepts_window_only():
    DriftAwareThompsonSampling(n_arms=3, window_size=20)


@pytest.mark.parametrize("policy_factory", [
    lambda: StaticHeuristicPolicy(n_arms=5, fixed_arm=2),
    lambda: StationaryThompsonSampling(n_arms=5, seed=1),
    lambda: DriftAwareThompsonSampling(n_arms=5, discount_factor=0.95, seed=1),
    lambda: DriftAwareThompsonSampling(n_arms=5, window_size=10, seed=1),
])
def test_select_arm_returns_a_valid_index(policy_factory):
    policy = policy_factory()
    for _ in range(20):
        arm = policy.select_arm()
        assert 0 <= arm < 5


def test_stationary_thompson_sampling_converges_toward_the_better_arm():
    rng = np.random.default_rng(42)
    true_probs = [0.2, 0.8]
    policy = StationaryThompsonSampling(n_arms=2, seed=42)

    for _ in range(500):
        arm = policy.select_arm()
        reward = float(rng.random() < true_probs[arm])
        policy.update(arm, reward)

    later_selections = [policy.select_arm() for _ in range(200)]
    assert later_selections.count(1) / len(later_selections) > 0.85


def test_discount_factor_keeps_pseudo_counts_bounded_unlike_stationary():
    seed = 7
    stationary = StationaryThompsonSampling(n_arms=1, seed=seed)
    drift_aware = DriftAwareThompsonSampling(n_arms=1, discount_factor=0.9, seed=seed)

    for _ in range(200):
        stationary.update(arm=0, reward=1.0)
        drift_aware.update(arm=0, reward=1.0)

    assert stationary.alpha[0] > 190
    assert drift_aware.alpha[0] < 15


def test_sliding_window_forgets_observations_outside_the_window():
    policy = DriftAwareThompsonSampling(n_arms=1, window_size=5, seed=0)
    for _ in range(5):
        policy.update(arm=0, reward=1.0)
    assert policy.alpha[0] == pytest.approx(6.0)

    for _ in range(5):
        policy.update(arm=0, reward=0.0)
    assert policy.alpha[0] == pytest.approx(1.0)
    assert policy.beta[0] == pytest.approx(6.0)


def test_drift_aware_recovers_faster_than_stationary_after_a_regime_change():
    seed = 123
    rng = np.random.default_rng(seed)

    n_rounds_per_regime = 300
    regime_a_probs = [0.8, 0.2]
    regime_b_probs = [0.2, 0.8]

    stationary = StationaryThompsonSampling(n_arms=2, seed=seed)
    drift_aware = DriftAwareThompsonSampling(n_arms=2, discount_factor=0.95, seed=seed)

    def run_regime(policy, true_probs, n_rounds):
        for _ in range(n_rounds):
            arm = policy.select_arm()
            reward = float(rng.random() < true_probs[arm])
            policy.update(arm, reward)

    run_regime(stationary, regime_a_probs, n_rounds_per_regime)
    run_regime(drift_aware, regime_a_probs, n_rounds_per_regime)

    post_change_rounds = 100

    def post_change_selection_rate_for_new_best(policy, true_probs):
        selections = []
        for _ in range(post_change_rounds):
            arm = policy.select_arm()
            reward = float(rng.random() < true_probs[arm])
            policy.update(arm, reward)
            selections.append(arm)
        return selections.count(1) / len(selections)

    stationary_rate = post_change_selection_rate_for_new_best(stationary, regime_b_probs)
    drift_aware_rate = post_change_selection_rate_for_new_best(drift_aware, regime_b_probs)

    assert drift_aware_rate > stationary_rate


@pytest.mark.parametrize("policy_factory", [
    lambda: StaticHeuristicPolicy(n_arms=3, fixed_arm=1),
    lambda: StationaryThompsonSampling(n_arms=3, seed=5),
    lambda: DriftAwareThompsonSampling(n_arms=3, discount_factor=0.9, seed=5),
    lambda: DriftAwareThompsonSampling(n_arms=3, window_size=10, seed=5),
])
def test_serialization_round_trip_preserves_state(policy_factory):
    policy = policy_factory()
    for i in range(10):
        policy.update(arm=i % 3, reward=float(i % 2))

    data = policy.to_dict()
    restored = type(policy).from_dict(data)

    assert restored.to_dict() == data
    if hasattr(policy, "_rng"):
        assert policy.select_arm() == restored.select_arm()


def test_learning_core_keeps_domains_independent():
    core = LearningCore()
    core.register_policy("subscription", StationaryThompsonSampling(n_arms=4, seed=1))
    core.register_policy("checkout_abandonment", StaticHeuristicPolicy(n_arms=3, fixed_arm=0))

    core.update("subscription", arm=2, reward=1.0)
    snap = core.snapshot()

    assert snap["subscription"]["arms"][2]["pull_count"] == 1
    assert all(a["pull_count"] == 0 for a in snap["checkout_abandonment"]["arms"])


def test_learning_core_missing_domain_raises_not_silently_falls_back():
    core = LearningCore()
    core.register_policy("subscription", StationaryThompsonSampling(n_arms=4, seed=1))

    with pytest.raises(KeyError):
        core.select_arm("checkout_abandonment")
    with pytest.raises(KeyError):
        core.update("checkout_abandonment", arm=0, reward=1.0)


def test_learning_core_one_domain_works_with_no_others_registered():
    core = LearningCore()
    core.register_policy("subscription", StationaryThompsonSampling(n_arms=4, seed=1))
    arm = core.select_arm("subscription")
    assert 0 <= arm < 4
    core.update("subscription", arm=arm, reward=1.0)
