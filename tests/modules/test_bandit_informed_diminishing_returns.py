from backend.core.learning_core import LearningCore, StationaryThompsonSampling
from backend.modules.checkout_abandonment.module import CheckoutAbandonmentModule
from backend.modules.subscription.module import SubscriptionModule


def _hopeless_core(domain_type, n_arms):
    """A LearningCore where every arm has been pulled enough times and all
    look genuinely bad — should trigger DIMINISHING_RETURNS."""
    core = LearningCore()
    policy = StationaryThompsonSampling(n_arms=n_arms, seed=1)
    for arm in range(n_arms):
        for _ in range(25):  # > DIMINISHING_RETURNS_MIN_PULLS_PER_ARM
            policy.update(arm, reward=0.0)  # every pull fails
    core.register_policy(domain_type, policy)
    return core


def _promising_core(domain_type, n_arms):
    """Every arm well-explored, but arm 0 looks genuinely good — must NOT trigger."""
    core = LearningCore()
    policy = StationaryThompsonSampling(n_arms=n_arms, seed=1)
    for arm in range(n_arms):
        for _ in range(25):
            reward = 1.0 if arm == 0 else 0.0
            policy.update(arm, reward)
    core.register_policy(domain_type, policy)
    return core


def _under_explored_core(domain_type, n_arms):
    """Arms haven't been pulled enough yet — must NOT trigger even if early signal looks bad."""
    core = LearningCore()
    policy = StationaryThompsonSampling(n_arms=n_arms, seed=1)
    for arm in range(n_arms):
        policy.update(arm, reward=0.0)  # only 1 pull each, far under the threshold
    core.register_policy(domain_type, policy)
    return core


def _history_with_n_execute_results(n):
    return [{"_event_type": "ExecutionResult"} for _ in range(n)]


# --- Subscription ---

def test_diminishing_returns_fires_when_all_arms_are_hopeless():
    core = _hopeless_core("subscription", 4)
    module = SubscriptionModule(learning_core=core)
    case = {"decline_code": "51"}
    history = _history_with_n_execute_results(2)

    decision = module.check_stop(case, history)
    assert decision.should_stop is True
    assert decision.stop_reason.value == "DIMINISHING_RETURNS"


def test_diminishing_returns_does_not_fire_when_an_arm_looks_promising():
    core = _promising_core("subscription", 4)
    module = SubscriptionModule(learning_core=core)
    case = {"decline_code": "51"}
    history = _history_with_n_execute_results(2)

    decision = module.check_stop(case, history)
    assert decision.should_stop is False


def test_diminishing_returns_does_not_fire_before_arms_are_well_explored():
    core = _under_explored_core("subscription", 4)
    module = SubscriptionModule(learning_core=core)
    case = {"decline_code": "51"}
    history = _history_with_n_execute_results(2)

    decision = module.check_stop(case, history)
    assert decision.should_stop is False  # not enough data yet to trust the estimate


def test_diminishing_returns_does_not_fire_before_the_case_has_had_a_fair_shot():
    core = _hopeless_core("subscription", 4)
    module = SubscriptionModule(learning_core=core)
    case = {"decline_code": "51"}
    history = _history_with_n_execute_results(0)  # this case's own first attempt

    decision = module.check_stop(case, history)
    assert decision.should_stop is False  # even hopeless globally, THIS case gets a first try


def test_without_a_learning_core_diminishing_returns_never_fires():
    module = SubscriptionModule()  # no learning_core at all
    case = {"decline_code": "51"}
    history = _history_with_n_execute_results(2)

    decision = module.check_stop(case, history)
    assert decision.should_stop is False


def test_compliance_limit_still_wins_over_diminishing_returns_at_the_cap():
    """The hard MAX_RETRY_ATTEMPTS ceiling must still fire correctly even
    when a learning_core is present — diminishing-returns is an EARLIER,
    additional exit, not a replacement for the compliance cap."""
    core = _promising_core("subscription", 4)  # arms look fine, won't trigger diminishing-returns
    module = SubscriptionModule(learning_core=core)
    case = {"decline_code": "51"}
    history = _history_with_n_execute_results(15)  # at MAX_RETRY_ATTEMPTS

    decision = module.check_stop(case, history)
    assert decision.should_stop is True
    assert decision.stop_reason.value == "COMPLIANCE_LIMIT"


# --- Checkout abandonment ---

def test_abandonment_diminishing_returns_fires_when_hopeless():
    core = _hopeless_core("checkout_abandonment", 3)
    module = CheckoutAbandonmentModule(learning_core=core)
    case = {"reached_checkout": True, "opt_in": True, "abandonment_signal": "shipping_cost_surprise"}
    history = _history_with_n_execute_results(1)

    decision = module.check_stop(case, history)
    assert decision.should_stop is True
    assert decision.stop_reason.value == "DIMINISHING_RETURNS"


def test_abandonment_diminishing_returns_does_not_fire_when_promising():
    core = _promising_core("checkout_abandonment", 3)
    module = CheckoutAbandonmentModule(learning_core=core)
    case = {"reached_checkout": True, "opt_in": True, "abandonment_signal": "shipping_cost_surprise"}
    history = _history_with_n_execute_results(1)

    decision = module.check_stop(case, history)
    assert decision.should_stop is False


def test_abandonment_max_nudges_still_fires_without_learning_core():
    module = CheckoutAbandonmentModule()  # no learning_core
    case = {"reached_checkout": True, "opt_in": True, "abandonment_signal": "shipping_cost_surprise"}
    history = _history_with_n_execute_results(3)  # at MAX_NUDGES

    decision = module.check_stop(case, history)
    assert decision.should_stop is True
    assert decision.stop_reason.value == "DIMINISHING_RETURNS"
