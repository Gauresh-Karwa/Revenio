from backend.data.subscription_generator import (
    compute_pressure_from_customer_history,
    generate_subscription_retry_sequences,
    update_causal_pressure,
)


def test_empty_history_is_perfectly_neutral():
    assert compute_pressure_from_customer_history([]) == 0.0


def test_single_recovery_pulls_pressure_toward_zero():
    pressure = compute_pressure_from_customer_history([True])
    assert pressure == update_causal_pressure(0.0, True)
    assert pressure < 0.5


def test_single_loss_pulls_pressure_toward_one():
    pressure = compute_pressure_from_customer_history([False])
    assert pressure == update_causal_pressure(0.0, False)
    assert pressure > 0.0


def test_matches_the_generators_own_trajectory_exactly():
    cases = generate_subscription_retry_sequences(n_customers=500, seed=11)

    by_customer: dict[str, list] = {}
    for c in cases:
        by_customer.setdefault(c.customer_id, []).append(c)

    checked_a_multi_case_customer = False
    for customer_id, customer_cases in by_customer.items():
        outcomes_so_far: list[bool] = []
        for case in customer_cases:
            recomputed = compute_pressure_from_customer_history(outcomes_so_far)
            assert recomputed == case.customer_recent_failure_pressure
            outcomes_so_far.append(case.final_recovered)
            if len(customer_cases) > 1:
                checked_a_multi_case_customer = True

    assert checked_a_multi_case_customer
