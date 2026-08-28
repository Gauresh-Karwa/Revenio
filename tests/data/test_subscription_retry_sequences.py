from backend.data.subscription_generator import (
    CODE_BASE_RECOVERY_RATE,
    HISTORY_EWMA_ALPHA,
    MAX_CHAIN_ATTEMPTS,
    MIN_HISTORY_FACTOR,
    _customer_history_factor,
    generate_subscription_retry_sequences,
)


def test_only_soft_codes_produce_sequences():
    cases = generate_subscription_retry_sequences(n_customers=2000, seed=1)
    assert len(cases) > 0
    assert all(c.decline_code in CODE_BASE_RECOVERY_RATE for c in cases)


def test_chain_stops_at_first_recovery():
    cases = generate_subscription_retry_sequences(n_customers=3000, seed=2)
    recovered_cases = [c for c in cases if c.final_recovered]
    assert len(recovered_cases) > 0
    for c in recovered_cases:
        assert all(not a.recovered for a in c.attempts[:-1])
        assert c.attempts[-1].recovered is True


def test_chain_never_exceeds_max_chain_attempts():
    cases = generate_subscription_retry_sequences(n_customers=3000, seed=3)
    assert all(len(c.attempts) <= MAX_CHAIN_ATTEMPTS for c in cases)
    assert any(len(c.attempts) == MAX_CHAIN_ATTEMPTS for c in cases)


def test_attempt_numbers_are_a_real_incrementing_chain_not_independent_draws():
    cases = generate_subscription_retry_sequences(n_customers=2000, seed=4)
    for c in cases:
        numbers = [a.attempt_number for a in c.attempts]
        assert numbers == list(range(1, len(numbers) + 1))


def test_unrecovered_chain_ends_at_max_attempts_all_failed():
    cases = generate_subscription_retry_sequences(n_customers=3000, seed=5)
    unrecovered = [c for c in cases if not c.final_recovered]
    assert len(unrecovered) > 0
    for c in unrecovered:
        assert len(c.attempts) == MAX_CHAIN_ATTEMPTS
        assert all(not a.recovered for a in c.attempts)


def test_deterministic_given_seed():
    cases_a = generate_subscription_retry_sequences(n_customers=500, seed=99)
    cases_b = generate_subscription_retry_sequences(n_customers=500, seed=99)
    assert [c.final_recovered for c in cases_a] == [c.final_recovered for c in cases_b]
    assert [len(c.attempts) for c in cases_a] == [len(c.attempts) for c in cases_b]


def test_customer_history_factor_bounds():
    assert _customer_history_factor(0.0) == 1.0
    assert _customer_history_factor(1.0) == MIN_HISTORY_FACTOR
    assert _customer_history_factor(0.25) > _customer_history_factor(0.75)


def test_first_case_for_every_customer_has_zero_pressure():
    cases = generate_subscription_retry_sequences(n_customers=2000, seed=6)
    seen_customers = set()
    for c in cases:
        if c.customer_id not in seen_customers:
            assert c.customer_recent_failure_pressure == 0.0
            seen_customers.add(c.customer_id)


def test_pressure_is_causal_matches_ewma_of_prior_outcomes_only():
    cases = generate_subscription_retry_sequences(n_customers=1000, seed=7)

    by_customer: dict[str, list] = {}
    for c in cases:
        by_customer.setdefault(c.customer_id, []).append(c)

    for customer_id, customer_cases in by_customer.items():
        expected_pressure = 0.0
        for case in customer_cases:
            assert case.customer_recent_failure_pressure == expected_pressure
            outcome_signal = 0.0 if case.final_recovered else 1.0
            expected_pressure = (
                HISTORY_EWMA_ALPHA * outcome_signal + (1.0 - HISTORY_EWMA_ALPHA) * expected_pressure
            )


def test_pressure_stays_within_unit_interval():
    cases = generate_subscription_retry_sequences(n_customers=3000, seed=8)
    assert all(0.0 <= c.customer_recent_failure_pressure <= 1.0 for c in cases)


def test_customer_history_effect_is_statistically_present_in_sampled_outcomes():
    cases = generate_subscription_retry_sequences(n_customers=8000, max_cases_per_customer=4, seed=9)

    high_pressure = [c for c in cases if c.customer_recent_failure_pressure > 0.5]
    low_pressure = [c for c in cases if c.customer_recent_failure_pressure < 0.1]

    assert len(high_pressure) > 20
    assert len(low_pressure) > 20

    high_rate = sum(c.final_recovered for c in high_pressure) / len(high_pressure)
    low_rate = sum(c.final_recovered for c in low_pressure) / len(low_pressure)

    assert high_rate < low_rate
