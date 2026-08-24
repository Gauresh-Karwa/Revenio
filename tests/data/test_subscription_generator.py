from backend.data.splitting import entity_level_split
from backend.data.subscription_generator import (
    CODE_BASE_RECOVERY_RATE,
    generate_subscription_dataset,
)


def test_hard_and_stop_codes_never_recover():
    """
    Hard/stop declines never reach a real retry in our system — a leaked
    label here (recovered=True on a hard decline) would be a real bug,
    since it would imply the taxonomy and the generator disagree.
    """
    records = generate_subscription_dataset(n_customers=300, seed=1)
    for r in records:
        if r.decline_code not in CODE_BASE_RECOVERY_RATE:
            assert r.recovered is False


def test_soft_decline_outcomes_are_genuinely_stochastic():
    """
    If a soft code's outcome were deterministic, a model would just be
    re-deriving our own rule lookup, not learning anything. This proves
    real variance exists for at least one soft code.
    """
    records = generate_subscription_dataset(n_customers=2000, seed=2)
    fifty_ones = [r for r in records if r.decline_code == "51"]
    outcomes = {r.recovered for r in fifty_ones}
    assert outcomes == {True, False}, "expected both outcomes to occur, found only one"


def test_aggregate_recovery_rate_lands_in_the_sourced_range():
    """
    Sourced claim: 60-70% of card declines are temporary/recoverable with the
    right strategy. This checks the generator's SOFT-code recovery rate lands
    in a plausible band around that — a sanity check against the source, not
    an exact replication (attempt-decay pulls the raw average down from the
    single-attempt base rates).
    """
    records = generate_subscription_dataset(n_customers=3000, seed=3)
    soft_records = [r for r in records if r.decline_code in CODE_BASE_RECOVERY_RATE]
    recovery_rate = sum(r.recovered for r in soft_records) / len(soft_records)
    assert 0.35 <= recovery_rate <= 0.70, f"soft-code recovery rate {recovery_rate:.2f} outside plausible band"


def test_entity_level_split_has_no_customer_overlap():
    records = generate_subscription_dataset(n_customers=300, seed=4)
    train, val, test = entity_level_split(records)

    train_customers = {r.customer_id for r in train}
    val_customers = {r.customer_id for r in val}
    test_customers = {r.customer_id for r in test}

    assert train_customers.isdisjoint(val_customers)
    assert train_customers.isdisjoint(test_customers)
    assert val_customers.isdisjoint(test_customers)


def test_entity_level_split_preserves_all_records():
    records = generate_subscription_dataset(n_customers=300, seed=5)
    train, val, test = entity_level_split(records)
    assert len(train) + len(val) + len(test) == len(records)


def test_regime_b_override_produces_a_different_distribution():
    """
    Proves the override mechanism for the cross-distribution generalization
    test actually changes the outcome distribution — if it didn't, the
    "different regime" dataset would secretly be identical to regime A, and
    the generalization test built on top of it would be meaningless.
    """
    regime_a = generate_subscription_dataset(n_customers=3000, seed=10)
    regime_b = generate_subscription_dataset(
        n_customers=3000,
        seed=10,
        base_recovery_override={"51": 0.20, "91": 0.30},
        payday_boost_override=1.0,
    )

    a_rate = sum(r.recovered for r in regime_a if r.decline_code == "51") / max(
        1, sum(1 for r in regime_a if r.decline_code == "51")
    )
    b_rate = sum(r.recovered for r in regime_b if r.decline_code == "51") / max(
        1, sum(1 for r in regime_b if r.decline_code == "51")
    )

    assert abs(a_rate - b_rate) > 0.15, "regime B should differ meaningfully from regime A"


def test_default_dataset_is_large_enough_for_a_fair_step5_model_comparison():
    """
    Guards against the default scale silently shrinking back down later.
    A GBM is fine on ~1,000-1,500 records; the neural net and sequence-model
    comparison points (architecture doc 6.3) need meaningfully more, or the
    comparison itself becomes noisy rather than a real signal.
    """
    records = generate_subscription_dataset()  # uses the real default, not a test-only override
    soft_records = [r for r in records if r.decline_code in CODE_BASE_RECOVERY_RATE]
    assert len(soft_records) >= 5000, (
        f"only {len(soft_records)} soft-decline records at default scale — "
        "too thin for a fair model comparison in step 5"
    )