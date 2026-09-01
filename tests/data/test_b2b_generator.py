from backend.data.b2b_generator import (
    AGING_BUCKET_FLOOR,
    AGING_BUCKETS,
    MSME_TAX_INCENTIVE_FACTOR,
    generate_b2b_dataset,
    true_payment_probability,
)


def test_dnd_opted_out_disputed_never_recover():
    records = generate_b2b_dataset(n_customers=3000, seed=1)
    blocked = [r for r in records if r.on_dnd_registry or r.has_opted_out or r.is_disputed]
    assert len(blocked) > 0
    assert all(r.recovered is False for r in blocked)


def test_aging_buckets_land_in_sourced_ranges():
    """
    Locks in the finding verified against real, multi-source AR-aging
    benchmarks (NACM/CCAA-cited, Crestmont, Eagle Rock CFO) — see
    b2b_generator.py's module docstring for the citations.
    """
    records = generate_b2b_dataset(n_customers=5000, seed=42)
    chased = [r for r in records if not (r.on_dnd_registry or r.has_opted_out or r.is_disputed)]

    def rate_in_bucket(lo, hi):
        bucket = [r for r in chased if lo <= r.days_overdue <= hi]
        assert len(bucket) > 50
        return sum(r.recovered for r in bucket) / len(bucket)

    assert rate_in_bucket(1, 30) > 0.90        # sourced: >95% current, generous lower bound
    assert 0.80 <= rate_in_bucket(31, 60) <= 0.95   # sourced: ~85-90%
    assert 0.65 <= rate_in_bucket(61, 90) <= 0.85    # sourced: ~70-80%, NACM/CCAA
    assert 0.40 <= rate_in_bucket(91, 120) <= 0.65   # sourced: ~50-60%
    assert 0.10 <= rate_in_bucket(121, 999) <= 0.40  # sourced: ~20-30%


def test_recovery_rate_strictly_decreases_with_aging():
    records = generate_b2b_dataset(n_customers=5000, seed=7)
    chased = [r for r in records if not (r.on_dnd_registry or r.has_opted_out or r.is_disputed)]

    buckets = [(1, 30), (31, 60), (61, 90), (91, 120), (121, 999)]
    rates = []
    for lo, hi in buckets:
        bucket = [r for r in chased if lo <= r.days_overdue <= hi]
        rates.append(sum(r.recovered for r in bucket) / len(bucket))

    assert rates == sorted(rates, reverse=True)  # strictly monotonic decay with age


def test_msme_registration_boosts_recovery_probability():
    # days_overdue=15 sits in the 0-30 "current" bucket, whose base rate
    # (0.97) is already at the probability ceiling — a x1.12 boost has no
    # room to show up there (0.97 * 1.12 clamps back to 0.97). That's a
    # realistic property, not a bug: a near-certain payment can't get much
    # more certain regardless of mechanism. Use a later bucket instead,
    # where the base rate has real headroom below the ceiling.
    p_msme = true_payment_probability(days_overdue=75, is_msme_registered=True)
    p_non_msme = true_payment_probability(days_overdue=75, is_msme_registered=False)
    assert p_msme > p_non_msme
    assert p_msme == p_non_msme * MSME_TAX_INCENTIVE_FACTOR


def test_msme_effect_is_observable_in_sampled_data_within_a_matched_bucket():
    """
    Not just a check on the pure probability function — confirms the
    effect actually shows up in SAMPLED outcomes, isolated to a single
    aging bucket so aging itself doesn't confound the comparison.
    """
    records = generate_b2b_dataset(n_customers=6000, seed=42)
    chased = [r for r in records if not (r.on_dnd_registry or r.has_opted_out or r.is_disputed)]
    matched = [r for r in chased if 61 <= r.days_overdue <= 90]

    msme = [r for r in matched if r.is_msme_registered]
    non_msme = [r for r in matched if not r.is_msme_registered]
    assert len(msme) > 30 and len(non_msme) > 30

    msme_rate = sum(r.recovered for r in msme) / len(msme)
    non_msme_rate = sum(r.recovered for r in non_msme) / len(non_msme)
    assert msme_rate > non_msme_rate


def test_customer_pressure_reduces_recovery_probability():
    p_neutral = true_payment_probability(15, False, customer_recent_payment_pressure=0.0)
    p_pressured = true_payment_probability(15, False, customer_recent_payment_pressure=1.0)
    assert p_pressured < p_neutral


def test_probability_bounds_always_respected():
    for days in [0, 30, 60, 90, 120, 200, 500]:
        for msme in [True, False]:
            for pressure in [0.0, 0.5, 1.0]:
                p = true_payment_probability(days, msme, pressure)
                assert 0.02 <= p <= 0.97


def test_default_call_has_no_customer_history_effect():
    records = generate_b2b_dataset(n_customers=500, seed=2)
    assert all(r.customer_recent_payment_pressure == 0.0 for r in records)


def test_include_customer_history_produces_nonzero_pressure():
    records = generate_b2b_dataset(n_customers=3000, seed=3, include_customer_history=True)
    assert any(r.customer_recent_payment_pressure > 0.0 for r in records)


def test_invoice_amounts_are_positive_and_right_skewed():
    records = generate_b2b_dataset(n_customers=2000, seed=4)
    amounts = sorted(r.invoice_amount for r in records)
    assert all(a > 0 for a in amounts)
    median = amounts[len(amounts) // 2]
    mean = sum(amounts) / len(amounts)
    assert mean > median  # right-skewed, as intended by the lognormal


def test_entity_level_split_reusable_without_modification():
    from backend.data.splitting import entity_level_split

    records = generate_b2b_dataset(n_customers=1000, seed=5)
    train, val, test = entity_level_split(records)
    assert len(train) + len(val) + len(test) == len(records)

    train_customers = {r.customer_id for r in train}
    val_customers = {r.customer_id for r in val}
    test_customers = {r.customer_id for r in test}
    assert train_customers.isdisjoint(val_customers)
    assert train_customers.isdisjoint(test_customers)
    assert val_customers.isdisjoint(test_customers)


def test_deterministic_given_seed():
    a = generate_b2b_dataset(n_customers=500, seed=99)
    b = generate_b2b_dataset(n_customers=500, seed=99)
    assert [r.recovered for r in a] == [r.recovered for r in b]
    assert [r.days_overdue for r in a] == [r.days_overdue for r in b]


def test_default_dataset_is_large_enough_for_a_fair_model_comparison():
    """
    Same discipline as subscription/checkout-abandonment: fails loudly if
    the default scale ever shrinks without anyone deciding to.
    """
    records = generate_b2b_dataset()
    chased = [r for r in records if not (r.on_dnd_registry or r.has_opted_out or r.is_disputed)]
    assert len(chased) >= 3000
