from backend.data.checkout_abandonment_generator import generate_checkout_abandonment_dataset
from backend.data.splitting import entity_level_split


def test_non_checkout_starters_never_recover():
    records = generate_checkout_abandonment_dataset(n_customers=300, seed=1)
    non_starters = [r for r in records if not r.reached_checkout]
    assert len(non_starters) > 0
    assert all(r.recovered is False for r in non_starters)


def test_no_consent_cases_never_recover():
    records = generate_checkout_abandonment_dataset(n_customers=500, seed=2)
    no_consent_recoverable_signal = [
        r for r in records if r.reached_checkout and not r.opt_in and r.abandonment_signal != "n/a"
    ]
    assert len(no_consent_recoverable_signal) > 0
    assert all(r.recovered is False for r in no_consent_recoverable_signal)


def test_low_purchase_intent_never_recovers():
    records = generate_checkout_abandonment_dataset(n_customers=500, seed=3)
    low_intent = [r for r in records if r.abandonment_signal == "low_purchase_intent"]
    assert len(low_intent) > 0
    assert all(r.recovered is False for r in low_intent)


def test_recoverable_signal_outcomes_are_stochastic():
    records = generate_checkout_abandonment_dataset(n_customers=2000, seed=4)
    error_records = [r for r in records if r.abandonment_signal == "checkout_page_error"]
    outcomes = {r.recovered for r in error_records}
    assert outcomes == {True, False}


def test_entity_level_split_no_overlap():
    records = generate_checkout_abandonment_dataset(n_customers=300, seed=5)
    train, val, test = entity_level_split(records)
    train_c = {r.customer_id for r in train}
    val_c = {r.customer_id for r in val}
    test_c = {r.customer_id for r in test}
    assert train_c.isdisjoint(val_c)
    assert train_c.isdisjoint(test_c)
    assert val_c.isdisjoint(test_c)


def test_default_dataset_is_large_enough_for_a_fair_step5_model_comparison():
    records = generate_checkout_abandonment_dataset()  # real default
    recoverable = [r for r in records if r.reached_checkout and r.abandonment_signal != "low_purchase_intent"]
    assert len(recoverable) >= 3000, (
        f"only {len(recoverable)} recoverable checkout-abandonment records at default scale — "
        "too thin for a fair model comparison in step 5"
    )