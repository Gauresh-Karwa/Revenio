from backend.data.subscription_generator import (
    generate_subscription_dataset,
    true_recovery_probability,
)


def test_default_generation_has_no_email_or_hardship():
    records = generate_subscription_dataset(n_customers=200, seed=42)
    assert all(r.has_support_email is False for r in records)
    assert all(r.email_text is None for r in records)
    assert all(r.true_hardship is False for r in records)


def test_support_email_signal_generates_emails_and_hardship():
    records = generate_subscription_dataset(
        n_customers=1000, seed=42, include_support_email_signal=True
    )
    with_email = [r for r in records if r.has_support_email]
    with_hardship = [r for r in records if r.true_hardship]

    assert len(with_email) > 0
    assert len(with_hardship) > 0
    assert all(r.email_text is not None for r in with_email)
    assert all(r.has_support_email for r in with_hardship)


def test_hardship_reduces_true_recovery_probability():
    prob_no_hardship = true_recovery_probability(
        decline_code="51", amount=50.0, attempt_number=1, hour_of_day=12,
        is_near_payday=False, has_hardship=False,
    )
    prob_hardship = true_recovery_probability(
        decline_code="51", amount=50.0, attempt_number=1, hour_of_day=12,
        is_near_payday=False, has_hardship=True,
    )

    assert prob_hardship < prob_no_hardship
    assert prob_hardship == prob_no_hardship * 0.55 or prob_hardship == max(0.02, prob_no_hardship * 0.55)


def test_reproducible_with_seed():
    r1 = generate_subscription_dataset(n_customers=100, seed=7, include_support_email_signal=True)
    r2 = generate_subscription_dataset(n_customers=100, seed=7, include_support_email_signal=True)
    assert [r.email_text for r in r1] == [r.email_text for r in r2]
    assert [r.true_hardship for r in r1] == [r.true_hardship for r in r2]
