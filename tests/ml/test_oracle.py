from backend.data.subscription_generator import generate_subscription_dataset
from backend.ml.oracle import compute_oracle_ceiling


def test_oracle_ceiling_is_a_valid_auc():
    records = generate_subscription_dataset(n_customers=2000, seed=7)
    auc, n = compute_oracle_ceiling(records)
    assert 0.5 <= auc <= 1.0
    assert n > 0


def test_oracle_ceiling_beats_a_naive_constant_predictor():
    """
    Sanity check: the oracle (which uses the generator's true per-row
    probability) must do meaningfully better than a model with zero
    information — otherwise something is wrong with the oracle itself,
    not just "the ceiling is low."
    """
    records = generate_subscription_dataset(n_customers=3000, seed=8)
    auc, _ = compute_oracle_ceiling(records)
    assert auc > 0.55
