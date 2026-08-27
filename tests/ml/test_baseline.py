from backend.data.subscription_generator import generate_subscription_dataset
from backend.ml.features import records_to_frame
from backend.ml.models.baseline import RuleBasedBaseline


def test_baseline_predictions_are_valid_probabilities():
    records = generate_subscription_dataset(n_customers=500, seed=1)
    df = records_to_frame(records)
    baseline = RuleBasedBaseline().fit(df)
    probs = baseline.predict_proba(df)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_baseline_uses_per_code_rates_not_one_global_number():
    records = generate_subscription_dataset(n_customers=1000, seed=2)
    df = records_to_frame(records)
    baseline = RuleBasedBaseline().fit(df)
    assert baseline._code_rates["91"] > baseline._code_rates["05"]