from backend.data.subscription_generator import generate_subscription_dataset
from backend.ml.features import build_feature_matrix, records_to_frame


def test_records_to_frame_filters_to_soft_codes_only():
    records = generate_subscription_dataset(n_customers=300, seed=1)
    df = records_to_frame(records)
    hard_stop_codes = {"04", "07", "12", "14", "15", "41", "43", "46", "57", "R0", "R1", "R3"}
    assert not set(df["decline_code"]).intersection(hard_stop_codes)


def test_feature_matrix_never_contains_customer_id():
    records = generate_subscription_dataset(n_customers=300, seed=2)
    df = records_to_frame(records)
    X, y, feature_names, customer_ids = build_feature_matrix(df)

    assert "customer_id" not in feature_names
    assert X.shape[0] == len(df)
    assert X.shape[1] == len(feature_names)
    assert len(customer_ids) == len(df)


def test_feature_matrix_labels_are_binary():
    records = generate_subscription_dataset(n_customers=300, seed=3)
    df = records_to_frame(records)
    _, y, _, _ = build_feature_matrix(df)
    assert set(y.tolist()).issubset({0, 1})