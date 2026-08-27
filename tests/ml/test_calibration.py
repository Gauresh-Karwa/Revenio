import numpy as np

from backend.ml.calibration import calibrate_and_evaluate


def test_calibrate_and_evaluate_runs_and_returns_finite_scores():
    rng = np.random.default_rng(0)
    X_val = rng.random((200, 3))
    y_val = (X_val[:, 0] > 0.5).astype(int)
    X_test = rng.random((200, 3))
    y_test = (X_test[:, 0] > 0.5).astype(int)

    def fake_predict_proba(X):
        return np.clip(X[:, 0], 0.01, 0.99)

    calibrated_model, brier_before, brier_after = calibrate_and_evaluate(
        fake_predict_proba, X_val, y_val, X_test, y_test
    )
    assert np.isfinite(brier_before)
    assert np.isfinite(brier_after)
    assert 0.0 <= brier_before <= 1.0
    assert 0.0 <= brier_after <= 1.0