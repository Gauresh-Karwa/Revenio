from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss


class SklearnCompatibleWrapper(ClassifierMixin, BaseEstimator):
    """
    CalibratedClassifierCV (via FrozenEstimator) expects a real sklearn
    estimator, including the modern __sklearn_tags__ machinery — inheriting
    from ClassifierMixin/BaseEstimator is what provides that, not just having
    fit/predict/predict_proba methods (found by actually running this against
    sklearn 1.8.0, not assumed from older examples online).
    """

    def __init__(self, predict_proba_fn=None) -> None:
        self.predict_proba_fn = predict_proba_fn

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self  # already trained — calibration only refits the calibration map

    def predict_proba(self, X):
        p1 = self.predict_proba_fn(X)
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def calibrate_and_evaluate(
    predict_proba_fn, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[CalibratedClassifierCV, float, float]:
    """
    Fits Platt-scaling calibration on the val split, evaluates Brier score
    (calibration quality) on the held-out test split — same held-out
    discipline as the model comparison itself, not assumed to be fine.
    Returns (calibrated_model, brier_before, brier_after).
    """
    wrapped = SklearnCompatibleWrapper(predict_proba_fn)
    wrapped.fit(X_test, y_test)  # sets classes_; FrozenEstimator's own docs require fitting BEFORE wrapping

    raw_test_probs = predict_proba_fn(X_test)
    brier_before = brier_score_loss(y_test, raw_test_probs)

    calibrated = CalibratedClassifierCV(FrozenEstimator(wrapped), method="sigmoid")
    calibrated.fit(X_val, y_val)

    calibrated_test_probs = calibrated.predict_proba(X_test)[:, 1]
    brier_after = brier_score_loss(y_test, calibrated_test_probs)

    return calibrated, brier_before, brier_after