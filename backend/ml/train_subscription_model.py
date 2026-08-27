from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.data.splitting import entity_level_split
from backend.data.subscription_generator import (
    CODE_BASE_RECOVERY_RATE,
    generate_subscription_dataset,
)
from backend.ml.features import (
    FEATURE_NAMES,
    build_feature_matrix,
    build_feature_vector,
    fit_scaler,
    records_to_frame,
)
from backend.ml.models.gbm import tune_gbm
from backend.ml.models.neural_net import _predict_proba, train_final_nn, tune_nn

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "subscription_winner.joblib"

BUNDLE_SCHEMA_VERSION = 2


class PyTorchMLPEstimator(ClassifierMixin, BaseEstimator):
    """
    Sklearn-compatible estimator wrapper for ConfigurableMLP, allowing it
    to plug directly into sklearn Pipelines and CalibratedClassifierCV.
    """

    def __init__(
        self,
        n_layers: int = 1,
        width_multiplier: int = 2,
        lr: float = 0.01,
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
        epochs: int = 150,
    ) -> None:
        self.n_layers = n_layers
        self.width_multiplier = width_multiplier
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PyTorchMLPEstimator":
        self.classes_ = np.unique(y)
        params = {
            "n_layers": self.n_layers,
            "width_multiplier": self.width_multiplier,
            "lr": self.lr,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
        }
        self.model_ = train_final_nn(np.asarray(X), np.asarray(y), params, epochs=self.epochs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = _predict_proba(self.model_, np.asarray(X))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _records_to_matrix(records) -> tuple[np.ndarray, np.ndarray]:
    soft_records = [r for r in records if r.decline_code in CODE_BASE_RECOVERY_RATE]
    X = np.array(
        [
            build_feature_vector(
                decline_code=r.decline_code,
                attempt_number=r.attempt_number,
                hour_of_day=r.hour_of_day,
                is_near_payday=r.is_near_payday,
                amount=r.amount,
            )
            for r in soft_records
        ]
    )
    y = np.array([1.0 if r.recovered else 0.0 for r in soft_records])
    return X, y


def _build_gbm_pipeline(best_params: dict) -> Pipeline:
    clf = XGBClassifier(
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        n_estimators=best_params["n_estimators"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        min_child_weight=best_params["min_child_weight"],
        reg_lambda=best_params["reg_lambda"],
        eval_metric="logloss",
        verbosity=0,
    )
    return Pipeline(
        steps=[
            ("clf", CalibratedClassifierCV(clf, method="sigmoid", cv=3)),
        ]
    )


def _build_mlp_pipeline(best_params: dict) -> Pipeline:
    clf = PyTorchMLPEstimator(
        n_layers=best_params["n_layers"],
        width_multiplier=best_params["width_multiplier"],
        lr=best_params["lr"],
        dropout=best_params["dropout"],
        weight_decay=best_params["weight_decay"],
        epochs=150,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(clf, method="sigmoid", cv=3)),
        ]
    )


def train(
    n_gbm_iter: int = 25,
    n_nn_iter: int = 15,
    n_splits: int = 5,
    seed: int = 42,
    show_progress: bool = True,
) -> dict:
    records = generate_subscription_dataset()
    train_records, val_records, test_records = entity_level_split(records, seed=seed)

    train_df = records_to_frame(train_records)
    X_train, y_train, _, groups_train = build_feature_matrix(train_df)

    X_val, y_val = _records_to_matrix(val_records)
    X_test, y_test = _records_to_matrix(test_records)

    # --- Hyperparameter search matching compare.py ---
    gbm_params, gbm_cv_auc = tune_gbm(
        X_train,
        y_train,
        groups_train,
        n_splits=n_splits,
        n_iter=n_gbm_iter,
        seed=seed,
        show_progress=show_progress,
    )

    nn_scaler = fit_scaler(X_train)
    X_train_nn = nn_scaler.transform(X_train)

    nn_params, nn_cv_auc = tune_nn(
        X_train_nn,
        y_train,
        groups_train,
        n_splits=n_splits,
        n_iter=n_nn_iter,
        seed=seed,
        show_progress=show_progress,
    )

    candidates = {
        "GBM": (_build_gbm_pipeline(gbm_params), gbm_params),
        "MLP": (_build_mlp_pipeline(nn_params), nn_params),
    }

    results: dict[str, dict] = {}
    for name, (pipeline, params) in candidates.items():
        pipeline.fit(X_train, y_train)
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        results[name] = {
            "pipeline": pipeline,
            "params": params,
            "val_auc": roc_auc_score(y_val, val_probs),
            "val_brier": brier_score_loss(y_val, val_probs),
        }

    # Winner is the candidate with the highest val AUC. Results are reproducible
    # run-to-run because tune_gbm and tune_nn both use the same fixed seed —
    # no tie-break rule needed. If MLP leads on val AUC, MLP wins; if GBM leads,
    # GBM wins. The margin is recorded in the bundle for audit visibility.
    winner_name = max(results, key=lambda n: results[n]["val_auc"])
    winner = results[winner_name]
    test_probs = winner["pipeline"].predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    test_brier = brier_score_loss(y_test, test_probs)
    auc_margin = results[winner_name]["val_auc"] - min(r["val_auc"] for r in results.values())



    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "model_type": winner_name,
        "pipeline": winner["pipeline"],
        "best_params": winner["params"],
        "feature_names": FEATURE_NAMES,
        "metrics": {
            name: {"val_auc": r["val_auc"], "val_brier": r["val_brier"]}
            for name, r in results.items()
        },
        "winner_test_auc": test_auc,
        "winner_test_brier": test_brier,
        # Audit trail: how far ahead was the winner on the val set?
        # Small margin here means the gap is inside noise — logged for
        # visibility, but the winner is whoever the data picked, not a rule.
        "auc_margin_vs_runner_up": round(auc_margin, 6),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    summary = {k: v for k, v in bundle.items() if k != "pipeline"}
    with open(MODEL_DIR / "subscription_winner_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return bundle


if __name__ == "__main__":
    result = train()
    print(f"Winner: {result['model_type']}")
    print(f"Saved bundle to {MODEL_PATH}")
    print(f"Per-candidate val AUC: {result['metrics']}")
    print(f"Winner test AUC: {result['winner_test_auc']:.4f}")
    print(f"Winner test Brier: {result['winner_test_brier']:.4f}")
    print(f"Best params: {result['best_params']}")
