"""
Offline trainer for the subscription diagnosis-layer model.

Run this manually (Gauresh runs training, not Claude). Trains every real
candidate from architecture doc section 6 that's practical to retrain here
(GBM, MLP), evaluates each on held-out AUC, and saves whichever wins as a
single self-contained "bundle" — not just raw model weights.

    python -m backend.ml.train_subscription_model

Produces: backend/ml/models/subscription_winner.joblib

WHY A BUNDLE, NOT JUST A MODEL:
Each candidate is wrapped in its own sklearn Pipeline, so model-specific
preprocessing (e.g. StandardScaler for the MLP — its absence previously
caused the net to collapse to near-identical predictions, per prior
debugging) lives INSIDE the saved object, not in the module's inference
code. SubscriptionModule never branches on model type; it just calls
pipeline.predict_proba(...) on whatever won. This means a future model
family (e.g. the sequence-model comparison point in architecture doc 6.3)
only requires adding a candidate here, not touching the module at all.

The bundle also carries its own feature_names schema, checked at load time
against the module's live build_feature_vector output — this is the
concrete fix for the exact failure mode flagged in review: a trained model
silently fed misaligned columns produces confident garbage with no error.

NOTE ON THE ORACLE CEILING: this trainer runs against the CURRENT generator,
which now includes code-51 amount-dependence. The 0.694 AUC ceiling
recorded from the original step-5 comparison was computed before that
change and is now stale — this script does not recompute it. Treat any
val_auc/test_auc printed below as not yet checked against a ceiling.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.data.splitting import entity_level_split
from backend.data.subscription_generator import (
    CODE_BASE_RECOVERY_RATE,
    generate_subscription_dataset,
)
from backend.ml.features import FEATURE_NAMES, build_feature_vector

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "subscription_winner.joblib"

BUNDLE_SCHEMA_VERSION = 1


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


def _build_candidates() -> dict[str, Pipeline]:
    """
    Every candidate is a full Pipeline (preprocessing + calibrated
    classifier), so downstream code only ever calls .predict_proba() on
    whichever one wins — no model-specific branching outside this file.
    """
    gbm = Pipeline(
        steps=[
            (
                "clf",
                CalibratedClassifierCV(
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                    ),
                    method="sigmoid",  # beat isotonic per architecture doc 6.5
                    cv=3,
                ),
            )
        ]
    )

    mlp = Pipeline(
        steps=[
            ("scaler", StandardScaler()),  # required — MLP collapsed without this previously
            (
                "clf",
                CalibratedClassifierCV(
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        alpha=1e-3,
                        max_iter=500,
                        random_state=42,
                    ),
                    method="sigmoid",
                    cv=3,
                ),
            )
        ]
    )

    return {"GBM": gbm, "MLP": mlp}


def train() -> dict:
    records = generate_subscription_dataset()  # real default scale, per architecture doc 5
    train_records, val_records, test_records = entity_level_split(records)

    X_train, y_train = _records_to_matrix(train_records)
    X_val, y_val = _records_to_matrix(val_records)
    X_test, y_test = _records_to_matrix(test_records)

    candidates = _build_candidates()
    results: dict[str, dict] = {}

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        results[name] = {
            "pipeline": pipeline,
            "val_auc": roc_auc_score(y_val, val_probs),
            "val_brier": brier_score_loss(y_val, val_probs),
        }

    winner_name = max(results, key=lambda n: results[n]["val_auc"])
    winner_pipeline = results[winner_name]["pipeline"]
    test_probs = winner_pipeline.predict_proba(X_test)[:, 1]

    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "model_type": winner_name,
        "pipeline": winner_pipeline,
        "feature_names": FEATURE_NAMES,
        "metrics": {
            name: {"val_auc": r["val_auc"], "val_brier": r["val_brier"]}
            for name, r in results.items()
        },
        "winner_test_auc": roc_auc_score(y_test, test_probs),
        "winner_test_brier": brier_score_loss(y_test, test_probs),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    # Human-readable copy alongside the joblib — metrics/model_type should be
    # inspectable without unpickling, for the audit trail this whole project
    # cares about.
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