"""
Offline trainer for the subscription diagnosis-layer model.

Run this manually (Gauresh runs training, not Claude).

    python -m backend.ml.train_subscription_model

Produces: backend/ml/models/subscription_winner.joblib

SCHEMA v3: trains on FEATURE_NAMES_WITH_HISTORY_AND_TEXT (12 features,
adding hardship_signal_detected). Any bundle trained by an OLDER version of
this script will be correctly rejected at load time by SubscriptionModule's
feature_names check.

SWAPPING THE EXTRACTOR:

    from backend.ml.text_signals import extract_hardship_signal_llm
    train(hardship_extractor=extract_hardship_signal_llm)

Requires `pip install anthropic` and ANTHROPIC_API_KEY set.
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
from backend.ml.features import (
    FEATURE_NAMES_WITH_HISTORY_AND_TEXT,
    build_feature_vector_with_history_and_text,
)
from backend.ml.text_signals import HardshipExtractor, extract_hardship_signal_embedding

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "subscription_winner.joblib"

BUNDLE_SCHEMA_VERSION = 3


def _records_to_matrix(records, extractor: HardshipExtractor) -> tuple[np.ndarray, np.ndarray]:
    soft_records = [r for r in records if r.decline_code in CODE_BASE_RECOVERY_RATE]
    X = np.array(
        [
            build_feature_vector_with_history_and_text(
                decline_code=r.decline_code,
                attempt_number=r.attempt_number,
                hour_of_day=r.hour_of_day,
                is_near_payday=r.is_near_payday,
                amount=r.amount,
                customer_recent_failure_pressure=r.customer_recent_failure_pressure,
                hardship_signal_detected=extractor(r.email_text)["hardship_signal_detected"],
            )
            for r in soft_records
        ]
    )
    y = np.array([1.0 if r.recovered else 0.0 for r in soft_records])
    return X, y


def _build_candidates() -> dict[str, Pipeline]:
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
                    method="sigmoid",
                    cv=3,
                ),
            )
        ]
    )

    mlp = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        alpha=1e-3,
                        max_iter=1000,
                        random_state=42,
                    ),
                    method="sigmoid",
                    cv=3,
                ),
            )
        ]
    )

    return {"GBM": gbm, "MLP": mlp}


def train(hardship_extractor: HardshipExtractor = extract_hardship_signal_embedding) -> dict:
    records = generate_subscription_dataset(
        include_customer_history=True, include_support_email_signal=True
    )
    train_records, val_records, test_records = entity_level_split(records)

    X_train, y_train = _records_to_matrix(train_records, hardship_extractor)
    X_val, y_val = _records_to_matrix(val_records, hardship_extractor)
    X_test, y_test = _records_to_matrix(test_records, hardship_extractor)

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
        "feature_names": FEATURE_NAMES_WITH_HISTORY_AND_TEXT,
        "hardship_extractor_name": hardship_extractor.__name__,
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

    summary = {k: v for k, v in bundle.items() if k != "pipeline"}
    with open(MODEL_DIR / "subscription_winner_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return bundle


if __name__ == "__main__":
    result = train()
    print(f"Winner: {result['model_type']}")
    print(f"Extractor used: {result['hardship_extractor_name']}")
    print(f"Saved bundle to {MODEL_PATH}")
    print(f"Per-candidate val AUC: {result['metrics']}")
    print(f"Winner test AUC: {result['winner_test_auc']:.4f}")
    print(f"Winner test Brier: {result['winner_test_brier']:.4f}")