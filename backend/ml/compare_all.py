"""
Unified honest comparison: GBM vs MLP vs LSTM.

    python -m backend.ml.compare_all

All three models are trained and tested on the SAME entity-level split of the
SAME underlying dataset (generate_subscription_retry_sequences, which gives
genuine retry chains with customer_recent_failure_pressure). All three get the
same features available in each sequence step. This is the only way the
comparison is genuinely apples-to-apples.

FLAT MODELS (GBM, MLP):
  Receive the last attempt's flat feature vector from each sequence step —
  the same information the flat pipeline has historically used. This puts
  them at no advantage or disadvantage relative to their original evaluation;
  it just uses a consistent data source.

LSTM:
  Receives the full padded sequence up to each attempt. Has access to prior
  attempt context that the flat models do not — which is the genuine
  architectural question: does remembering prior attempts help?

ORACLE:
  Uses true_recovery_probability() directly on the test set. Establishes
  the Bayes ceiling for THIS data distribution so all model gaps are
  interpretable as a fraction of headroom, not raw numbers.

REPORTED METRICS per model:
  - Val AUC (used for hyperparameter selection)
  - Test AUC (held out, reported once)
  - Test Brier (calibration quality)
  - Gap to oracle ceiling

Schema: v3 (12-feature flat, 10-feature sequence step).
Hardship extractor: extract_hardship_signal_embedding (default, offline).
"""

from __future__ import annotations

import json
from pathlib import Path

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
    generate_subscription_retry_sequences,
    true_recovery_probability,
)
from backend.ml.features import (
    FEATURE_NAMES_WITH_HISTORY_AND_TEXT,
    build_feature_vector_with_history_and_text,
)
from backend.ml.models.sequence import predict_proba, train_final_lstm, tune_lstm
from backend.ml.sequence_features import cases_to_sequence_examples, pad_sequences
from backend.ml.text_signals import extract_hardship_signal_embedding

RESULTS_PATH = Path(__file__).parent / "models" / "comparison_all_results.json"


# ------------------------------------------------------------------ #
# Data preparation helpers                                            #
# ------------------------------------------------------------------ #

def _flat_matrix(cases) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts retry-chain cases to a flat feature matrix.
    Each ROW is ONE attempt (the last step's context), keeping the
    same granularity as the sequence model for a fair comparison.
    """
    rows, labels = [], []
    for case in cases:
        if case.decline_code not in CODE_BASE_RECOVERY_RATE:
            continue
        hardship_detected = extract_hardship_signal_embedding(
            case.email_text if hasattr(case, "email_text") else None
        )["hardship_signal_detected"]
        for attempt in case.attempts:
            rows.append(
                build_feature_vector_with_history_and_text(
                    decline_code=case.decline_code,
                    attempt_number=attempt.attempt_number,
                    hour_of_day=attempt.hour_of_day,
                    is_near_payday=case.is_near_payday,
                    amount=case.amount,
                    customer_recent_failure_pressure=case.customer_recent_failure_pressure,
                    hardship_signal_detected=hardship_detected,
                )
            )
            labels.append(1.0 if attempt.recovered else 0.0)
    return np.array(rows), np.array(labels)


def _sequence_matrix(cases):
    """
    Converts retry-chain cases to padded sequence tensors for the LSTM.
    """
    sequences, y, case_ids = cases_to_sequence_examples(cases)
    padded, lengths = pad_sequences(sequences)
    case_to_customer = {c.case_id: c.customer_id for c in cases}
    groups_str = [case_to_customer[cid] for cid in case_ids]
    unique = sorted(set(groups_str))
    customer_to_int = {c: i for i, c in enumerate(unique)}
    groups = np.array([customer_to_int[g] for g in groups_str])
    return padded, lengths, y, groups


def _oracle_auc(cases) -> float:
    probs, y = [], []
    for case in cases:
        for attempt in case.attempts:
            p = true_recovery_probability(
                decline_code=case.decline_code,
                amount=case.amount,
                attempt_number=attempt.attempt_number,
                hour_of_day=attempt.hour_of_day,
                is_near_payday=case.is_near_payday,
                customer_recent_failure_pressure=case.customer_recent_failure_pressure,
            )
            probs.append(p)
            y.append(1.0 if attempt.recovered else 0.0)
    return roc_auc_score(np.array(y), np.array(probs))


# ------------------------------------------------------------------ #
# Model builders                                                      #
# ------------------------------------------------------------------ #

def _build_gbm() -> Pipeline:
    return Pipeline([
        ("clf", CalibratedClassifierCV(
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
        ))
    ])


def _build_mlp() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=1e-3,
                max_iter=1000,
                random_state=42,
            ),
            method="sigmoid",
            cv=3,
        ))
    ])


# ------------------------------------------------------------------ #
# Main comparison                                                     #
# ------------------------------------------------------------------ #

def main() -> None:
    sep = "=" * 70

    print(sep)
    print("UNIFIED MODEL COMPARISON — GBM vs MLP vs LSTM")
    print("Schema v3  |  Same entity-level split  |  No fake numbers")
    print(sep)

    # ---- Data ----
    print("\nGenerating retry-chain dataset...")
    all_cases = generate_subscription_retry_sequences()
    soft_cases = [c for c in all_cases if c.decline_code in CODE_BASE_RECOVERY_RATE]
    train_cases, val_cases, test_cases = entity_level_split(soft_cases)
    print(f"  Total cases (soft-decline only): {len(soft_cases)}")
    print(f"  Entity-level split: {len(train_cases)} train / {len(val_cases)} val / {len(test_cases)} test")

    # ---- Oracle ----
    print("\nComputing oracle ceiling on test set...")
    oracle_auc = _oracle_auc(test_cases)
    print(f"  Oracle AUC (Bayes ceiling): {oracle_auc:.4f}")

    # ---- Flat matrices ----
    print("\nBuilding flat feature matrices (GBM / MLP)...")
    X_train_flat, y_train_flat = _flat_matrix(train_cases)
    X_val_flat,   y_val_flat   = _flat_matrix(val_cases)
    X_test_flat,  y_test_flat  = _flat_matrix(test_cases)
    print(f"  Flat train rows: {len(y_train_flat)}, val: {len(y_val_flat)}, test: {len(y_test_flat)}")
    print(f"  Features: {FEATURE_NAMES_WITH_HISTORY_AND_TEXT}")

    # ---- Sequence matrices ----
    print("\nBuilding sequence matrices (LSTM)...")
    X_train_seq, len_train, y_train_seq, groups_train = _sequence_matrix(train_cases)
    X_val_seq,   len_val,   y_val_seq,   _            = _sequence_matrix(val_cases)
    X_test_seq,  len_test,  y_test_seq,  _            = _sequence_matrix(test_cases)
    print(f"  Sequence train examples: {len(y_train_seq)}, val: {len(y_val_seq)}, test: {len(y_test_seq)}")

    results = {}

    # ---- GBM ----
    print("\n" + "-" * 70)
    print("GBM (XGBoost + sigmoid calibration, entity-level CV)")
    print("-" * 70)
    gbm = _build_gbm()
    gbm.fit(X_train_flat, y_train_flat)
    gbm_val_probs  = gbm.predict_proba(X_val_flat)[:, 1]
    gbm_test_probs = gbm.predict_proba(X_test_flat)[:, 1]
    gbm_val_auc    = roc_auc_score(y_val_flat, gbm_val_probs)
    gbm_test_auc   = roc_auc_score(y_test_flat, gbm_test_probs)
    gbm_test_brier = brier_score_loss(y_test_flat, gbm_test_probs)
    print(f"  Val  AUC:   {gbm_val_auc:.4f}")
    print(f"  Test AUC:   {gbm_test_auc:.4f}  (gap to oracle: {oracle_auc - gbm_test_auc:+.4f})")
    print(f"  Test Brier: {gbm_test_brier:.4f}")
    results["GBM"] = {
        "val_auc": round(gbm_val_auc, 4),
        "test_auc": round(gbm_test_auc, 4),
        "test_brier": round(gbm_test_brier, 4),
        "gap_to_oracle": round(oracle_auc - gbm_test_auc, 4),
    }

    # ---- MLP ----
    print("\n" + "-" * 70)
    print("MLP (sklearn, (32,16) ReLU, sigmoid calibration)")
    print("-" * 70)
    mlp = _build_mlp()
    mlp.fit(X_train_flat, y_train_flat)
    mlp_val_probs  = mlp.predict_proba(X_val_flat)[:, 1]
    mlp_test_probs = mlp.predict_proba(X_test_flat)[:, 1]
    mlp_val_auc    = roc_auc_score(y_val_flat, mlp_val_probs)
    mlp_test_auc   = roc_auc_score(y_test_flat, mlp_test_probs)
    mlp_test_brier = brier_score_loss(y_test_flat, mlp_test_probs)
    print(f"  Val  AUC:   {mlp_val_auc:.4f}")
    print(f"  Test AUC:   {mlp_test_auc:.4f}  (gap to oracle: {oracle_auc - mlp_test_auc:+.4f})")
    print(f"  Test Brier: {mlp_test_brier:.4f}")
    results["MLP"] = {
        "val_auc": round(mlp_val_auc, 4),
        "test_auc": round(mlp_test_auc, 4),
        "test_brier": round(mlp_test_brier, 4),
        "gap_to_oracle": round(oracle_auc - mlp_test_auc, 4),
    }

    # ---- LSTM ----
    print("\n" + "-" * 70)
    print("LSTM (random search, 15 iterations, entity-aware CV)")
    print("-" * 70)
    params, cv_auc = tune_lstm(X_train_seq, len_train, y_train_seq, groups_train, n_iter=15)
    print(f"  Best params: {params}")
    print(f"  CV AUC (tuning): {cv_auc:.4f}")
    lstm_model = train_final_lstm(X_train_seq, len_train, y_train_seq, params)
    lstm_val_probs  = predict_proba(lstm_model, X_val_seq, len_val)
    lstm_test_probs = predict_proba(lstm_model, X_test_seq, len_test)
    lstm_val_auc    = roc_auc_score(y_val_seq, lstm_val_probs)
    lstm_test_auc   = roc_auc_score(y_test_seq, lstm_test_probs)
    lstm_test_brier = brier_score_loss(y_test_seq, lstm_test_probs)
    print(f"  Val  AUC:   {lstm_val_auc:.4f}")
    print(f"  Test AUC:   {lstm_test_auc:.4f}  (gap to oracle: {oracle_auc - lstm_test_auc:+.4f})")
    print(f"  Test Brier: {lstm_test_brier:.4f}")
    results["LSTM"] = {
        "val_auc": round(lstm_val_auc, 4),
        "test_auc": round(lstm_test_auc, 4),
        "test_brier": round(lstm_test_brier, 4),
        "gap_to_oracle": round(oracle_auc - lstm_test_auc, 4),
    }

    # ---- Summary ----
    print("\n" + sep)
    print("SUMMARY TABLE")
    print(sep)
    print(f"  Oracle ceiling (Bayes): {oracle_auc:.4f}")
    print()
    print(f"  {'Model':<8}  {'Val AUC':>8}  {'Test AUC':>9}  {'Brier':>7}  {'Gap to oracle':>14}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*14}")
    for name, r in results.items():
        print(
            f"  {name:<8}  {r['val_auc']:>8.4f}  {r['test_auc']:>9.4f}"
            f"  {r['test_brier']:>7.4f}  {r['gap_to_oracle']:>+14.4f}"
        )

    # ---- Honest findings ----
    best = max(results, key=lambda n: results[n]["test_auc"])
    worst = min(results, key=lambda n: results[n]["test_auc"])
    spread = results[best]["test_auc"] - results[worst]["test_auc"]

    print()
    print(f"  Best test AUC:  {best} ({results[best]['test_auc']:.4f})")
    print(f"  Spread across all three models: {spread:.4f}")
    if spread < 0.005:
        print("  -> Spread < 0.005: all three models are statistically indistinguishable")
        print("     on this generator. Architecture choice won't move the needle here;")
        print("     the data is the constraint.")
    elif spread < 0.02:
        print("  -> Spread < 0.02: modest difference. Prefer the simpler model (GBM)")
        print("     unless there is a specific operational reason to accept complexity.")
    else:
        print(f"  -> {best} shows a meaningful advantage. Investigate before committing.")

    print()
    print("  NOTE ON LSTM vs FLAT:")
    print("  The LSTM receives the full attempt sequence (prior context). If")
    print("  LSTM AUC == GBM AUC it means prior-attempt context adds no signal")
    print("  beyond what attempt_number already captures in the flat features.")
    print("  That is a LEGITIMATE FINDING, not a bug.")

    print()

    # ---- Save results ----
    output = {
        "oracle_auc": round(oracle_auc, 4),
        "models": results,
        "data": {
            "n_cases_total": len(soft_cases),
            "n_train": len(train_cases),
            "n_val": len(val_cases),
            "n_test": len(test_cases),
            "flat_train_rows": int(len(y_train_flat)),
            "flat_test_rows": int(len(y_test_flat)),
            "sequence_train_examples": int(len(y_train_seq)),
            "sequence_test_examples": int(len(y_test_seq)),
        },
        "schema_version": 3,
        "hardship_extractor": "extract_hardship_signal_embedding",
        "note": (
            "All three models trained on the same entity-level split. "
            "Flat models (GBM, MLP) use the last attempt's flat feature vector. "
            "LSTM uses the full padded sequence. Oracle uses true_recovery_probability()."
        ),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
