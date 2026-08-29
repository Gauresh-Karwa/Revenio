"""
Does GBM/NN benefit from customer_recent_failure_pressure the same way the
LSTM did? Reuses compare.py's exact tune_gbm/tune_nn/calibrate_and_evaluate
functions on an enriched, separate dataset.

    python -m backend.ml.compare_with_history

Kept as a SEPARATE script from compare.py: compare.py's baseline/GBM/NN
numbers (already recorded in README.md) are a valid historical reference
for the original flat feature set. This script's dataset and features are
different (include_customer_history=True), so it gets its own script
rather than silently changing compare.py's own output.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score

from backend.data.splitting import entity_level_split
from backend.data.subscription_generator import (
    generate_subscription_dataset,
    true_recovery_probability,
)
from backend.ml.calibration import calibrate_and_evaluate
from backend.ml.features import (
    build_feature_matrix_with_history,
    fit_scaler,
    records_to_frame_with_history,
)
from backend.ml.models.baseline import RuleBasedBaseline
from backend.ml.models.gbm import tune_gbm, train_final_gbm
from backend.ml.models.neural_net import tune_nn, train_final_nn, _predict_proba


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, probs),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "brier": brier_score_loss(y_true, probs),
    }


def _oracle_auc_for_enriched_flat_distribution(records) -> float:
    from backend.data.subscription_generator import CODE_BASE_RECOVERY_RATE

    soft_records = [r for r in records if r.decline_code in CODE_BASE_RECOVERY_RATE]
    true_probs = [
        true_recovery_probability(
            decline_code=r.decline_code, amount=r.amount, attempt_number=r.attempt_number,
            hour_of_day=r.hour_of_day, is_near_payday=r.is_near_payday,
            customer_recent_failure_pressure=r.customer_recent_failure_pressure,
        )
        for r in soft_records
    ]
    y = [int(r.recovered) for r in soft_records]
    return roc_auc_score(y, true_probs)


def main() -> None:
    print("=" * 70)
    print("DOES GBM/NN BENEFIT FROM customer_recent_failure_pressure LIKE THE LSTM DID?")
    print("=" * 70)

    records = generate_subscription_dataset(include_customer_history=True)
    print(f"\n{len(records)} total records generated (soft + hard + stop codes), WITH customer history.")

    train_records, val_records, test_records = entity_level_split(records)

    train_df = records_to_frame_with_history(train_records)
    val_df = records_to_frame_with_history(val_records)
    test_df = records_to_frame_with_history(test_records)
    print(f"Entity-level split (soft-decline rows only): "
          f"{len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    X_train, y_train, feature_names, groups_train = build_feature_matrix_with_history(train_df)
    X_val, y_val, _, _ = build_feature_matrix_with_history(val_df)
    X_test, y_test, _, _ = build_feature_matrix_with_history(test_df)
    print(f"Feature set ({len(feature_names)}): {feature_names}")

    oracle_auc = _oracle_auc_for_enriched_flat_distribution(test_records)
    print(f"\nOracle AUC for this enriched flat distribution: {oracle_auc:.4f}")

    results = {}

    print("\n--- Baseline (rule-based lookup) ---")
    baseline = RuleBasedBaseline().fit(train_df)
    baseline_probs = baseline.predict_proba(test_df)
    results["baseline"] = evaluate_probs(y_test, baseline_probs)
    print(f"  AUC={results['baseline']['auc']:.3f}")

    print("\n--- GBM (XGBoost) WITH customer_recent_failure_pressure, random search ---")
    gbm_params, gbm_cv_score = tune_gbm(X_train, y_train, groups_train, n_iter=25)
    print(f"  Best params: {gbm_params}")
    print(f"  CV AUC={gbm_cv_score:.3f}")
    gbm_model = train_final_gbm(X_train, y_train, gbm_params)
    gbm_probs = gbm_model.predict_proba(X_test)[:, 1]
    results["gbm_with_history"] = evaluate_probs(y_test, gbm_probs)
    print(f"  Test: AUC={results['gbm_with_history']['auc']:.3f}")

    nn_scaler = fit_scaler(X_train)
    X_train_nn = nn_scaler.transform(X_train)
    X_test_nn = nn_scaler.transform(X_test)

    print("\n--- NN WITH customer_recent_failure_pressure, random search ---")
    nn_params, nn_cv_score = tune_nn(X_train_nn, y_train, groups_train, n_iter=15)
    print(f"  Best params: {nn_params}")
    print(f"  CV AUC={nn_cv_score:.3f}")
    nn_model = train_final_nn(X_train_nn, y_train, nn_params)
    nn_probs = _predict_proba(nn_model, X_test_nn)
    results["nn_with_history"] = evaluate_probs(y_test, nn_probs)
    print(f"  Test: AUC={results['nn_with_history']['auc']:.3f}")

    print("\n" + "=" * 70)
    print("Result — is a second production model (LSTM) actually justified?")
    print("=" * 70)
    gbm_gap = oracle_auc - results["gbm_with_history"]["auc"]
    nn_gap = oracle_auc - results["nn_with_history"]["auc"]
    print(f"  GBM (with history) test AUC:  {results['gbm_with_history']['auc']:.4f}   gap to oracle: {gbm_gap:.4f}")
    print(f"  NN  (with history) test AUC:  {results['nn_with_history']['auc']:.4f}   gap to oracle: {nn_gap:.4f}")
    print(f"  Oracle ceiling (this dist.):  {oracle_auc:.4f}")
    print()
    print("  For reference (README.md): LSTM gap to ITS OWN chain-distribution")
    print("  oracle was 0.0049. Original (no-history) GBM's gap was 0.0026.")
    best_flat_gap = min(gbm_gap, nn_gap)
    if best_flat_gap <= 0.0049 + 0.005:
        print("\n  A flat model given the SAME feature tracks its own ceiling about as")
        print("  tightly as the LSTM did. Recommendation: retrain the ONE deployed")
        print("  bundle with this feature added; do not wire in the LSTM.")
    else:
        print("\n  The flat model's gap is meaningfully wider than the LSTM's — worth")
        print("  a closer look before deciding either way.")


if __name__ == "__main__":
    main()
