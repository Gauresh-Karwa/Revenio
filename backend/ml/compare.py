from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score

from backend.data.splitting import entity_level_split
from backend.data.subscription_generator import generate_subscription_dataset
from backend.ml.calibration import calibrate_and_evaluate
from backend.ml.evaluation import print_per_code_breakdown, print_reliability_table
from backend.ml.features import build_feature_matrix, fit_scaler, records_to_frame
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


def main():
    print("=" * 70)
    print("STEP 5 — Subscription diagnosis-layer model comparison")
    print("=" * 70)

    records = generate_subscription_dataset()
    print(f"\n{len(records)} total records generated (soft + hard + stop codes).")

    train_records, val_records, test_records = entity_level_split(records)

    train_df = records_to_frame(train_records)
    val_df = records_to_frame(val_records)
    test_df = records_to_frame(test_records)
    print(f"Entity-level split (soft-decline rows only): "
          f"{len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    X_train, y_train, feature_names, groups_train = build_feature_matrix(train_df)
    X_val, y_val, _, _ = build_feature_matrix(val_df)
    X_test, y_test, _, _ = build_feature_matrix(test_df)
    print(f"Feature set ({len(feature_names)}): {feature_names}")

    results = {}
    probs_by_model = {}

    print("\n--- Baseline (rule-based lookup) ---")
    baseline = RuleBasedBaseline().fit(train_df)
    baseline_probs = baseline.predict_proba(test_df)
    results["baseline"] = evaluate_probs(y_test, baseline_probs)
    probs_by_model["baseline"] = baseline_probs
    print(f"  AUC={results['baseline']['auc']:.3f}  Precision={results['baseline']['precision']:.3f}  "
          f"Recall={results['baseline']['recall']:.3f}  Brier={results['baseline']['brier']:.3f}")

    print("\n--- Gradient-boosted trees (XGBoost), random search over real ranges ---")
    gbm_params, gbm_cv_score = tune_gbm(X_train, y_train, groups_train, n_iter=25)
    print(f"  Best params: {gbm_params}")
    print(f"  CV AUC={gbm_cv_score:.3f}")
    gbm_model = train_final_gbm(X_train, y_train, gbm_params)
    gbm_probs = gbm_model.predict_proba(X_test)[:, 1]
    results["gbm"] = evaluate_probs(y_test, gbm_probs)
    probs_by_model["gbm"] = gbm_probs
    print(f"  Test: AUC={results['gbm']['auc']:.3f}  Precision={results['gbm']['precision']:.3f}  "
          f"Recall={results['gbm']['recall']:.3f}  Brier={results['gbm']['brier']:.3f}")

    nn_scaler = fit_scaler(X_train)  # fit on TRAIN only — same leakage discipline as everything else
    X_train_nn = nn_scaler.transform(X_train)
    X_val_nn = nn_scaler.transform(X_val)
    X_test_nn = nn_scaler.transform(X_test)

    print("\n--- Neural net, random search over real ranges (width scaled to n_features) ---")
    nn_params, nn_cv_score = tune_nn(X_train_nn, y_train, groups_train, n_iter=15)
    hidden_width = X_train.shape[1] * nn_params["width_multiplier"]
    print(f"  Best params: {nn_params}  (-> hidden width {hidden_width}, {nn_params['n_layers']} layers)")
    print(f"  CV AUC={nn_cv_score:.3f}")
    nn_model = train_final_nn(X_train_nn, y_train, nn_params)
    nn_probs = _predict_proba(nn_model, X_test_nn)
    results["nn"] = evaluate_probs(y_test, nn_probs)
    probs_by_model["nn"] = nn_probs
    print(f"  Test: AUC={results['nn']['auc']:.3f}  Precision={results['nn']['precision']:.3f}  "
          f"Recall={results['nn']['recall']:.3f}  Brier={results['nn']['brier']:.3f}")
    print("\n" + "=" * 70)
    winner_name = max(results, key=lambda k: results[k]["auc"])
    print(f"WINNER (by test AUC): {winner_name}  (AUC={results[winner_name]['auc']:.3f})")
    print("=" * 70)
    for name, metrics in sorted(results.items(), key=lambda kv: -kv[1]["auc"]):
        tag = " <-- WINNER" if name == winner_name else "   lost"
        print(f"  {name:10s} AUC={metrics['auc']:.3f}{tag}")

    print("\n" + "=" * 70)
    print("Reliability curves (raw, before calibration)")
    print("=" * 70)
    for name, probs in probs_by_model.items():
        print_reliability_table(y_test, probs, label=name)
        print()

    print("=" * 70)
    print("Per-decline-code breakdown")
    print("=" * 70)
    for name, probs in probs_by_model.items():
        print_per_code_breakdown(test_df, y_test, probs, label=name)
        print()

    print("=" * 70)
    print(f"Calibrating the winner ({winner_name}), per architecture doc 6.1")
    print("=" * 70)
    if winner_name == "gbm":
        predict_fn = lambda X: gbm_model.predict_proba(X)[:, 1]
    elif winner_name == "nn":
        predict_fn = lambda X: _predict_proba(nn_model, nn_scaler.transform(X))
    else:
        predict_fn = None

    if predict_fn is not None:
        calibrated_model, brier_before, brier_after = calibrate_and_evaluate(
            predict_fn, X_val, y_val, X_test, y_test
        )
        print(f"  Brier score before calibration: {brier_before:.4f}")
        print(f"  Brier score after calibration:  {brier_after:.4f}")
        print(f"  {'IMPROVED' if brier_after < brier_before else 'DID NOT IMPROVE'} — reported honestly either way.")

        calibrated_probs = calibrated_model.predict_proba(X_test)[:, 1]
        print()
        print_reliability_table(y_test, calibrated_probs, label=f"{winner_name} (calibrated)")
    else:
        print("  Baseline won — it's already a lookup table of observed rates, calibration doesn't apply.")

    # --- Cross-distribution generalization test (architecture doc section 5) ---
    # Promised since step 4, not actually run until now. Trains nothing new —
    # evaluates the SAME winning model, trained only on regime A, against a
    # deliberately different regime B (smaller payday effect, harder-to-
    # recover issuer/system-error codes), never seen during training.
    print("\n" + "=" * 70)
    print("Cross-distribution generalization test (architecture doc section 5)")
    print("=" * 70)
    from backend.data.subscription_generator import generate_subscription_dataset as _gen

    regime_b_records = _gen(
        seed=999,
        base_recovery_override={"91": 0.55, "96": 0.50},
        payday_boost_override=1.05,
    )
    _, _, regime_b_test = entity_level_split(regime_b_records, seed=999)
    regime_b_df = records_to_frame(regime_b_test)
    X_regime_b, y_regime_b, _, _ = build_feature_matrix(regime_b_df)

    if winner_name == "gbm":
        regime_b_probs = gbm_model.predict_proba(X_regime_b)[:, 1]
    elif winner_name == "nn":
        regime_b_probs = _predict_proba(nn_model, nn_scaler.transform(X_regime_b))
    else:
        regime_b_probs = baseline.predict_proba(regime_b_df)

    baseline_probs_on_b = baseline.predict_proba(regime_b_df)
    same_dist_auc = results[winner_name]["auc"]
    shifted_auc = roc_auc_score(y_regime_b, regime_b_probs)
    baseline_shifted_auc = roc_auc_score(y_regime_b, baseline_probs_on_b)

    print(f"  {winner_name} — same distribution (regime A):  AUC={same_dist_auc:.3f}")
    print(f"  {winner_name} — shifted distribution (regime B): AUC={shifted_auc:.3f}  (drop: {same_dist_auc-shifted_auc:+.3f})")
    print(f"  baseline — shifted distribution (regime B):     AUC={baseline_shifted_auc:.3f}  "
          f"(drop: {results['baseline']['auc']-baseline_shifted_auc:+.3f})")
    if (same_dist_auc - shifted_auc) < (results["baseline"]["auc"] - baseline_shifted_auc):
        print(f"  {winner_name} degrades LESS than baseline under shift — genuinely more generalizable, not just better-fit.")
    else:
        print(f"  {winner_name} degrades AS MUCH OR MORE than baseline under shift — worth investigating before trusting it broadly.")


if __name__ == "__main__":
    main()