"""
B2B receivables diagnosis-layer model comparison.

    python -m backend.ml.compare_b2b

Follows the SAME discipline as the subscription compare_all.py:
- Entity-level train/val/test split throughout.
- Oracle ceiling computed from true_payment_probability() — the exact
  function b2b_generator.py samples against. "Ground-truth-aware" here
  means we genuinely know the data-generating function; calling it oracle
  is precise, not aspirational.
- All models calibrated (sigmoid/Platt) before any probability is read out.
- One run, reported once — no cherry-picking.

WHY NO SEQUENCE MODEL HERE:
The subscription comparison added an LSTM as a fourth point because
subscription retries ARE a genuine sequence — multiple attempts per case,
over time, with ordering that matters. B2B invoice records are not: each
record is a SINGLE invoice's state at a point in time, not a sequence of
contact attempts. Building an LSTM on flat, unordered invoice records would
be adding architecture that doesn't match the data shape — the very thing
the architecture doc warns against. Three flat comparison points (Baseline,
GBM, MLP) is the honest scope here.

FEATURE SET — 5 features, all grounded in what b2b_generator.py's
true_payment_probability actually depends on:
1. days_overdue           -- the primary aging-bucket driver
2. is_msme_registered     -- Section 43B(h) incentive direction sourced
3. customer_recent_payment_pressure -- causal customer EWMA (0.0 in default dataset)
4. invoice_amount         -- lognormal-distributed B2B invoice scale
5. has_written_agreement  -- only useful when is_msme_registered=True (15 vs 45d)

NOTE: on_dnd_registry, has_opted_out, is_disputed are NOT features here —
they never produce recovered=True by construction (check_stop fires before
execute), so they carry zero predictive signal for the binary outcome this
model is trained to predict. Including them would be a leakage mistake, not
a richer model.

HONEST EXPECTATION FOR THIS DATASET:
The subscription oracle was 0.703. B2B's true_payment_probability() is
simpler — it depends on fewer factors (aging bucket, MSME flag, customer
pressure) and customer_recent_payment_pressure is 0.0 in the default
dataset (include_customer_history=False). Expect a lower oracle ceiling and
tighter model/oracle gap as a result — this is not a dataset designed to
be hard, it's designed to reflect the actual generating function's structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.data.b2b_generator import generate_b2b_dataset, true_payment_probability
from backend.data.splitting import entity_level_split

RESULTS_PATH = Path(__file__).parent / "models" / "comparison_b2b_results.json"

# Feature names — fixed order, same "one source of truth" discipline as
# features.py for the subscription domain.
B2B_FEATURE_NAMES: list[str] = [
    "days_overdue",
    "is_msme_registered",
    "customer_recent_payment_pressure",
    "invoice_amount",
    "has_written_agreement",
]


# ------------------------------------------------------------------ #
# Data preparation                                                    #
# ------------------------------------------------------------------ #

def _build_feature_vector(record) -> list[float]:
    """
    Maps one B2BInvoiceRecord to the fixed-length feature vector.
    Same "one place feature engineering happens" rule as features.py.
    """
    return [
        float(record.days_overdue),
        1.0 if record.is_msme_registered else 0.0,
        float(record.customer_recent_payment_pressure),
        float(record.invoice_amount),
        1.0 if record.has_written_agreement else 0.0,
    ]


def _chased_only(records):
    """
    Drop records that check_stop would halt before execute — same discipline
    as subscription's hard/stop-code filtering. Including them would be
    leakage (the label is determined by construction, not by the data).
    """
    return [r for r in records if not (r.on_dnd_registry or r.has_opted_out or r.is_disputed)]


def _flat_matrix(records) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([_build_feature_vector(r) for r in records])
    y = np.array([1.0 if r.recovered else 0.0 for r in records])
    return X, y


def _oracle_auc(records) -> float:
    probs = np.array([
        true_payment_probability(
            days_overdue=r.days_overdue,
            is_msme_registered=r.is_msme_registered,
            customer_recent_payment_pressure=r.customer_recent_payment_pressure,
        )
        for r in records
    ])
    y = np.array([1.0 if r.recovered else 0.0 for r in records])
    return roc_auc_score(y, probs)


# ------------------------------------------------------------------ #
# Model builders                                                      #
# ------------------------------------------------------------------ #

def _build_baseline() -> Pipeline:
    """
    Logistic regression on standardized features — the simplest real model
    that can use the features, i.e. a trained linear baseline. Calibrated
    with sigmoid. This is the floor to beat; a DummyClassifier is too
    trivial to be instructive when we have real, sourced features.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, random_state=42),
            method="sigmoid",
            cv=3,
        ))
    ])


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
    print("B2B RECEIVABLES MODEL COMPARISON -- Baseline vs GBM vs MLP")
    print("Entity-level split  |  Calibrated (sigmoid)  |  No fake numbers")
    print(sep)

    # ---- Data ----
    print("\nGenerating B2B invoice dataset (default scale: 4000 customers)...")
    all_records = generate_b2b_dataset()
    chased = _chased_only(all_records)
    train_records, val_records, test_records = entity_level_split(chased)

    print(f"  Total records generated:               {len(all_records)}")
    print(f"  Chased records (DND/opt-out/disputed filtered): {len(chased)}")
    print(f"  Entity-level split: {len(train_records)} train / {len(val_records)} val / {len(test_records)} test")
    print(f"  Features: {B2B_FEATURE_NAMES}")

    # Recovery rate summary
    overall_rate = sum(r.recovered for r in chased) / len(chased)
    print(f"  Overall recovery rate (chased): {overall_rate:.3f}")

    # ---- Oracle ----
    print("\nComputing oracle ceiling on test set...")
    oracle_auc = _oracle_auc(test_records)
    print(f"  Oracle AUC (Bayes ceiling): {oracle_auc:.4f}")
    print(
        "  NOTE: Customer-pressure=0.0 for all records (default dataset has\n"
        "  include_customer_history=False), so that feature carries no signal\n"
        "  here. Oracle ceiling reflects this simplified distribution."
    )

    # ---- Matrices ----
    print("\nBuilding feature matrices...")
    X_train, y_train = _flat_matrix(train_records)
    X_val,   y_val   = _flat_matrix(val_records)
    X_test,  y_test  = _flat_matrix(test_records)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    results = {}

    # ---- Baseline (Logistic Regression) ----
    print("\n" + "-" * 70)
    print("Baseline (Logistic Regression + sigmoid calibration)")
    print("-" * 70)
    baseline = _build_baseline()
    baseline.fit(X_train, y_train)
    bl_val_probs  = baseline.predict_proba(X_val)[:, 1]
    bl_test_probs = baseline.predict_proba(X_test)[:, 1]
    bl_val_auc    = roc_auc_score(y_val, bl_val_probs)
    bl_test_auc   = roc_auc_score(y_test, bl_test_probs)
    bl_test_brier = brier_score_loss(y_test, bl_test_probs)
    print(f"  Val  AUC:   {bl_val_auc:.4f}")
    print(f"  Test AUC:   {bl_test_auc:.4f}  (gap to oracle: {oracle_auc - bl_test_auc:+.4f})")
    print(f"  Test Brier: {bl_test_brier:.4f}")
    results["Baseline"] = {
        "val_auc": round(bl_val_auc, 4),
        "test_auc": round(bl_test_auc, 4),
        "test_brier": round(bl_test_brier, 4),
        "gap_to_oracle": round(oracle_auc - bl_test_auc, 4),
    }

    # ---- GBM ----
    print("\n" + "-" * 70)
    print("GBM (XGBoost + sigmoid calibration)")
    print("-" * 70)
    gbm = _build_gbm()
    gbm.fit(X_train, y_train)
    gbm_val_probs  = gbm.predict_proba(X_val)[:, 1]
    gbm_test_probs = gbm.predict_proba(X_test)[:, 1]
    gbm_val_auc    = roc_auc_score(y_val, gbm_val_probs)
    gbm_test_auc   = roc_auc_score(y_test, gbm_test_probs)
    gbm_test_brier = brier_score_loss(y_test, gbm_test_probs)
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
    print("MLP (sklearn, (32,16) ReLU + sigmoid calibration)")
    print("-" * 70)
    mlp = _build_mlp()
    mlp.fit(X_train, y_train)
    mlp_val_probs  = mlp.predict_proba(X_val)[:, 1]
    mlp_test_probs = mlp.predict_proba(X_test)[:, 1]
    mlp_val_auc    = roc_auc_score(y_val, mlp_val_probs)
    mlp_test_auc   = roc_auc_score(y_test, mlp_test_probs)
    mlp_test_brier = brier_score_loss(y_test, mlp_test_probs)
    print(f"  Val  AUC:   {mlp_val_auc:.4f}")
    print(f"  Test AUC:   {mlp_test_auc:.4f}  (gap to oracle: {oracle_auc - mlp_test_auc:+.4f})")
    print(f"  Test Brier: {mlp_test_brier:.4f}")
    results["MLP"] = {
        "val_auc": round(mlp_val_auc, 4),
        "test_auc": round(mlp_test_auc, 4),
        "test_brier": round(mlp_test_brier, 4),
        "gap_to_oracle": round(oracle_auc - mlp_test_auc, 4),
    }

    # ---- Summary ----
    print("\n" + sep)
    print("SUMMARY TABLE")
    print(sep)
    print(f"  Oracle ceiling (Bayes): {oracle_auc:.4f}")
    print()
    print(f"  {'Model':<10}  {'Val AUC':>8}  {'Test AUC':>9}  {'Brier':>7}  {'Gap to oracle':>14}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*14}")
    for name, r in results.items():
        print(
            f"  {name:<10}  {r['val_auc']:>8.4f}  {r['test_auc']:>9.4f}"
            f"  {r['test_brier']:>7.4f}  {r['gap_to_oracle']:>+14.4f}"
        )

    best = max(results, key=lambda n: results[n]["test_auc"])
    worst = min(results, key=lambda n: results[n]["test_auc"])
    spread = results[best]["test_auc"] - results[worst]["test_auc"]

    print()
    print(f"  Best test AUC:  {best} ({results[best]['test_auc']:.4f})")
    print(f"  Spread across all three models: {spread:.4f}")

    if spread < 0.005:
        print("  -> Spread < 0.005: all three models are statistically indistinguishable.")
        print("     The data generating function (aging bucket + MSME flag) is learnable")
        print("     by any of these models equally well — architecture doesn't matter here.")
    elif spread < 0.02:
        print("  -> Spread < 0.02: modest difference. Prefer the simpler model.")
    else:
        print(f"  -> {best} shows a meaningful advantage. Investigate before committing.")

    print()
    print("  INTERPRETATION NOTES:")
    print("  - Oracle < subscription oracle is expected: B2B's true_payment_probability()")
    print("    depends on fewer, simpler factors than subscription's (no decline-code one-hot,")
    print("    no hour-of-day, no attempt-number decay). The data is structurally simpler.")
    print("  - A model close to oracle on B2B means it has learned the aging-bucket")
    print("    monotonic decay, not that it has solved a hard inference problem.")
    print("  - Gap to oracle is the honest gap: 0.00 is impossible (sampling noise),")
    print("    < 0.01 means the model is essentially learning the generating function.")
    print()

    # ---- Save ----
    output = {
        "oracle_auc": round(oracle_auc, 4),
        "models": results,
        "data": {
            "n_records_total": len(all_records),
            "n_chased": len(chased),
            "n_train": len(train_records),
            "n_val": len(val_records),
            "n_test": len(test_records),
            "overall_recovery_rate": round(overall_rate, 4),
        },
        "feature_names": B2B_FEATURE_NAMES,
        "note": (
            "Entity-level split on customer_id. DND/opt-out/disputed records excluded "
            "(recovered=False by construction, zero predictive signal). "
            "No sequence model: B2B records are single-invoice states, not retry chains. "
            "customer_recent_payment_pressure=0.0 for all (default dataset)."
        ),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
