"""
Fourth comparison point for the step-5 model-off: the LSTM sequence model
required by architecture doc 6.3. Kept separate from compare.py.

    python -m backend.ml.compare_sequence

v2: the retry-sequence generator now includes a real, causal
customer_recent_failure_pressure signal. This means the LSTM now has
access to information the flat GBM/NN models (compare.py) do NOT have —
so this run's AUC is NOT directly apples-to-apples with GBM/NN's flat
numbers anymore. What this run DOES establish honestly: whether
customer-history is a real, exploitable signal at all, via gap-to-own-oracle.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from backend.data.splitting import entity_level_split
from backend.data.subscription_generator import (
    generate_subscription_retry_sequences,
    true_recovery_probability,
)
from backend.ml.models.sequence import predict_proba, train_final_lstm, tune_lstm
from backend.ml.sequence_features import cases_to_sequence_examples, pad_sequences


def _prepare_split(cases) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequences, y, case_ids = cases_to_sequence_examples(cases)
    case_to_customer = {c.case_id: c.customer_id for c in cases}
    groups_str = [case_to_customer[cid] for cid in case_ids]
    unique = sorted(set(groups_str))
    customer_to_int = {cust: i for i, cust in enumerate(unique)}
    groups = np.array([customer_to_int[g] for g in groups_str])

    padded, lengths = pad_sequences(sequences)
    return padded, lengths, y, groups


def _oracle_auc_for_chain_distribution(cases) -> float:
    """
    Uses each case's STORED customer_recent_failure_pressure (the exact
    value in effect when generated) rather than recomputing it.
    """
    true_probs = []
    y = []
    for case in cases:
        for attempt in case.attempts:
            p = true_recovery_probability(
                decline_code=case.decline_code, amount=case.amount,
                attempt_number=attempt.attempt_number, hour_of_day=attempt.hour_of_day,
                is_near_payday=case.is_near_payday,
                customer_recent_failure_pressure=case.customer_recent_failure_pressure,
            )
            true_probs.append(p)
            y.append(1.0 if attempt.recovered else 0.0)
    return roc_auc_score(np.array(y), np.array(true_probs))


def _pressure_effect_sanity_check(cases) -> None:
    """
    Reports the observed recovery rate for high- vs low-pressure cases, AND
    a two-proportion z-test on the difference — not just raw percentages.
    A large swing on a small high-pressure sample (n~89 in earlier runs)
    could plausibly be noise; the z-test settles that question with a
    number instead of an eyeball judgment. Confirmed once at z=4.22,
    p=0.000025 — this is now permanent so every future rerun gets the same
    check automatically, not a one-off manual calculation.
    """
    import math

    high_pressure = [c for c in cases if c.customer_recent_failure_pressure > 0.5]
    low_pressure = [c for c in cases if c.customer_recent_failure_pressure < 0.1]

    if not (high_pressure and low_pressure):
        print("  Sanity check skipped — not enough cases in one of the pressure buckets.")
        return

    n1, n2 = len(low_pressure), len(high_pressure)
    p1 = sum(c.final_recovered for c in low_pressure) / n1
    p2 = sum(c.final_recovered for c in high_pressure) / n2

    print(f"  Sanity check — final recovery rate, low pressure (<0.1, n={n1}): {p1:.3f}")
    print(f"  Sanity check — final recovery rate, high pressure (>0.5, n={n2}): {p2:.3f}")

    # Two-proportion z-test (pooled), so "is this swing real or just a
    # small high-pressure sample" has an actual answer, not a guess.
    x1, x2 = n1 * p1, n2 * p2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        print("  z-test skipped — zero variance in pooled proportion (degenerate sample).")
        return

    z = (p1 - p2) / se
    # Standard normal CDF via erf — no scipy dependency needed for this.
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    se_diff = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    ci_low, ci_high = diff - 1.96 * se_diff, diff + 1.96 * se_diff

    print(f"  z-test: z={z:.3f}, p={p_value:.6f}, "
          f"difference={diff:.3f}, 95% CI=[{ci_low:.3f}, {ci_high:.3f}]")
    if p_value < 0.05:
        print("  -> statistically significant: the pressure effect is real, not sample noise.")
    else:
        print("  -> NOT statistically significant at this sample size — treat the effect")
        print("     as unconfirmed until more high-pressure cases accumulate.")


def main() -> None:
    print("=" * 70)
    print("STEP 5, COMPARISON POINT 4 — LSTM sequence model (architecture doc 6.3)")
    print("v2: includes causal customer-history (recency-weighted) effect")
    print("=" * 70)

    cases = generate_subscription_retry_sequences()
    print(f"\n{len(cases)} genuine retry-chain cases generated (soft-decline only).")
    _pressure_effect_sanity_check(cases)

    train_cases, val_cases, test_cases = entity_level_split(cases)
    print(f"\nEntity-level split (cases): {len(train_cases)} train / {len(val_cases)} val / {len(test_cases)} test")

    X_train, len_train, y_train, groups_train = _prepare_split(train_cases)
    X_val, len_val, y_val, _ = _prepare_split(val_cases)
    X_test, len_test, y_test, _ = _prepare_split(test_cases)
    print(f"Per-attempt training examples: {len(y_train)} train / {len(y_val)} val / {len(y_test)} test")

    chain_oracle_auc = _oracle_auc_for_chain_distribution(test_cases)
    print(f"\nOracle AUC for THIS chain-derived test distribution: {chain_oracle_auc:.4f}")
    print("(Not the same as backend/ml/oracle.py's flat-distribution number.)")

    print("\n--- LSTM, random search over real ranges ---")
    params, cv_auc = tune_lstm(X_train, len_train, y_train, groups_train, n_iter=15)
    print(f"  Best params: {params}")
    print(f"  CV AUC={cv_auc:.3f}")

    model = train_final_lstm(X_train, len_train, y_train, params)
    test_probs = predict_proba(model, X_test, len_test)
    test_auc = roc_auc_score(y_test, test_probs)
    print(f"  Test AUC={test_auc:.3f}")

    print("\n" + "=" * 70)
    print("Result")
    print("=" * 70)
    gap_to_own_oracle = chain_oracle_auc - test_auc
    print(f"  LSTM test AUC:                        {test_auc:.4f}")
    print(f"  Oracle ceiling for THIS distribution:  {chain_oracle_auc:.4f}")
    print(f"  Gap to own ceiling:                    {gap_to_own_oracle:.4f}")
    print()
    print("  IMPORTANT — this AUC is NOT apples-to-apples with GBM/NN's flat 0.693:")
    print("  the LSTM here has access to customer_recent_failure_pressure, a feature")
    print("  the flat models were never given.")
    if gap_to_own_oracle < 0:
        print("\n  NEGATIVE gap: investigate before trusting this number.")


if __name__ == "__main__":
    main()
