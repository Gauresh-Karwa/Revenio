"""
Recomputes the oracle AUC ceiling directly from the generator's true
probability function (backend.data.subscription_generator.true_recovery_probability),
NOT estimated from a trained model. This is the same method architecture doc
6.5 used to get the original 0.694 figure — that figure is now STALE, since
it predates the code-51 amount-dependence added to the generator. This
script exists so the ceiling is re-anchored against the CURRENT generator,
not silently left pointing at a probability function that no longer exists.

    python -m backend.ml.oracle

Why this is legitimate (not leakage): the oracle uses the generator's true
per-row probability as the "prediction" and scores it against the same
generator's sampled binary outcome. Real models never get to see this true
probability — they only see the sampled 0/1 label and the features. The
oracle is a ceiling BECAUSE it's privileged information; comparing GBM/MLP
against it (not against 1.0) is what makes "how close is our model to the
best any model could do on this feature set" a meaningful question.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from backend.data.splitting import entity_level_split
from backend.data.subscription_generator import (
    CODE_BASE_RECOVERY_RATE,
    generate_subscription_dataset,
    true_recovery_probability,
)


def compute_oracle_ceiling(records) -> tuple[float, int]:
    """
    Returns (oracle_auc, n_soft_records). Restricted to soft-decline records
    only, matching every other AUC number in this project (hard/stop codes
    never reach a real retry, so they're excluded from the diagnosis-layer
    comparison entirely — see architecture doc 5.1).
    """
    soft_records = [r for r in records if r.decline_code in CODE_BASE_RECOVERY_RATE]

    y_true = np.array([1.0 if r.recovered else 0.0 for r in soft_records])
    true_probs = np.array(
        [
            true_recovery_probability(
                decline_code=r.decline_code,
                amount=r.amount,
                attempt_number=r.attempt_number,
                hour_of_day=r.hour_of_day,
                is_near_payday=r.is_near_payday,
            )
            for r in soft_records
        ]
    )
    return float(roc_auc_score(y_true, true_probs)), len(soft_records)


def main() -> None:
    records = generate_subscription_dataset()  # real default scale, matches every other step-5 number
    _, _, test_records = entity_level_split(records)

    full_auc, n_full = compute_oracle_ceiling(records)
    test_auc, n_test = compute_oracle_ceiling(test_records)

    print("=" * 70)
    print("ORACLE CEILING — recomputed against the CURRENT generator")
    print("(includes code-51 amount-dependence; supersedes the 0.694 figure")
    print("in architecture doc 5.1/6.5, which predates that change)")
    print("=" * 70)
    print(f"  Full dataset  (n={n_full}):  oracle AUC = {full_auc:.4f}")
    print(f"  Test split only (n={n_test}): oracle AUC = {test_auc:.4f}")
    print()
    print("  This is the number to compare GBM/MLP test AUC against going")
    print("  forward — not the old 0.694.")


if __name__ == "__main__":
    main()
