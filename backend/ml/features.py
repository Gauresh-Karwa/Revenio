"""
Canonical feature construction for the subscription diagnosis-layer model.

This is the ONE place feature engineering happens. Both the trainer
(train_subscription_model.py) and the inference path (SubscriptionModule)
import from here — never re-implement feature construction in either place.
A drift between "how the model was trained" and "how the module builds
features at inference time" is silent and produces a model that returns
confident garbage without ever raising an error, which is worse than no
model at all.

Feature set is exactly what architecture doc section 6.5 states the
generator's true probability function depends on: decline code, attempt
number, night-hours, and payday — plus `amount`, now that code 51 has a
real amount-dependence (see subscription_generator.py). Only known soft
codes are scoreable; hard/stop/unmapped codes were never part of the
training distribution (check_stop halts before a retry is ever attempted
for hard/stop codes) and must never be passed through this function.
"""

from __future__ import annotations

from typing import Any

# Fixed order — the trainer builds its training matrix in this exact column
# order, and the module builds its single-row inference vector in the same
# order. Changing this list means re-training, not just re-loading.
SOFT_DECLINE_CODE_ORDER = ["51", "05", "91", "96", "65", "61"]

FEATURE_NAMES: list[str] = [
    f"code_{code}" for code in SOFT_DECLINE_CODE_ORDER
] + [
    "attempt_number",
    "is_night",
    "is_near_payday",
    "amount",
]


def build_feature_vector(
    decline_code: str,
    attempt_number: int,
    hour_of_day: int,
    is_near_payday: bool,
    amount: float,
    night_hours: frozenset[int] = frozenset(range(0, 6)),
) -> list[float]:
    """
    Builds one feature row, in FEATURE_NAMES order. Only valid for codes in
    SOFT_DECLINE_CODE_ORDER — callers (module and trainer alike) are
    responsible for filtering to known soft codes before calling this.
    """
    if decline_code not in SOFT_DECLINE_CODE_ORDER:
        raise ValueError(
            f"build_feature_vector called with non-soft decline_code={decline_code!r}; "
            "hard/stop/unmapped codes were never in the training distribution."
        )

    one_hot = [1.0 if decline_code == code else 0.0 for code in SOFT_DECLINE_CODE_ORDER]
    is_night = 1.0 if hour_of_day in night_hours else 0.0

    return one_hot + [
        float(attempt_number),
        is_night,
        1.0 if is_near_payday else 0.0,
        float(amount),
    ]


def build_feature_vector_from_case(case: dict[str, Any]) -> list[float]:
    """
    Convenience wrapper for the orchestrator's case dict shape (module-side
    inference), rather than the generator's SubscriptionRecord shape
    (trainer-side). Same underlying feature construction either way.
    """
    return build_feature_vector(
        decline_code=case["decline_code"],
        attempt_number=case.get("attempt_number", 1),
        hour_of_day=case.get("hour_of_day", 12),
        is_near_payday=case.get("is_near_payday", False),
        amount=case.get("amount", 0.0),
    )


# ---------------------------------------------------------------------------
# DataFrame and matrix helpers — used by compare.py, trainer, and test suite.
# ---------------------------------------------------------------------------


def records_to_frame(records: list) -> "pd.DataFrame":
    """
    Converts a list of SubscriptionRecord objects to a pandas DataFrame,
    filtering to soft-decline codes only (hard/stop codes were never part
    of the training distribution and must not be scored).
    """
    import pandas as pd
    from backend.data.subscription_generator import CODE_BASE_RECOVERY_RATE

    soft_records = [r for r in records if r.decline_code in CODE_BASE_RECOVERY_RATE]
    return pd.DataFrame(
        [
            {
                "case_id": r.case_id,
                "customer_id": r.customer_id,
                "decline_code": r.decline_code,
                "amount": r.amount,
                "attempt_number": r.attempt_number,
                "hour_of_day": r.hour_of_day,
                "is_near_payday": r.is_near_payday,
                "recovered": int(r.recovered),
            }
            for r in soft_records
        ]
    )


def build_feature_matrix(
    df: "pd.DataFrame",
) -> "tuple[np.ndarray, np.ndarray, list[str], np.ndarray]":
    import numpy as np

    X = np.array(
        [
            build_feature_vector(
                decline_code=row["decline_code"],
                attempt_number=int(row["attempt_number"]),
                hour_of_day=int(row["hour_of_day"]),
                is_near_payday=bool(row["is_near_payday"]),
                amount=float(row["amount"]),
            )
            for _, row in df.iterrows()
        ]
    )
    y = df["recovered"].to_numpy().astype(int)

    unique_customers = sorted(df["customer_id"].unique())
    customer_to_int = {cid: i for i, cid in enumerate(unique_customers)}
    groups = np.array([customer_to_int[cid] for cid in df["customer_id"]])

    return X, y, FEATURE_NAMES, groups


def fit_scaler(X_train: "np.ndarray") -> "sklearn.preprocessing.StandardScaler":
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def build_group_kfold(n_splits: int = 5) -> "GroupKFold":
    from sklearn.model_selection import GroupKFold

    return GroupKFold(n_splits=n_splits)