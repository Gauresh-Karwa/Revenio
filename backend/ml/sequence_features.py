"""
Sequence feature construction for the LSTM comparison point (architecture
doc 6.3). Mirrors backend/ml/features.py's role for the flat models.

TASK FORMULATION: attempt k's input is the sequence of all attempts in the
case up to and including attempt k (features only, never leaking attempt
k's own outcome). Reduces to the flat model's information content at
attempt 1, adds genuine prior-attempt context at attempt 2+.

PER-STEP FEATURES (v2 — customer_recent_failure_pressure added): decline_code
one-hot (6), is_night (1), is_near_payday (1), amount (1),
customer_recent_failure_pressure (1) — 10 total.

IMPORTANT CAVEAT: the flat models (GBM/NN in compare.py) do NOT have access
to customer_recent_failure_pressure — the flat generator has no cross-case
customer history at all. This means an LSTM trained with this feature is
not directly apples-to-apples comparable to GBM/NN's flat AUC anymore — see
compare_sequence.py's printed output for the explicit caveat.
"""

from __future__ import annotations

import numpy as np

SOFT_DECLINE_CODE_ORDER = ["51", "05", "91", "96", "65", "61"]
STEP_FEATURE_NAMES = [f"code_{c}" for c in SOFT_DECLINE_CODE_ORDER] + [
    "is_night",
    "is_near_payday",
    "amount",
    "customer_recent_failure_pressure",
]
N_STEP_FEATURES = len(STEP_FEATURE_NAMES)  # 10

NIGHT_HOURS = frozenset(range(0, 6))


def _step_features(
    decline_code: str,
    hour_of_day: int,
    is_near_payday: bool,
    amount: float,
    customer_recent_failure_pressure: float,
) -> list[float]:
    one_hot = [1.0 if decline_code == c else 0.0 for c in SOFT_DECLINE_CODE_ORDER]
    is_night = 1.0 if hour_of_day in NIGHT_HOURS else 0.0
    return one_hot + [
        is_night,
        1.0 if is_near_payday else 0.0,
        float(amount),
        float(customer_recent_failure_pressure),
    ]


def cases_to_sequence_examples(cases: list) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    case_ids: list[str] = []

    for case in cases:
        step_history: list[list[float]] = []
        for attempt in case.attempts:
            step_history.append(
                _step_features(
                    case.decline_code, attempt.hour_of_day, case.is_near_payday,
                    case.amount, case.customer_recent_failure_pressure,
                )
            )
            sequences.append(np.array(step_history, dtype=np.float32))
            labels.append(1.0 if attempt.recovered else 0.0)
            case_ids.append(case.case_id)

    return sequences, np.array(labels, dtype=np.float32), case_ids


def pad_sequences(sequences: list[np.ndarray], max_len: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    if max_len is None:
        max_len = int(lengths.max())

    n = len(sequences)
    padded = np.zeros((n, max_len, N_STEP_FEATURES), dtype=np.float32)
    for i, seq in enumerate(sequences):
        L = len(seq)
        padded[i, :L, :] = seq

    return padded, lengths
