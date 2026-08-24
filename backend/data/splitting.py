from __future__ import annotations

import random
from typing import Any, Protocol


class HasCustomerId(Protocol):
    customer_id: str


def entity_level_split(
    records: list[Any],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 123,
) -> tuple[list[Any], list[Any], list[Any]]:
    """
    Splits by customer_id, not by row. Every record belonging to a given
    customer_id goes entirely into exactly one of train/val/test.
    """
    customer_ids = sorted({r.customer_id for r in records})
    rng = random.Random(seed)
    rng.shuffle(customer_ids)

    n = len(customer_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = set(customer_ids[:n_train])
    val_ids = set(customer_ids[n_train : n_train + n_val])
    test_ids = set(customer_ids[n_train + n_val :])

    train = [r for r in records if r.customer_id in train_ids]
    val = [r for r in records if r.customer_id in val_ids]
    test = [r for r in records if r.customer_id in test_ids]

    return train, val, test