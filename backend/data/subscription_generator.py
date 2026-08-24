from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

SOFT_CODES = ["51", "05", "91", "96", "65", "61"]
HARD_CODES = ["04", "07", "12", "14", "15", "41", "43", "46", "57"]
STOP_CODES = ["R0", "R1", "R3"]

# Bucket weights: soft ~80% lands inside the sourced 70-90% band; hard/stop
# split the remainder. Not individually sourced.
BUCKET_WEIGHTS = {"soft": 0.80, "hard": 0.15, "stop": 0.05}

# Within soft: 51 and 05 set directly from the sourced 40.5%/7.5% figures,
# rescaled to soft-bucket share; the rest split the remainder. Flagged, not
# sourced, beyond 51 and 05.
SOFT_CODE_WEIGHTS = {
    "51": 0.45,   # anchored to the sourced 40.5%-of-all-failures figure
    "05": 0.12,   # anchored to the sourced 7.5%-of-all-failures figure
    "91": 0.16,
    "96": 0.13,
    "65": 0.08,
    "61": 0.06,
}

# NOT individually sourced — my own estimates within the real 60-70% aggregate.
CODE_BASE_RECOVERY_RATE = {
    "51": 0.55,
    "05": 0.35,
    "91": 0.75,
    "96": 0.70,
    "65": 0.50,
    "61": 0.45,
}

# Sourced direction (recovery diminishes sharply after first few attempts),
# NOT sourced magnitude.
ATTEMPT_DECAY = [1.0, 0.70, 0.45, 0.25]

# Sourced direction (Adyen ~2% lower at night), applied as a small multiplier.
NIGHT_HOURS = set(range(0, 6))
NIGHT_PENALTY = 0.98

# Sourced direction (insufficient-funds retries near payday do better),
# NOT sourced magnitude.
PAYDAY_BOOST = 1.25


@dataclass(frozen=True)
class SubscriptionRecord:
    case_id: str
    customer_id: str
    decline_code: str
    amount: float
    attempt_number: int
    hour_of_day: int
    is_near_payday: bool
    recovered: bool  # the label — stochastic, see module docstring


def _pick_decline_code(rng: random.Random) -> str:
    bucket = rng.choices(
        population=["soft", "hard", "stop"],
        weights=[BUCKET_WEIGHTS["soft"], BUCKET_WEIGHTS["hard"], BUCKET_WEIGHTS["stop"]],
        k=1,
    )[0]
    if bucket == "soft":
        codes = list(SOFT_CODE_WEIGHTS.keys())
        weights = list(SOFT_CODE_WEIGHTS.values())
        return rng.choices(codes, weights=weights, k=1)[0]
    if bucket == "hard":
        return rng.choice(HARD_CODES)
    return rng.choice(STOP_CODES)


def _sample_recovered(
    rng: random.Random,
    decline_code: str,
    attempt_number: int,
    hour_of_day: int,
    is_near_payday: bool,
    rates: dict[str, float] | None = None,
    payday_boost: float = PAYDAY_BOOST,
) -> bool:
    rates = rates or CODE_BASE_RECOVERY_RATE

    if decline_code not in rates:
        # Hard/stop codes never reach a real retry in our system (check_stop
        # halts before execute) — they get recorded with recovered=False by
        # construction, not sampled, since no retry is ever attempted.
        return False

    p = rates[decline_code]
    p *= ATTEMPT_DECAY[min(attempt_number - 1, len(ATTEMPT_DECAY) - 1)]

    if hour_of_day in NIGHT_HOURS:
        p *= NIGHT_PENALTY

    if decline_code == "51" and is_near_payday:
        p *= payday_boost

    # Real-world noise floor/ceiling — never fully deterministic even for the
    # most/least favorable case. This is part of what makes the label
    # genuinely learnable rather than a re-derivation of the rule lookup.
    p = max(0.02, min(0.97, p))

    return rng.random() < p


def generate_subscription_dataset(
    n_customers: int = 5000,
    max_failures_per_customer: int = 4,
    seed: int = 42,
    base_recovery_override: dict[str, float] | None = None,
    payday_boost_override: float | None = None,
) -> list[SubscriptionRecord]:
    """
    base_recovery_override / payday_boost_override exist ONLY so a second,
    deliberately different-parameter "regime B" dataset can be generated for
    the cross-distribution generalization test (architecture doc 5.1) —
    never used for the primary training dataset.
    """
    rng = random.Random(seed)
    rates = dict(CODE_BASE_RECOVERY_RATE)
    if base_recovery_override:
        rates.update(base_recovery_override)
    payday_boost = payday_boost_override if payday_boost_override is not None else PAYDAY_BOOST

    records: list[SubscriptionRecord] = []
    case_counter = 0

    for customer_index in range(n_customers):
        customer_id = f"cust-{customer_index:05d}"
        n_failures = rng.randint(0, max_failures_per_customer)

        for _ in range(n_failures):
            decline_code = _pick_decline_code(rng)
            amount = round(rng.lognormvariate(mu=5.5, sigma=0.6), 2)  # realistic right-skewed spend
            attempt_number = rng.randint(1, 3)
            hour_of_day = rng.randint(0, 23)
            is_near_payday = rng.random() < 0.3

            recovered = _sample_recovered(
                rng, decline_code, attempt_number, hour_of_day, is_near_payday,
                rates=rates, payday_boost=payday_boost,
            )

            case_counter += 1
            records.append(
                SubscriptionRecord(
                    case_id=f"case-{case_counter:06d}",
                    customer_id=customer_id,
                    decline_code=decline_code,
                    amount=amount,
                    attempt_number=attempt_number,
                    hour_of_day=hour_of_day,
                    is_near_payday=is_near_payday,
                    recovered=recovered,
                )
            )

    return records