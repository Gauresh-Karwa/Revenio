from __future__ import annotations

import math
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

# NEW — code-51 (insufficient funds) amount-dependence. Mechanistic, not a
# free-floating tuning knob: "insufficient funds" is a gap between balance
# and requested amount, not a binary state, so recovery odds should fall as
# the requested amount rises relative to what's typical. Only applied to
# code 51 — every other soft code's generating function is unchanged, so
# this is additive, not a rework of the existing (already-tested) rates.
#
# Shape: a logistic (sigmoid) decay centered on the dataset's amount
# distribution median, not an arbitrary cutoff. amount ~ lognormal(mu=5.5,
# sigma=0.6) below, whose median is exp(mu) = exp(5.5) ≈ 244.69 — computed
# analytically, not eyeballed, so this stays correct if mu ever changes.
# factor(median) == 1.0 exactly (midpoint of MIN/MAX), so amounts at the
# median leave the base rate untouched; small amounts push the factor
# toward MAX_AMOUNT_FACTOR (easier to cover a small gap), large amounts
# toward MIN_AMOUNT_FACTOR (harder to cover a large one). A move from $10 to
# $50 sits on the steep part of the curve near a small median-relative
# amount; a move from $10,000 to $10,050 barely moves the factor at all —
# the log-scale AMOUNT_SCALE is what gives the curve that shape.
CODE_51_AMOUNT_MU = 5.5  # must match the amount lognormvariate mu below
MEDIAN_AMOUNT = math.exp(CODE_51_AMOUNT_MU)
AMOUNT_SCALE = MEDIAN_AMOUNT / 2  # controls steepness of the decay
MIN_AMOUNT_FACTOR = 0.75
MAX_AMOUNT_FACTOR = 1.25


def _code_51_amount_factor(amount: float) -> float:
    """
    Logistic decay in [MIN_AMOUNT_FACTOR, MAX_AMOUNT_FACTOR], centered at
    MEDIAN_AMOUNT. Pure function of amount — no RNG — so it's independently
    testable and deterministic given an amount.
    """
    return MIN_AMOUNT_FACTOR + (MAX_AMOUNT_FACTOR - MIN_AMOUNT_FACTOR) / (
        1 + math.exp((amount - MEDIAN_AMOUNT) / AMOUNT_SCALE)
    )


# --- Support-email / hardship-disclosure signal ---
SUPPORT_EMAIL_CONTACT_RATE = 0.15
HARDSHIP_RATE_AMONG_CONTACTS = 0.30
MIN_HARDSHIP_FACTOR = 0.55


def _hardship_factor(has_hardship: bool) -> float:
    return MIN_HARDSHIP_FACTOR if has_hardship else 1.0


HARDSHIP_EMAIL_TEMPLATES = [
    "I lost my job last week and can't cover this charge right now.",
    "Going through a medical emergency, please give me some time.",
    "I'm in a really tough financial situation at the moment.",
    "My hours got cut and money is tight this month.",
    "There's been a death in the family and finances are a mess right now.",
    "Things have been really rough for us lately, could we work something out?",
    "We're dealing with a hard stretch right now, hope you understand.",
]

NEUTRAL_EMAIL_TEMPLATES = [
    "Can you tell me when my card will be charged again?",
    "I'd like to update my payment method on file.",
    "Please cancel my subscription for next month.",
    "Why was my payment declined? My card should be fine.",
    "I want to change my billing email address.",
]


@dataclass(frozen=True)
class SubscriptionRecord:
    case_id: str
    customer_id: str
    decline_code: str
    amount: float
    attempt_number: int
    hour_of_day: int
    is_near_payday: bool
    recovered: bool
    customer_recent_failure_pressure: float = 0.0
    has_support_email: bool = False
    email_text: str | None = None
    true_hardship: bool = False


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


def true_recovery_probability(
    decline_code: str,
    amount: float,
    attempt_number: int,
    hour_of_day: int,
    is_near_payday: bool,
    rates: dict[str, float] | None = None,
    payday_boost: float = PAYDAY_BOOST,
    customer_recent_failure_pressure: float = 0.0,
    has_hardship: bool = False,
) -> float:
    """
    The exact probability _sample_recovered draws against — extracted as its
    own public function so an oracle-ceiling calculator (backend/ml/oracle.py)
    can call the SAME math the generator uses, rather than re-deriving it in
    a second place where the two could silently drift apart. Returns 0.0 for
    any code outside `rates` (hard/stop codes never reach a real retry, so
    their true probability of a real retry recovering is 0, not "unknown").

    customer_recent_failure_pressure defaults to 0.0 (neutral, no effect) —
    generate_subscription_dataset (the FLAT generator) never passes a
    non-default value, so its output and every existing test/oracle number
    computed from it are byte-identical to before this parameter existed.
    Only generate_subscription_retry_sequences (the CHAIN generator) passes
    a real, causally-computed value. See that function's docstring for the
    grounding.
    """
    rates = rates or CODE_BASE_RECOVERY_RATE

    if decline_code not in rates:
        return 0.0

    p = rates[decline_code]
    p *= ATTEMPT_DECAY[min(attempt_number - 1, len(ATTEMPT_DECAY) - 1)]

    if hour_of_day in NIGHT_HOURS:
        p *= NIGHT_PENALTY

    if decline_code == "51" and is_near_payday:
        p *= payday_boost

    if decline_code == "51":
        p *= _code_51_amount_factor(amount)

    p *= _customer_history_factor(customer_recent_failure_pressure)
    p *= _hardship_factor(has_hardship)

    return max(0.02, min(0.97, p))


def _sample_recovered(
    rng: random.Random,
    decline_code: str,
    amount: float,
    attempt_number: int,
    hour_of_day: int,
    is_near_payday: bool,
    rates: dict[str, float] | None = None,
    payday_boost: float = PAYDAY_BOOST,
    customer_recent_failure_pressure: float = 0.0,
    has_hardship: bool = False,
) -> bool:
    if decline_code not in (rates or CODE_BASE_RECOVERY_RATE):
        return False

    p = true_recovery_probability(
        decline_code, amount, attempt_number, hour_of_day, is_near_payday,
        rates=rates, payday_boost=payday_boost,
        customer_recent_failure_pressure=customer_recent_failure_pressure,
        has_hardship=has_hardship,
    )
    return rng.random() < p


def generate_subscription_dataset(
    n_customers: int = 5000,
    max_failures_per_customer: int = 4,
    seed: int = 42,
    base_recovery_override: dict[str, float] | None = None,
    payday_boost_override: float | None = None,
    include_customer_history: bool = False,
    include_support_email_signal: bool = False,
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
        pressure = 0.0

        for _ in range(n_failures):
            decline_code = _pick_decline_code(rng)
            amount = round(rng.lognormvariate(mu=CODE_51_AMOUNT_MU, sigma=0.6), 2)
            attempt_number = rng.randint(1, 3)
            hour_of_day = rng.randint(0, 23)
            is_near_payday = rng.random() < 0.3

            row_pressure = pressure if include_customer_history else 0.0

            has_support_email = False
            email_text = None
            true_hardship = False
            if include_support_email_signal:
                has_support_email = rng.random() < SUPPORT_EMAIL_CONTACT_RATE
                if has_support_email:
                    true_hardship = rng.random() < HARDSHIP_RATE_AMONG_CONTACTS
                    template_bank = HARDSHIP_EMAIL_TEMPLATES if true_hardship else NEUTRAL_EMAIL_TEMPLATES
                    email_text = rng.choice(template_bank)

            recovered = _sample_recovered(
                rng, decline_code, amount, attempt_number, hour_of_day, is_near_payday,
                rates=rates, payday_boost=payday_boost,
                customer_recent_failure_pressure=row_pressure,
                has_hardship=true_hardship,
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
                    customer_recent_failure_pressure=row_pressure,
                    has_support_email=has_support_email,
                    email_text=email_text,
                    true_hardship=true_hardship,
                )
            )

            if include_customer_history:
                pressure = update_causal_pressure(pressure, recovered)

    return records


MAX_CHAIN_ATTEMPTS = len(ATTEMPT_DECAY)  # 4 — beyond this, ATTEMPT_DECAY's
# index is clamped (see true_recovery_probability), so the true probability
# stops changing qualitatively; chaining further would add no new signal.

HISTORY_EWMA_ALPHA = 0.5  # recency weight: how much the MOST RECENT case's
# outcome dominates the running "failure pressure" vs older cases. Direction
# (recent matters more than old) is standard RFM/collections practice;
# 0.5 itself is an estimate, not individually sourced.
MIN_HISTORY_FACTOR = 0.65  # floor: a customer whose recent cases have all
# failed sees at most a 35% reduction in recovery probability — meaningful,
# not catastrophic. Estimated, flagged.


def _customer_history_factor(pressure: float) -> float:
    return 1.0 - (1.0 - MIN_HISTORY_FACTOR) * pressure


def update_causal_pressure(previous_pressure: float, this_case_recovered: bool) -> float:
    """
    THE single implementation of the causal, recency-weighted EWMA update —
    used by both generators below AND by live inference (see
    compute_pressure_from_customer_history and SubscriptionModule.diagnose).
    One implementation, not multiple copies that could silently drift apart.
    """
    outcome_signal = 0.0 if this_case_recovered else 1.0
    return HISTORY_EWMA_ALPHA * outcome_signal + (1.0 - HISTORY_EWMA_ALPHA) * previous_pressure


def compute_pressure_from_customer_history(past_case_outcomes: list[bool]) -> float:
    """
    Given a customer's PAST case outcomes in chronological order (True =
    recovered, False = lost), replays the same causal EWMA update used at
    generation time and returns the resulting pressure. This is what live
    inference calls (SubscriptionModule.diagnose), so training and
    inference compute this identically by construction. Empty list -> 0.0.
    """
    pressure = 0.0
    for recovered in past_case_outcomes:
        pressure = update_causal_pressure(pressure, recovered)
    return pressure


@dataclass(frozen=True)
class RetryAttempt:
    attempt_number: int
    hour_of_day: int
    recovered: bool  # True only for the final attempt in a recovered case


@dataclass(frozen=True)
class RetryCase:
    case_id: str
    customer_id: str
    decline_code: str
    amount: float
    is_near_payday: bool
    customer_recent_failure_pressure: float
    attempts: list[RetryAttempt]

    @property
    def final_recovered(self) -> bool:
        return self.attempts[-1].recovered if self.attempts else False


def _generate_one_retry_case(
    rng: random.Random,
    case_id: str,
    customer_id: str,
    decline_code: str,
    amount: float,
    is_near_payday: bool,
    rates: dict[str, float],
    payday_boost: float,
    customer_recent_failure_pressure: float,
) -> RetryCase:
    attempts: list[RetryAttempt] = []
    for attempt_number in range(1, MAX_CHAIN_ATTEMPTS + 1):
        hour_of_day = rng.randint(0, 23)
        recovered = _sample_recovered(
            rng, decline_code, amount, attempt_number, hour_of_day, is_near_payday,
            rates=rates, payday_boost=payday_boost,
            customer_recent_failure_pressure=customer_recent_failure_pressure,
        )
        attempts.append(
            RetryAttempt(attempt_number=attempt_number, hour_of_day=hour_of_day, recovered=recovered)
        )
        if recovered:
            break

    return RetryCase(
        case_id=case_id, customer_id=customer_id, decline_code=decline_code,
        amount=amount, is_near_payday=is_near_payday,
        customer_recent_failure_pressure=customer_recent_failure_pressure,
        attempts=attempts,
    )


def generate_subscription_retry_sequences(
    n_customers: int = 5000,
    max_cases_per_customer: int = 4,
    seed: int = 42,
) -> list[RetryCase]:

    rng = random.Random(seed)
    cases: list[RetryCase] = []
    case_counter = 0

    for customer_index in range(n_customers):
        customer_id = f"cust-{customer_index:05d}"
        n_cases = rng.randint(0, max_cases_per_customer)
        pressure = 0.0  # neutral prior — no history yet for this customer

        for _ in range(n_cases):
            decline_code = _pick_decline_code(rng)
            if decline_code not in CODE_BASE_RECOVERY_RATE:
                continue  # hard/stop codes never retry — no chain, no history update

            amount = round(rng.lognormvariate(mu=CODE_51_AMOUNT_MU, sigma=0.6), 2)
            is_near_payday = rng.random() < 0.3

            case_counter += 1
            case = _generate_one_retry_case(
                rng, f"seqcase-{case_counter:06d}", customer_id, decline_code,
                amount, is_near_payday, CODE_BASE_RECOVERY_RATE, PAYDAY_BOOST,
                customer_recent_failure_pressure=pressure,
            )
            cases.append(case)

            # Update pressure causally, using ONLY this case's own outcome,
            # for whichever case (if any) comes next for this customer.
            pressure = update_causal_pressure(pressure, case.final_recovered)

    return cases