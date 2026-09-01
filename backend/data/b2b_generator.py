"""
Grounded synthetic data for the B2B receivables domain. Same discipline as
subscription_generator.py: real sourced direction (and magnitude, where a
real number exists) for every effect, flagged estimates where it doesn't.

AGING-BUCKET COLLECTION PROBABILITY — genuinely well-sourced, not a single
weak citation. Multiple independent sources (NACM/CCAA-reported industry
data, Crestmont Capital, Eagle Rock CFO's AR benchmarks report, Resolve's
write-off statistics) converge tightly on the same shape: >95% current
(0-30 days), ~85-90% at 31-60 days, ~70-80% at 61-90 days (NACM/CCAA data
specifically), ~50-60% at 91-120 days, ~20-30% beyond 120 days. This is a
STANDARD, industry-wide methodology (every accounting package generates an
"AR aging report" using exactly these bucket boundaries) — not an invented
curve. Midpoints of each cited range are used as bucket rates below.

SECTION 43B(h) TAX-DEDUCTION INCENTIVE — direction sourced (verified
separately, see backend/modules/b2b_receivables/module.py's docstring):
buyers who don't pay a Udyam-registered Micro/Small enterprise within the
15/45-day statutory window lose the tax deduction for that expense until
the year of actual payment. This creates a real, mechanistic incentive for
buyers to prioritize paying MSME-registered vendors over otherwise-similar
non-MSME vendors. Direction is grounded in that real mechanism; the
magnitude of the resulting probability boost is estimated, not sourced —
same footing as ATTEMPT_DECAY's magnitude in subscription_generator.py.

CUSTOMER-LEVEL PAYMENT-RELATIONSHIP PRESSURE — reuses subscription_generator's
update_causal_pressure/compute_pressure_from_customer_history AS-IS, not a
reimplementation. The underlying math (a causal, recency-weighted EWMA over
a customer's past outcomes) is genuinely domain-agnostic — a business that
has recently been slow/non-paying is plausibly likely to continue being so,
same "recent behavior predicts near-future behavior" logic already
grounded (RFM/collections practice) for the subscription domain. Reusing
the same function is the "one implementation, not duplicated" discipline
already applied everywhere else in this project.

DISPUTED / DND / OPTED-OUT INVOICES: mirror subscription_generator's
treatment of hard/stop decline codes exactly — these never reach a real,
automated chase (check_stop halts them before execute), so their outcome
is recovered=False BY CONSTRUCTION, never sampled. This is not a claim
that disputed invoices never get paid in reality (they do, eventually,
usually via a different, human-led process this system doesn't automate)
— it's a claim that the AUTOMATED system this project builds never
recovers them, which is the only outcome this generator needs to be
honest about.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from backend.data.subscription_generator import (
    compute_pressure_from_customer_history,
    update_causal_pressure,
)

# --- Aging-bucket base collection probability. Sourced (see module
# docstring) — midpoint of each cited range, in ascending day-threshold
# order. NOT individually re-derived per source; this is the standard
# AR-aging bucket shape used industry-wide.
AGING_BUCKETS: list[tuple[int, float]] = [
    (30, 0.97),           # >95% current -> midpoint of the >95% claim
    (60, 0.875),           # ~85-90%
    (90, 0.75),             # ~70-80%, NACM/CCAA-cited specifically
    (120, 0.55),           # ~50-60%
]
AGING_BUCKET_FLOOR = 0.25  # beyond 120 days: ~20-30%, midpoint

# Section 43B(h) incentive — direction sourced, magnitude estimated (flagged).
MSME_TAX_INCENTIVE_FACTOR = 1.12

# --- NOT individually sourced — Indian-market-specific Udyam registration
# rate, dispute rate, and DND/opt-out rates were not found in this round of
# research (the AR-aging sources above are largely US/global). Estimated,
# flagged, same footing as several subscription/checkout-abandonment
# generator constants.
MSME_REGISTRATION_RATE = 0.35
WRITTEN_AGREEMENT_RATE = 0.60
DND_RATE = 0.10
OPT_OUT_RATE = 0.05
DISPUTE_RATE = 0.08

# Days-overdue distribution among records this generator produces — this
# dataset represents invoices that have ALREADY entered the automated
# chase workflow (days_overdue >= 1 for every record, by construction;
# this module's check_stop never fires on a not-yet-due invoice in the
# first place). Bucket weights are directionally consistent with the cited
# sources (most overdue accounts cluster in the earlier buckets, tapering
# into a long tail) but the exact percentages are estimated, not
# individually sourced.
DAYS_OVERDUE_BUCKET_WEIGHTS = {
    (1, 30): 0.45,
    (31, 60): 0.25,
    (61, 90): 0.15,
    (91, 120): 0.08,
    (121, 240): 0.07,
}

INVOICE_AMOUNT_MU = 8.5   # lognormal -> median ~= exp(8.5) =~ 4915, right-skewed B2B invoice scale
INVOICE_AMOUNT_SIGMA = 1.0


def _aging_bucket_rate(days_overdue: int) -> float:
    for threshold, rate in AGING_BUCKETS:
        if days_overdue <= threshold:
            return rate
    return AGING_BUCKET_FLOOR


def true_payment_probability(
    days_overdue: int,
    is_msme_registered: bool,
    customer_recent_payment_pressure: float = 0.0,
) -> float:
    """
    The exact probability _sample_payment draws against — public, same
    "single source of truth for both sampling and oracle computation"
    pattern as subscription_generator.true_recovery_probability.
    """
    p = _aging_bucket_rate(days_overdue)

    if is_msme_registered:
        p *= MSME_TAX_INCENTIVE_FACTOR

    # Reuses subscription's customer-history factor SHAPE conceptually but
    # not its exact function (that one is named/scoped to subscription's
    # MIN_HISTORY_FACTOR constant) — B2B gets its own, smaller floor since
    # a business relationship is more durable than a single interrupted
    # subscription (going into default over one late invoice is a stronger
    # claim than one missed payment) — estimated, flagged.
    b2b_min_history_factor = 0.70
    p *= 1.0 - (1.0 - b2b_min_history_factor) * customer_recent_payment_pressure

    return max(0.02, min(0.97, p))


def _sample_payment(
    rng: random.Random,
    days_overdue: int,
    is_msme_registered: bool,
    customer_recent_payment_pressure: float,
) -> bool:
    p = true_payment_probability(days_overdue, is_msme_registered, customer_recent_payment_pressure)
    return rng.random() < p


def _sample_days_overdue(rng: random.Random) -> int:
    buckets = list(DAYS_OVERDUE_BUCKET_WEIGHTS.keys())
    weights = list(DAYS_OVERDUE_BUCKET_WEIGHTS.values())
    low, high = rng.choices(buckets, weights=weights, k=1)[0]
    return rng.randint(low, high)


@dataclass(frozen=True)
class B2BInvoiceRecord:
    case_id: str
    customer_id: str
    invoice_amount: float
    days_overdue: int
    is_msme_registered: bool
    has_written_agreement: bool
    on_dnd_registry: bool
    has_opted_out: bool
    is_disputed: bool
    customer_recent_payment_pressure: float = 0.0
    recovered: bool = False


def generate_b2b_dataset(
    n_customers: int = 4000,
    max_invoices_per_customer: int = 3,
    seed: int = 42,
    include_customer_history: bool = False,
) -> list[B2BInvoiceRecord]:
    """
    include_customer_history (default False): mirrors
    subscription_generator's own flag exactly — when True, a customer's
    invoices are treated as chronologically ordered and
    customer_recent_payment_pressure is tracked causally across them, using
    the SAME shared update_causal_pressure function subscription's
    generator uses (not a reimplementation). When False, every record's
    pressure is 0.0 and no extra randomness is consumed — output is
    deterministic and independent of this flag's existence.
    """
    rng = random.Random(seed)
    records: list[B2BInvoiceRecord] = []
    case_counter = 0

    for customer_index in range(n_customers):
        customer_id = f"biz-{customer_index:05d}"
        n_invoices = rng.randint(0, max_invoices_per_customer)
        pressure = 0.0

        for _ in range(n_invoices):
            invoice_amount = round(rng.lognormvariate(mu=INVOICE_AMOUNT_MU, sigma=INVOICE_AMOUNT_SIGMA), 2)
            days_overdue = _sample_days_overdue(rng)
            is_msme_registered = rng.random() < MSME_REGISTRATION_RATE
            has_written_agreement = rng.random() < WRITTEN_AGREEMENT_RATE
            on_dnd_registry = rng.random() < DND_RATE
            has_opted_out = rng.random() < OPT_OUT_RATE
            is_disputed = rng.random() < DISPUTE_RATE

            row_pressure = pressure if include_customer_history else 0.0

            # Mirrors subscription_generator's hard/stop-code treatment
            # exactly: never actively chased by this automated system, so
            # never sampled — recovered=False by construction, no RNG draw.
            if on_dnd_registry or has_opted_out or is_disputed:
                recovered = False
            else:
                recovered = _sample_payment(rng, days_overdue, is_msme_registered, row_pressure)

            case_counter += 1
            records.append(
                B2BInvoiceRecord(
                    case_id=f"b2b-case-{case_counter:06d}",
                    customer_id=customer_id,
                    invoice_amount=invoice_amount,
                    days_overdue=days_overdue,
                    is_msme_registered=is_msme_registered,
                    has_written_agreement=has_written_agreement,
                    on_dnd_registry=on_dnd_registry,
                    has_opted_out=has_opted_out,
                    is_disputed=is_disputed,
                    customer_recent_payment_pressure=row_pressure,
                    recovered=recovered,
                )
            )

            if include_customer_history:
                pressure = update_causal_pressure(pressure, recovered)

    return records
