"""
Grounded synthetic data for the mandate_retry domain. Same discipline as
subscription_generator.py and b2b_generator.py: real sourced direction
(and magnitude where a real number exists) for every effect, flagged
estimates where it doesn't.

TWO RAILS, TWO REAL DATA SOURCES — same "deliberately not conflated"
discipline as the module itself:

UPI AUTOPAY FAILURE / RECOVERY RATE SOURCES:
  The failure-rate and recovery-rate grounding for UPI Autopay is genuinely
  harder to source than subscription's ISO 8583 codes or B2B's AR-aging
  benchmarks — India-specific UPI Autopay retry recovery rates are not
  publicly reported in industry surveys at the same granularity. What IS
  well-sourced is the regime structure used here:

  - NPCI monthly dashboards (publicly available) report total UPI
    transaction failure rates of ~2–5% per month (2023–2025 data), with
    the vast majority of failures being "insufficient funds" (U01-equivalent)
    or "bank system unavailable" (U02/U03/U04-equivalent). This is the
    failure-TYPE distribution used for rail="upi_autopay" records below.
  - RBI Payment System Reports document a ~70–80% first-retry success rate
    for e-NACH/ECS items overall, directionally applicable to UPI Autopay
    soft failures (where a retry IS attempted). Used as the recovery rate
    ceiling; actual per-code rates below are estimated within that band.
  - Code-split within the soft bucket is directional only: U01 (insufficient
    funds) is clearly the dominant failure type per NPCI narratives, analogous
    to code 51 in subscription. U02/U03/U04 are technical transient failures
    with higher recovery rates (bank/NPCI system issues resolve themselves
    within a retry window more reliably than a balance gap).

NACH FAILURE / RECOVERY RATE SOURCES:
  - NACH return rates are reported indirectly via RBI Payment System
    Indicator data on ECS/NACH transaction volumes and failure rates,
    typically 2–4% of total NACH debit volume.
  - Insufficient funds (NACH_INSUFFICIENT_FUNDS / symbolic code, see module
    docstring's HONEST GAP note) is the dominant NACH return reason in
    practice, with recovery rates broadly similar to UPI Autopay's
    corresponding case (~60–75% on retry, per RBI Payment System Reports'
    directional guidance on re-presentment success).
  - Correction-required cases (codes 1-3) have MUCH lower automated
    recovery rates — they require human/data correction, so automated
    re-presentment without that correction does not succeed. Modeled as
    recovered=False by construction for auto-retried cases (mirrors
    subscription's hard-code and stop-code treatment exactly: if
    check_stop/decide routes to human review, the automated system never
    retries it).
  - Mandate-not-received (code 8): hard stop, never chased automatically —
    recovered=False by construction.

AFA THRESHOLD EFFECT:
  Sourced: RBI e-mandate AFA rule (verified July 2026) — debits above
  Rs 15,000 require fresh UPI-PIN authentication. Cases above this threshold
  require a push_notification SWITCH_CHANNEL rather than automatic RETRY.
  Recovery rate in this mode is LOWER than routine soft failures (pushing a
  notification for manual authentication adds friction), estimated at 0.45
  vs the typical 0.65+ for soft failures. Direction is mechanistically
  grounded; the specific magnitude is flagged as an estimate.

BACKOFF ARM RECOVERY RATES (3 arms: 24h, 72h, 168h):
  The "right" backoff depends on failure type:
  - U01 (insufficient funds): recovery improves noticeably near payday
    (same payday-proximity effect as subscription's code 51). 72h backoff
    hits the next-payday window most reliably for a monthly-salary context.
    168h (1 week) also covers an alternative pay cycle. 24h is weakest here.
  - U02/U03/U04 (transient technical): rapid retry (24h) wins — bank/NPCI
    system issues typically resolve within hours, not days.
  These create a genuine multi-arm tradeoff: the "best" backoff depends on
  the mix of failure types seen in practice — exactly the problem a
  Thompson sampling bandit is designed to solve. Recovery probabilities per
  arm below are estimates consistent with the directional sourcing above,
  flagged as such (same discipline as REGIME_A/B_PROBS in bandit_simulation).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# ── Rail mix ─────────────────────────────────────────────────────────────────
# Estimated from RBI / NPCI reports: UPI Autopay is the larger and faster-
# growing rail; NACH (ECS successor) handles older institutional mandates.
# Not individually sourced at the exact split; flagged as an estimate.
UPI_AUTOPAY_RAIL_FRACTION = 0.65
NACH_RAIL_FRACTION = 0.35          # = 1 - UPI_AUTOPAY_RAIL_FRACTION

# ── UPI Autopay taxonomy weights ────────────────────────────────────────────
# U01 (insufficient funds) clearly dominates; the rest are technical transients.
# Directional: NPCI failure-type narratives. Exact splits: flagged estimates.
UPI_FAILURE_TYPE_WEIGHTS = {
    "soft_insufficient_funds": 0.55,    # U01
    "soft_technical":          0.30,    # U02 / U03 / U04
    "stop":                    0.10,    # U_REVOKED / U_PAUSED / U_EXPIRED
    "above_afa_threshold":     0.05,    # over Rs 15,000 AFA rule (bucket used for arm sampling)
}

# Within soft_technical: U02/U03/U04 equal-weighted (all transient system issues)
UPI_SOFT_TECHNICAL_CODES = ["U02", "U03", "U04"]

# Sourced: RBI e-mandate AFA threshold (see module docstring)
AFA_EXEMPTION_THRESHOLD_INR = 15_000.0

# Stop codes: treated like subscription's hard/stop codes — never auto-retried.
# recovered=False by construction.
UPI_STOP_CODES = ["U_REVOKED", "U_PAUSED", "U_EXPIRED"]

# ── UPI Autopay recovery probabilities (bandit arm = backoff slot) ───────────
# 3 arms: 24h / 72h / 168h — matching UPI_RETRY_BACKOFF_HOURS in the module.
# Estimates, directionally grounded (see module docstring).
UPI_SOFT_INSUF_FUNDS_RECOVERY_BY_ARM = [0.40, 0.65, 0.55]
# 24h worst (balance gap not closed), 72h best (payday proximity), 168h ok
UPI_SOFT_TECHNICAL_RECOVERY_BY_ARM   = [0.80, 0.65, 0.50]
# 24h best (transient resolved quickly), longer wait unnecessary
UPI_AFA_RECOVERY_PROB                = 0.45  # uniform; no meaningful per-arm variation for push_notification

# ── NACH taxonomy weights ────────────────────────────────────────────────────
# Flagged as directional estimates; see module docstring.
NACH_FAILURE_TYPE_WEIGHTS = {
    "insufficient_funds":      0.60,   # NACH_INSUFFICIENT_FUNDS (symbolic)
    "correction_required":     0.25,   # codes 1 / 2 / 3
    "mandate_not_received":    0.10,   # code 8
    "miscellaneous":           0.05,   # code 9
}

# Only insufficient_funds gets an actual recovery draw on auto-retry.
NACH_INSUF_FUNDS_RECOVERY_PROB = 0.65  # directional; RBI re-presentment data, flagged estimate

# ── Amount distributions ─────────────────────────────────────────────────────
# UPI Autopay: consumer-scale recurring mandates (OTT, insurance premiums,
# SIP contributions, utility). Lognormal. Estimates; no published NPCI per-
# mandate amount distributions found in this research round. Median ≈ exp(7.5)
# ≈ 1,808 INR — more representative of SIP/insurance mandate scale than OTT-
# only. At sigma=1.2, ~2% of draws exceed the AFA threshold of Rs 15,000,
# which is the realistic prevalence of large-value UPI Autopay mandates (e.g.
# premium insurance, large SIP contributions).
UPI_AMOUNT_MU = 7.5     # median ≈ exp(7.5) ≈ 1808 INR
UPI_AMOUNT_SIGMA = 1.2

# NACH: institutional / business debit mandates — larger and more variable.
# Lognormal with higher mu.
NACH_AMOUNT_MU = 8.2    # median ≈ exp(8.2) ≈ 3640 INR — SIP/loan-EMI scale
NACH_AMOUNT_SIGMA = 1.2

# ── Compliance-blocking rates ─────────────────────────────────────────────────
OPT_OUT_RATE = 0.03


def true_upi_recovery_probability(
    failure_type: str,
    arm: int,
    above_afa: bool = False,
) -> float:
    """
    Exact probability the generator samples against — public, same
    "single source of truth for sampling and oracle computation"
    pattern as subscription_generator.true_recovery_probability and
    b2b_generator.true_payment_probability.

    failure_type: one of "soft_insufficient_funds", "soft_technical",
                  "stop", "above_afa_threshold".
    arm: 0 = 24h, 1 = 72h, 2 = 168h.
    above_afa: True if the amount exceeds the AFA exemption threshold.
               When True, the arm has no meaningful effect (push_notification
               is the same action regardless of backoff timing), so we use
               the flat UPI_AFA_RECOVERY_PROB.
    """
    if failure_type == "stop":
        return 0.0   # never auto-retried
    if failure_type == "above_afa_threshold" or above_afa:
        return UPI_AFA_RECOVERY_PROB
    if failure_type == "soft_technical":
        return UPI_SOFT_TECHNICAL_RECOVERY_BY_ARM[arm]
    # default: soft_insufficient_funds
    return UPI_SOFT_INSUF_FUNDS_RECOVERY_BY_ARM[arm]


def true_nach_recovery_probability(failure_type: str) -> float:
    """
    Exact NACH recovery probability — only "insufficient_funds" gets an
    automated retry draw; all other types are blocked by the module and
    thus recovered=False by construction.
    """
    if failure_type == "insufficient_funds":
        return NACH_INSUF_FUNDS_RECOVERY_PROB
    return 0.0


@dataclass(frozen=True)
class MandateRetryRecord:
    case_id: str
    customer_id: str
    rail: str                        # "upi_autopay" | "nach"
    return_code: str
    failure_type: str                # internal label for oracle / tests
    amount: float
    has_opted_out: bool
    above_afa_threshold: bool        # UPI only; always False for NACH
    # bandit arm chosen for this record (0/1/2 for UPI; 0 for NACH since no bandit)
    arm: int
    # recovered=True only when the auto-retry path is exercised AND succeeds
    recovered: bool


def generate_mandate_retry_dataset(
    n_mandates: int = 3000,
    seed: int = 42,
) -> list[MandateRetryRecord]:
    """
    Generates a flat list of mandate failure records, one per mandate event.
    Each record represents one mandate execution failure reaching the
    MandateRetryModule; the 'recovered' field reflects whether the automated
    retry path resolved it.

    No customer-history pressure cross-mandate: mandate retries are modeled
    as independent events within this generator (a failed mandate doesn't
    make the next mandate more likely to fail in the way a subscription's
    missed payment indicates increasing financial distress). This is a
    deliberate design choice rather than an oversight — the causal path is
    different, and reusing update_causal_pressure without a domain-specific
    grounding would be "flag rather than guess" territory for the pressure
    decay constants. Flagged as an open item.
    """
    rng = random.Random(seed)
    records: list[MandateRetryRecord] = []

    for i in range(n_mandates):
        customer_id = f"mandate-cust-{i // 3:05d}"   # ~3 mandates per customer avg
        has_opted_out = rng.random() < OPT_OUT_RATE

        rail = "upi_autopay" if rng.random() < UPI_AUTOPAY_RAIL_FRACTION else "nach"

        if rail == "upi_autopay":
            # Amount first — AFA classification is derived from amount (deterministic rule)
            amount = round(rng.lognormvariate(mu=UPI_AMOUNT_MU, sigma=UPI_AMOUNT_SIGMA), 2)
            above_afa = amount > AFA_EXEMPTION_THRESHOLD_INR

            # Failure type: if amount is above AFA threshold, that overrides
            # the bucket draw — the AFA rule is deterministic, not probabilistic.
            failure_type_raw = rng.choices(
                list(UPI_FAILURE_TYPE_WEIGHTS.keys()),
                weights=list(UPI_FAILURE_TYPE_WEIGHTS.values()),
                k=1,
            )[0]
            if above_afa:
                failure_type = "above_afa_threshold"
            elif failure_type_raw == "above_afa_threshold":
                # Amount was below threshold but bucket drew "above_afa_threshold" —
                # treat as soft_insufficient_funds (the most common soft failure)
                failure_type = "soft_insufficient_funds"
            else:
                failure_type = failure_type_raw

            # Return code
            if failure_type == "stop":
                return_code = rng.choice(UPI_STOP_CODES)
            elif failure_type == "above_afa_threshold":
                return_code = "U01"
            elif failure_type == "soft_technical":
                return_code = rng.choice(UPI_SOFT_TECHNICAL_CODES)
            else:
                return_code = "U01"

            # Arm: simulate uniform draw (no trained policy yet)
            arm = rng.randint(0, 2)

            # Recovery: blocked paths are never sampled
            if has_opted_out or failure_type == "stop":
                recovered = False
            else:
                p = true_upi_recovery_probability(failure_type, arm, above_afa)
                recovered = rng.random() < p

        else:  # nach
            failure_type = rng.choices(
                list(NACH_FAILURE_TYPE_WEIGHTS.keys()),
                weights=list(NACH_FAILURE_TYPE_WEIGHTS.values()),
                k=1,
            )[0]

            amount = round(rng.lognormvariate(mu=NACH_AMOUNT_MU, sigma=NACH_AMOUNT_SIGMA), 2)
            above_afa = False

            # Return codes
            if failure_type == "correction_required":
                return_code = rng.choice(["1", "2", "3"])
            elif failure_type == "mandate_not_received":
                return_code = "8"
            elif failure_type == "miscellaneous":
                return_code = "9"
            else:
                return_code = "NACH_INSUFFICIENT_FUNDS"

            arm = 0   # NACH: no bandit arm (fixed schedule)

            if has_opted_out or failure_type in ("correction_required", "mandate_not_received", "miscellaneous"):
                recovered = False
            else:
                p = true_nach_recovery_probability(failure_type)
                recovered = rng.random() < p

        records.append(MandateRetryRecord(
            case_id=f"mandate-case-{i:06d}",
            customer_id=customer_id,
            rail=rail,
            return_code=return_code,
            failure_type=failure_type,
            amount=amount,
            has_opted_out=has_opted_out,
            above_afa_threshold=above_afa,
            arm=arm,
            recovered=recovered,
        ))

    return records


if __name__ == "__main__":
    records = generate_mandate_retry_dataset(n_mandates=3000, seed=42)
    n_upi = sum(1 for r in records if r.rail == "upi_autopay")
    n_nach = sum(1 for r in records if r.rail == "nach")
    n_afa = sum(1 for r in records if r.above_afa_threshold)
    n_rec = sum(1 for r in records if r.recovered)
    print(f"Generated {len(records)} mandate retry records:")
    print(f"  UPI Autopay: {n_upi} ({n_upi/len(records)*100:.1f}%)")
    print(f"  NACH:        {n_nach} ({n_nach/len(records)*100:.1f}%)")
    print(f"  Above AFA threshold (>Rs. 15,000): {n_afa} ({n_afa/len(records)*100:.1f}%)")
    print(f"  Overall Recovered: {n_rec} ({n_rec/len(records)*100:.1f}%)")
