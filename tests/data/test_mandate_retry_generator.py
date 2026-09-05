"""
Tests for backend/data/mandate_retry_generator.py.

Same discipline as test_b2b_generator.py:
- Sourced-range assertions where a real range exists.
- Direction-only assertions where only direction is sourced.
- Construction-guarantee assertions (stopped cases never recover by design).
- Oracle function (true_*_recovery_probability) tested as a pure function.
"""

from backend.data.mandate_retry_generator import (
    AFA_EXEMPTION_THRESHOLD_INR,
    NACH_INSUF_FUNDS_RECOVERY_PROB,
    UPI_SOFT_INSUF_FUNDS_RECOVERY_BY_ARM,
    UPI_SOFT_TECHNICAL_RECOVERY_BY_ARM,
    generate_mandate_retry_dataset,
    true_nach_recovery_probability,
    true_upi_recovery_probability,
)


# ---------------------------------------------------------------------------
# Oracle pure-function tests
# ---------------------------------------------------------------------------

def test_upi_stop_type_always_zero():
    for arm in range(3):
        assert true_upi_recovery_probability("stop", arm) == 0.0


def test_upi_afa_type_returns_flat_probability():
    # AFA recovery is flat across arms (push_notification is the same action)
    probs = {true_upi_recovery_probability("above_afa_threshold", arm) for arm in range(3)}
    assert len(probs) == 1  # all arms give the same rate


def test_upi_technical_arm0_beats_arm1_beats_arm2():
    """
    Transient technical failures recover fastest at 24h (arm 0),
    where the bank/NPCI system has already resolved the issue.
    Sourced direction (see generator docstring).
    """
    p0 = true_upi_recovery_probability("soft_technical", 0)
    p1 = true_upi_recovery_probability("soft_technical", 1)
    p2 = true_upi_recovery_probability("soft_technical", 2)
    assert p0 > p1 > p2
    assert p0 == UPI_SOFT_TECHNICAL_RECOVERY_BY_ARM[0]


def test_upi_insuf_funds_arm1_is_best():
    """
    Insufficient-funds (U01) recovery peaks at the 72h arm (payday-proximity
    effect, same direction as subscription's code-51 payday boost).
    """
    p0 = true_upi_recovery_probability("soft_insufficient_funds", 0)
    p1 = true_upi_recovery_probability("soft_insufficient_funds", 1)
    p2 = true_upi_recovery_probability("soft_insufficient_funds", 2)
    assert p1 > p0
    assert p1 > p2
    assert p1 == UPI_SOFT_INSUF_FUNDS_RECOVERY_BY_ARM[1]


def test_upi_above_afa_flag_overrides_failure_type():
    """above_afa=True overrides failure_type, same prob regardless of type."""
    p_insuf = true_upi_recovery_probability("soft_insufficient_funds", 1, above_afa=True)
    p_technical = true_upi_recovery_probability("soft_technical", 0, above_afa=True)
    assert p_insuf == p_technical


def test_nach_stop_types_always_zero():
    for ft in ("correction_required", "mandate_not_received", "miscellaneous"):
        assert true_nach_recovery_probability(ft) == 0.0


def test_nach_insuf_funds_matches_constant():
    assert true_nach_recovery_probability("insufficient_funds") == NACH_INSUF_FUNDS_RECOVERY_PROB


# ---------------------------------------------------------------------------
# Generator structural guarantees
# ---------------------------------------------------------------------------

def test_dataset_has_both_rails():
    records = generate_mandate_retry_dataset(n_mandates=2000, seed=42)
    rails = {r.rail for r in records}
    assert "upi_autopay" in rails
    assert "nach" in rails


def test_opted_out_never_recover():
    records = generate_mandate_retry_dataset(n_mandates=3000, seed=1)
    opted_out = [r for r in records if r.has_opted_out]
    assert len(opted_out) > 0
    assert all(not r.recovered for r in opted_out)


def test_upi_stop_codes_never_recover():
    records = generate_mandate_retry_dataset(n_mandates=3000, seed=2)
    stopped = [r for r in records if r.rail == "upi_autopay" and r.failure_type == "stop"]
    assert len(stopped) > 0
    assert all(not r.recovered for r in stopped)


def test_nach_correction_required_never_recover():
    records = generate_mandate_retry_dataset(n_mandates=3000, seed=3)
    blocked = [r for r in records if r.rail == "nach" and r.failure_type in (
        "correction_required", "mandate_not_received", "miscellaneous"
    )]
    assert len(blocked) > 0
    assert all(not r.recovered for r in blocked)


def test_above_afa_threshold_flag_set_when_amount_exceeds_threshold():
    # UPI_AMOUNT_MU=7.5, sigma=1.2: ~2% of draws exceed Rs 15,000.
    # n_mandates=5000 produces ~3250 UPI records -> ~65 above threshold.
    records = generate_mandate_retry_dataset(n_mandates=5000, seed=42)
    upi = [r for r in records if r.rail == "upi_autopay"]
    above = [r for r in upi if r.above_afa_threshold]
    assert len(above) > 0, "Expected some UPI records above AFA threshold at this dataset size"
    assert all(r.amount > AFA_EXEMPTION_THRESHOLD_INR for r in above)


def test_nach_above_afa_threshold_always_false():
    records = generate_mandate_retry_dataset(n_mandates=2000, seed=5)
    nach = [r for r in records if r.rail == "nach"]
    assert all(not r.above_afa_threshold for r in nach)


def test_nach_arm_always_zero():
    """NACH has no bandit arm — generator must always emit arm=0."""
    records = generate_mandate_retry_dataset(n_mandates=2000, seed=6)
    nach = [r for r in records if r.rail == "nach"]
    assert all(r.arm == 0 for r in nach)


def test_upi_arm_is_in_valid_range():
    records = generate_mandate_retry_dataset(n_mandates=2000, seed=7)
    upi = [r for r in records if r.rail == "upi_autopay"]
    assert all(0 <= r.arm <= 2 for r in upi)


def test_above_afa_failure_type_only_when_above_threshold():
    """above_afa_threshold failure_type must never appear with amount <= threshold."""
    records = generate_mandate_retry_dataset(n_mandates=5000, seed=42)
    wrong = [r for r in records
             if r.failure_type == "above_afa_threshold"
             and r.amount <= AFA_EXEMPTION_THRESHOLD_INR]
    assert wrong == []


# ---------------------------------------------------------------------------
# Direction-only sourced assertions on sampled data
# ---------------------------------------------------------------------------

def test_upi_technical_24h_arm_recovers_more_than_168h_arm_in_data():
    """
    Arm 0 (24h) has a higher base recovery probability than arm 2 (168h)
    for technical failures. This should be visible in sampled data with
    enough records.
    """
    records = generate_mandate_retry_dataset(n_mandates=5000, seed=42)
    technical = [r for r in records
                 if r.rail == "upi_autopay" and r.failure_type == "soft_technical"]
    arm0 = [r for r in technical if r.arm == 0]
    arm2 = [r for r in technical if r.arm == 2]
    assert len(arm0) > 30 and len(arm2) > 30
    rate0 = sum(r.recovered for r in arm0) / len(arm0)
    rate2 = sum(r.recovered for r in arm2) / len(arm2)
    assert rate0 > rate2


def test_upi_insuf_funds_72h_arm_recovers_more_than_24h_arm_in_data():
    """Arm 1 (72h, payday proximity) beats arm 0 (24h) for insufficient funds."""
    records = generate_mandate_retry_dataset(n_mandates=5000, seed=42)
    insuf = [r for r in records
             if r.rail == "upi_autopay" and r.failure_type == "soft_insufficient_funds"]
    arm0 = [r for r in insuf if r.arm == 0]
    arm1 = [r for r in insuf if r.arm == 1]
    assert len(arm0) > 30 and len(arm1) > 30
    rate0 = sum(r.recovered for r in arm0) / len(arm0)
    rate1 = sum(r.recovered for r in arm1) / len(arm1)
    assert rate1 > rate0


def test_nach_insuf_funds_overall_recovery_in_sourced_range():
    """~60–75% directional recovery on NACH insufficient-funds re-presentment."""
    records = generate_mandate_retry_dataset(n_mandates=5000, seed=42)
    insuf = [r for r in records
             if r.rail == "nach" and r.failure_type == "insufficient_funds"]
    assert len(insuf) > 50
    rate = sum(r.recovered for r in insuf) / len(insuf)
    assert 0.50 <= rate <= 0.80   # generous bounds around sourced ~65%
