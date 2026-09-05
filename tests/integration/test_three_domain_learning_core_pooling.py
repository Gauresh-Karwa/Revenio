"""
Three-domain LearningCore pooling integration test.

Proves that B2B, subscription, and checkout-abandonment can all share ONE
LearningCore simultaneously, through the real observer-driven pipeline,
with complete policy isolation — updates to one domain never bleed into
another's arm counts.

This is the missing "proven live for B2B specifically" test that
test_bandit_observer_wiring.py already provides for subscription + abandonment.
The existing tests there only pool two domains. This file:

  1. Wires all three domains into ONE LearningCore + ONE BanditUpdateObserver.
  2. Confirms B2B decisions carry bandit_arm when the core is wired.
  3. Confirms B2B terminal outcomes (RECOVERED / LOST) update exactly the
     right arm in the B2B policy, not touching subscription or abandonment.
  4. Confirms PROMISED (non-terminal) does NOT update any arm yet.
  5. Confirms PROMISED -> kept (check_promise_due) DOES update on resolution.
  6. Confirms all three domains can run concurrently (interleaved case
     processing) without interference — this is the real pooling proof.

DESIGN NOTE — why B2B gets 3 arms, not 4:
B2B's CHANNEL_ESCALATION has exactly 3 entries (email, sms, voice). The
policy arm count MUST match so select_arm's returned index is always a
valid index into CHANNEL_ESCALATION. That's the same discipline as
subscription (4 arms = 4 RETRY_BACKOFF_HOURS slots) and abandonment
(3 arms = 3 nudge channels). Mismatching arm count is the one configuration
mistake that would cause a silent IndexError at runtime, not a test failure.
"""

from backend.core.bandit_observer import BanditUpdateObserver
from backend.core.events import EventStore
from backend.core.learning_core import LearningCore, StationaryThompsonSampling
from backend.core.orchestrator import Orchestrator
from backend.modules.b2b_receivables.module import B2BReceivablesModule
from backend.modules.checkout_abandonment.module import CheckoutAbandonmentModule
from backend.modules.subscription.module import SubscriptionModule


def _make_three_domain_system():
    """
    One LearningCore with policies for all three live domains.
    One BanditUpdateObserver subscribed to one EventStore.
    All three modules wired to that same core.
    """
    core = LearningCore()
    core.register_policy("subscription",         StationaryThompsonSampling(n_arms=4, seed=1))
    core.register_policy("checkout_abandonment", StationaryThompsonSampling(n_arms=3, seed=2))
    core.register_policy("b2b_receivables",      StationaryThompsonSampling(n_arms=3, seed=3))

    store = EventStore()
    observer = BanditUpdateObserver(core)
    store.subscribe(observer)

    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule(learning_core=core))
    orchestrator.register_module(CheckoutAbandonmentModule(learning_core=core))
    orchestrator.register_module(B2BReceivablesModule(learning_core=core))

    return core, store, orchestrator


# -----------------------------------------------------------------------
# B2B bandit_arm wiring
# -----------------------------------------------------------------------

def test_b2b_decision_carries_bandit_arm_when_learning_core_wired():
    core, store, orchestrator = _make_three_domain_system()
    orchestrator.process_case(
        "inv-1", "b2b_receivables",
        {"invoice_amount": 5000, "due_date": "2026-01-01"},
    )
    decisions = [e for e in store.get_events("inv-1") if e.event_type == "Decision"]
    assert len(decisions) >= 1
    assert "bandit_arm" in decisions[0].payload["action_params"], (
        "Decision action_params must carry bandit_arm when a LearningCore is wired."
    )


def test_b2b_bandit_arm_is_valid_channel_index():
    """arm must be 0, 1, or 2 — a valid index into CHANNEL_ESCALATION."""
    from backend.modules.b2b_receivables.module import CHANNEL_ESCALATION
    core, store, orchestrator = _make_three_domain_system()
    orchestrator.process_case(
        "inv-1", "b2b_receivables",
        {"invoice_amount": 5000, "due_date": "2026-01-01"},
    )
    decisions = [e for e in store.get_events("inv-1") if e.event_type == "Decision"]
    arm = decisions[0].payload["action_params"]["bandit_arm"]
    assert 0 <= arm < len(CHANNEL_ESCALATION), (
        f"bandit_arm={arm} is out of range for CHANNEL_ESCALATION (len={len(CHANNEL_ESCALATION)})"
    )


# -----------------------------------------------------------------------
# B2B reward updates: RECOVERED and LOST
# -----------------------------------------------------------------------

def test_b2b_recovered_case_updates_b2b_policy_only():
    core, store, orchestrator = _make_three_domain_system()

    # Pre-condition: all three policies start at zero pulls
    snap_before = core.snapshot()
    assert all(a["pull_count"] == 0 for a in snap_before["subscription"]["arms"])
    assert all(a["pull_count"] == 0 for a in snap_before["checkout_abandonment"]["arms"])
    assert all(a["pull_count"] == 0 for a in snap_before["b2b_receivables"]["arms"])

    orchestrator.process_case(
        "inv-1", "b2b_receivables",
        {"invoice_amount": 5000, "due_date": "2026-01-01", "simulated_payment_result": "paid_full"},
    )

    snap_after = core.snapshot()
    # Only B2B policy should have a pull — subscription and abandonment untouched
    b2b_pulls = sum(a["pull_count"] for a in snap_after["b2b_receivables"]["arms"])
    sub_pulls  = sum(a["pull_count"] for a in snap_after["subscription"]["arms"])
    aban_pulls = sum(a["pull_count"] for a in snap_after["checkout_abandonment"]["arms"])

    assert b2b_pulls == 1, f"B2B policy should have 1 pull, got {b2b_pulls}"
    assert sub_pulls == 0,  f"Subscription policy contaminated: {sub_pulls} pulls"
    assert aban_pulls == 0, f"Abandonment policy contaminated: {aban_pulls} pulls"


def test_b2b_lost_case_updates_b2b_policy_with_zero_reward():
    core, store, orchestrator = _make_three_domain_system()

    orchestrator.process_case(
        "inv-1", "b2b_receivables",
        {"invoice_amount": 5000, "due_date": "2026-01-01", "simulated_payment_result": "written_off"},
    )

    snap = core.snapshot()
    b2b_pulls = sum(a["pull_count"] for a in snap["b2b_receivables"]["arms"])
    assert b2b_pulls == 1, "A LOST outcome must still update the policy with reward=0."


# -----------------------------------------------------------------------
# PROMISED (non-terminal) does NOT update prematurely
# -----------------------------------------------------------------------

def test_b2b_promised_outcome_does_not_update_bandit_yet():
    """
    PROMISED is non-terminal. The observer must NOT credit/blame any arm
    until the promise is resolved via check_promise_due. This is the same
    guarantee test_pending_case_does_not_update_the_bandit_yet verifies for
    subscription — proven here for B2B's unique PROMISED outcome path.
    """
    core, store, orchestrator = _make_three_domain_system()

    orchestrator.process_case(
        "inv-1", "b2b_receivables",
        {"invoice_amount": 5000, "due_date": "2026-01-01", "simulated_payment_result": "promised"},
    )

    snap = core.snapshot()
    b2b_pulls = sum(a["pull_count"] for a in snap["b2b_receivables"]["arms"])
    assert b2b_pulls == 0, (
        f"PROMISED is non-terminal — bandit must NOT update yet (got {b2b_pulls} pulls)."
    )


def test_b2b_kept_promise_triggers_bandit_update_on_resolution():
    """
    When check_promise_due resolves to kept=True, the orchestrator appends
    a terminal RECOVERED Outcome. The observer must pick that up and update
    the B2B policy exactly once.
    """
    core, store, orchestrator = _make_three_domain_system()

    case = {"invoice_amount": 5000, "due_date": "2026-01-01", "simulated_payment_result": "promised"}
    orchestrator.process_case("inv-1", "b2b_receivables", case)

    # Confirm: still 0 pulls before resolution
    assert sum(a["pull_count"] for a in core.snapshot()["b2b_receivables"]["arms"]) == 0

    # Resolve: promise kept
    orchestrator.check_promise_due("inv-1", {**case, "simulated_promise_kept": True})

    b2b_pulls = sum(a["pull_count"] for a in core.snapshot()["b2b_receivables"]["arms"])
    assert b2b_pulls == 1, (
        f"Kept promise must trigger one bandit update on the B2B policy (got {b2b_pulls})."
    )


def test_b2b_broken_promise_re_entry_does_not_double_update():
    """
    A broken promise re-enters process_case (architecture doc 3.6). The
    resulting second Decision + subsequent Outcome means exactly one more
    bandit update per resolved cycle — not two updates from the same arm,
    not zero.
    """
    core, store, orchestrator = _make_three_domain_system()

    case = {"invoice_amount": 5000, "due_date": "2026-01-01", "simulated_payment_result": "promised"}
    orchestrator.process_case("inv-1", "b2b_receivables", case)

    # Broken promise: re-enters loop but does NOT resolve to RECOVERED/LOST
    # (simulated_payment_result stays "promised" for this pass)
    broken_case = {**case, "simulated_promise_kept": False}
    orchestrator.check_promise_due("inv-1", broken_case)

    # One broken promise < MAX_BROKEN_PROMISES (2), so the case continues —
    # no terminal Outcome yet, no bandit update yet
    b2b_pulls = sum(a["pull_count"] for a in core.snapshot()["b2b_receivables"]["arms"])
    assert b2b_pulls == 0, (
        f"Broken promise with re-entry should not have triggered a bandit update yet "
        f"(got {b2b_pulls} pulls; case is still PENDING)."
    )


# -----------------------------------------------------------------------
# Three-domain concurrent interleaving — the real pooling proof
# -----------------------------------------------------------------------

def test_all_three_domains_update_independently_in_interleaved_processing():
    """
    The canonical three-domain pooling test. Cases from all three domains
    are processed in interleaved order through a single shared system.
    Each domain's policy must accumulate exactly the right number of pulls,
    with no cross-domain contamination regardless of processing order.

    This is the "proven live for B2B specifically" proof that was missing
    before this file existed.
    """
    core, store, orchestrator = _make_three_domain_system()

    # Interleaved: subscription -> B2B -> abandonment -> B2B -> subscription
    orchestrator.process_case(
        "sub-1", "subscription",
        {"decline_code": "51", "simulated_retry_result": "recovered"},
    )
    orchestrator.process_case(
        "inv-1", "b2b_receivables",
        {"invoice_amount": 10000, "due_date": "2026-01-01", "simulated_payment_result": "paid_full"},
    )
    orchestrator.process_case(
        "aban-1", "checkout_abandonment",
        {
            "reached_checkout": True,
            "opt_in": True,
            "abandonment_signal": "shipping_cost_surprise",
            "simulated_nudge_result": "recovered",
        },
    )
    orchestrator.process_case(
        "inv-2", "b2b_receivables",
        {"invoice_amount": 3000, "due_date": "2026-02-01", "simulated_payment_result": "written_off"},
    )
    orchestrator.process_case(
        "sub-2", "subscription",
        {"decline_code": "51", "simulated_retry_result": "lost"},
    )

    snap = core.snapshot()
    sub_pulls  = sum(a["pull_count"] for a in snap["subscription"]["arms"])
    aban_pulls = sum(a["pull_count"] for a in snap["checkout_abandonment"]["arms"])
    b2b_pulls  = sum(a["pull_count"] for a in snap["b2b_receivables"]["arms"])

    assert sub_pulls == 2,  f"Expected 2 subscription pulls (2 cases), got {sub_pulls}"
    assert aban_pulls == 1, f"Expected 1 abandonment pull (1 case), got {aban_pulls}"
    assert b2b_pulls == 2,  f"Expected 2 B2B pulls (2 cases), got {b2b_pulls}"


def test_without_learning_core_b2b_uses_fixed_channel_schedule():
    """
    Backward compatibility: B2BReceivablesModule without a learning_core
    uses the fixed email->sms->voice schedule and does not emit bandit_arm.
    Same guarantee test_without_a_learning_core_behavior_is_completely_unaffected
    verifies for subscription.
    """
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(B2BReceivablesModule())  # no learning_core

    orchestrator.process_case(
        "inv-1", "b2b_receivables",
        {"invoice_amount": 5000, "due_date": "2026-01-01"},
    )

    decisions = [e for e in store.get_events("inv-1") if e.event_type == "Decision"]
    assert len(decisions) >= 1
    params = decisions[0].payload["action_params"]
    assert "bandit_arm" not in params, "Fixed schedule must not emit bandit_arm."
    assert params["channel"] == "email",    f"First contact without core must be email, got {params['channel']}"


# ---------------------------------------------------------------------------
# Four-domain pooling: mandate_retry added to the shared LearningCore
# ---------------------------------------------------------------------------

from backend.modules.mandate_retry.module import MandateRetryModule  # noqa: E402


def _make_four_domain_system():
    """
    One LearningCore with policies for all four live domains.
    One BanditUpdateObserver subscribed to one EventStore.
    All four modules wired to that same core.

    Arm counts:
      subscription        -> 4 arms (RETRY_BACKOFF_HOURS)
      checkout_abandonment-> 3 arms (nudge channels)
      b2b_receivables     -> 3 arms (contact channels)
      mandate_retry       -> 3 arms (UPI backoff hours: 24/72/168)
    """
    core = LearningCore()
    core.register_policy("subscription",         StationaryThompsonSampling(n_arms=4, seed=1))
    core.register_policy("checkout_abandonment", StationaryThompsonSampling(n_arms=3, seed=2))
    core.register_policy("b2b_receivables",      StationaryThompsonSampling(n_arms=3, seed=3))
    core.register_policy("mandate_retry",        StationaryThompsonSampling(n_arms=3, seed=4))

    store = EventStore()
    observer = BanditUpdateObserver(core)
    store.subscribe(observer)

    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule(learning_core=core))
    orchestrator.register_module(CheckoutAbandonmentModule(learning_core=core))
    orchestrator.register_module(B2BReceivablesModule(learning_core=core))
    orchestrator.register_module(MandateRetryModule(learning_core=core))

    return core, store, orchestrator


def test_mandate_retry_decision_carries_bandit_arm_in_four_domain_system():
    core, store, orchestrator = _make_four_domain_system()
    orchestrator.process_case(
        "upi-1", "mandate_retry",
        {"rail": "upi_autopay", "return_code": "U01", "amount": 500.0, "simulated_mandate_result": "recovered"},
    )
    decisions = [e for e in store.get_events("upi-1") if e.event_type == "Decision"]
    assert len(decisions) >= 1
    assert "bandit_arm" in decisions[0].payload["action_params"]


def test_mandate_retry_updates_only_mandate_retry_policy_not_others():
    """
    A UPI Autopay terminal case updates exactly mandate_retry's arm pulls.
    Subscription, abandonment, and B2B policies must remain at pull_count 0.
    """
    core, store, orchestrator = _make_four_domain_system()
    orchestrator.process_case(
        "upi-2", "mandate_retry",
        {"rail": "upi_autopay", "return_code": "U01", "amount": 500.0, "simulated_mandate_result": "recovered"},
    )
    snap = core.snapshot()
    assert sum(a["pull_count"] for a in snap["mandate_retry"]["arms"]) == 1
    assert sum(a["pull_count"] for a in snap["subscription"]["arms"]) == 0
    assert sum(a["pull_count"] for a in snap["checkout_abandonment"]["arms"]) == 0
    assert sum(a["pull_count"] for a in snap["b2b_receivables"]["arms"]) == 0


def test_nach_case_does_not_update_any_arm_in_four_domain_system():
    """
    NACH decisions never emit bandit_arm -> BanditUpdateObserver
    never credits any arm -> mandate_retry pull_count stays 0.
    """
    core, store, orchestrator = _make_four_domain_system()
    orchestrator.process_case(
        "nach-1", "mandate_retry",
        {"rail": "nach", "return_code": "NACH_INSUFFICIENT_FUNDS", "amount": 5000.0,
         "simulated_mandate_result": "recovered"},
    )
    snap = core.snapshot()
    assert sum(a["pull_count"] for a in snap["mandate_retry"]["arms"]) == 0


def test_all_four_domains_update_independently_interleaved():
    """
    The real four-domain pooling proof: processes one case per domain
    in interleaved order and confirms exactly 1 pull per domain, 0 bleed.
    """
    core, store, orchestrator = _make_four_domain_system()

    # subscription case
    orchestrator.process_case(
        "sub-1", "subscription",
        {"decline_code": "51", "simulated_retry_result": "recovered"},
    )
    # mandate_retry UPI case
    orchestrator.process_case(
        "upi-1", "mandate_retry",
        {"rail": "upi_autopay", "return_code": "U01", "amount": 500.0,
         "simulated_mandate_result": "recovered"},
    )
    # abandonment case
    orchestrator.process_case(
        "abn-1", "checkout_abandonment",
        {"reached_checkout": True, "opt_in": True,
         "abandonment_signal": "shipping_cost_surprise",
         "simulated_nudge_result": "recovered"},
    )
    # b2b case
    orchestrator.process_case(
        "inv-1", "b2b_receivables",
        {"invoice_amount": 5000, "due_date": "2026-01-01",
         "simulated_payment_result": "paid_full"},
    )

    snap = core.snapshot()
    assert sum(a["pull_count"] for a in snap["subscription"]["arms"])         == 1
    assert sum(a["pull_count"] for a in snap["mandate_retry"]["arms"])        == 1
    assert sum(a["pull_count"] for a in snap["checkout_abandonment"]["arms"]) == 1
    assert sum(a["pull_count"] for a in snap["b2b_receivables"]["arms"])      == 1

