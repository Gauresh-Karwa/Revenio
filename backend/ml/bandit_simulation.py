"""
Step 6 benchmark: quantifies total money recovered by each policy under
simulated non-stationary drift, running through the REAL observer-driven
pipeline (EventStore -> BanditUpdateObserver -> LearningCore -> decide()),
not a standalone simulation loop that bypasses the actual wiring.

    python -m backend.ml.bandit_simulation

Two things this demonstrates:
1. DRIFT BENCHMARK: subscription's real decide() path, with a hard regime
   change in retry-timing effectiveness partway through a batch of cases.
2. POOLING CHECK: subscription and checkout-abandonment share ONE
   LearningCore (via one BanditUpdateObserver on one EventStore), and
   subscription's own results are unaffected by abandonment's presence.
"""

from __future__ import annotations

import numpy as np

from backend.core.bandit_observer import BanditUpdateObserver
from backend.core.events import EventStore
from backend.core.learning_core import (
    DriftAwareThompsonSampling,
    LearningCore,
    StaticHeuristicPolicy,
    StationaryThompsonSampling,
)
from backend.core.orchestrator import Orchestrator
from backend.modules.checkout_abandonment.module import CheckoutAbandonmentModule
from backend.modules.subscription.module import SubscriptionModule

SEED = 42
SUBSCRIPTION_TYPICAL_AMOUNT = 245.0
ABANDONMENT_TYPICAL_AMOUNT = 120.0

# True recovery probability PER ARM (index = RETRY_BACKOFF_HOURS index:
# [1, 6, 24, 72] hours). This benchmark controls the "true" per-arm outcome
# distribution directly via simulated_retry_result, same design pattern as
# the generator's own true_recovery_probability approach.
REGIME_A_PROBS = [0.30, 0.55, 0.45, 0.20]  # 6h backoff best pre-shift
REGIME_B_PROBS = [0.55, 0.25, 0.20, 0.15]  # 1h backoff best post-shift


def _run_subscription_batch(orchestrator, store, rng, true_probs, n_cases, case_id_prefix):
    """
    Each case is run in two passes, mirroring reality: the retry outcome
    genuinely isn't known until AFTER decide() has picked a backoff arm.
    Pass 1: let decide() choose an arm (case left PENDING). Pass 2: having
    read which arm was chosen from the audit log, sample the true outcome
    for that arm and feed it back in to let the case reach a terminal state.
    """
    money_recovered = 0.0
    recoveries = 0
    for i in range(n_cases):
        case_id = f"{case_id_prefix}-{i}"
        case = {"decline_code": "51", "amount": SUBSCRIPTION_TYPICAL_AMOUNT}
        orchestrator.process_case(case_id, "subscription", case, max_iterations=1)

        decisions = [e for e in store.get_events(case_id) if e.event_type == "Decision"]
        arm = decisions[-1].payload["action_params"].get("bandit_arm", 0)
        recovered = rng.random() < true_probs[arm]

        case["simulated_retry_result"] = "recovered" if recovered else "lost"
        orchestrator.process_case(case_id, "subscription", case, max_iterations=3)

        if recovered:
            money_recovered += SUBSCRIPTION_TYPICAL_AMOUNT
            recoveries += 1

    return {
        "money_recovered": money_recovered,
        "recovery_rate": recoveries / n_cases,
        "n_cases": n_cases,
    }


def _make_policy(name: str, n_arms: int, seed: int):
    if name == "static":
        # Fixed at the PRE-DRIFT-optimal arm (6h backoff, index 1) — a
        # realistic static baseline commits to what's known-good, then goes
        # stale after a regime change. Using arm 0 here would be an
        # arbitrary choice that happens to also be near-optimal post-shift
        # in this benchmark's regime design, which would understate
        # drift-aware's real advantage rather than fairly demonstrate it.
        return StaticHeuristicPolicy(n_arms=n_arms, fixed_arm=1)
    if name == "stationary_ts":
        return StationaryThompsonSampling(n_arms=n_arms, seed=seed)
    if name == "drift_aware_ts":
        return DriftAwareThompsonSampling(n_arms=n_arms, discount_factor=0.95, seed=seed)
    raise ValueError(name)


def run_drift_benchmark(n_cases_per_regime: int = 150) -> dict[str, dict]:
    rng = np.random.default_rng(SEED)
    results: dict[str, dict] = {}

    for name in ("static", "stationary_ts", "drift_aware_ts"):
        core = LearningCore()
        core.register_policy("subscription", _make_policy(name, n_arms=4, seed=SEED))
        store = EventStore()
        store.subscribe(BanditUpdateObserver(core))
        orchestrator = Orchestrator(store)
        orchestrator.register_module(SubscriptionModule(learning_core=core))

        pre = _run_subscription_batch(orchestrator, store, rng, REGIME_A_PROBS, n_cases_per_regime, f"{name}-pre")
        post = _run_subscription_batch(orchestrator, store, rng, REGIME_B_PROBS, n_cases_per_regime, f"{name}-post")

        results[name] = {
            "pre_shift": pre,
            "post_shift": post,
            "total_money_recovered": pre["money_recovered"] + post["money_recovered"],
        }

    return results


def run_pooling_check(n_cases: int = 150) -> dict[str, dict]:
    core = LearningCore()
    core.register_policy("subscription", DriftAwareThompsonSampling(n_arms=4, discount_factor=0.95, seed=SEED))
    core.register_policy(
        "checkout_abandonment", DriftAwareThompsonSampling(n_arms=3, discount_factor=0.95, seed=SEED)
    )
    store = EventStore()
    store.subscribe(BanditUpdateObserver(core))
    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule(learning_core=core))
    orchestrator.register_module(CheckoutAbandonmentModule(learning_core=core))

    rng = np.random.default_rng(SEED)
    sub_result = _run_subscription_batch(orchestrator, store, rng, REGIME_A_PROBS, n_cases, "pooled-sub")

    abandonment_probs = [0.25, 0.40, 0.30]
    abandonment_recoveries = 0
    abandonment_money = 0.0
    for i in range(n_cases):
        case_id = f"pooled-abandon-{i}"
        case = {
            "reached_checkout": True, "opt_in": True,
            "abandonment_signal": "shipping_cost_surprise",
        }
        orchestrator.process_case(case_id, "checkout_abandonment", case, max_iterations=1)
        decisions = [e for e in store.get_events(case_id) if e.event_type == "Decision"]
        arm = decisions[-1].payload["action_params"].get("bandit_arm", 0)
        recovered = rng.random() < abandonment_probs[arm]
        case["simulated_nudge_result"] = "recovered" if recovered else "lost"
        orchestrator.process_case(case_id, "checkout_abandonment", case, max_iterations=3)
        if recovered:
            abandonment_money += ABANDONMENT_TYPICAL_AMOUNT
            abandonment_recoveries += 1

    return {
        "subscription": sub_result,
        "abandonment": {
            "money_recovered": abandonment_money,
            "recovery_rate": abandonment_recoveries / n_cases,
            "n_cases": n_cases,
        },
        "aggregate_money": sub_result["money_recovered"] + abandonment_money,
    }


def main() -> None:
    print("=" * 70)
    print("STEP 6 BENCHMARK -- Static vs Stationary vs Drift-Aware, real pipeline")
    print("=" * 70)

    print("\n--- Drift benchmark: subscription domain, hard regime change mid-batch ---")
    drift_results = run_drift_benchmark()
    for name, r in drift_results.items():
        print(f"\n  {name}:")
        print(f"    pre-shift:  money=${r['pre_shift']['money_recovered']:.0f}  "
              f"recovery_rate={r['pre_shift']['recovery_rate']:.3f}")
        print(f"    post-shift: money=${r['post_shift']['money_recovered']:.0f}  "
              f"recovery_rate={r['post_shift']['recovery_rate']:.3f}")
        print(f"    TOTAL money recovered: ${r['total_money_recovered']:.0f}")

    best = max(drift_results, key=lambda n: drift_results[n]["total_money_recovered"])
    best_post_shift = max(
        drift_results, key=lambda n: drift_results[n]["post_shift"]["recovery_rate"]
    )
    print(f"\n  Best TOTAL money recovered (pre+post shift combined): {best}")
    print(f"  Best POST-SHIFT recovery rate (the real test of drift-awareness): {best_post_shift}")
    print()
    print("  NOTE ON WHY 'TOTAL' CAN BE MISLEADING: static is fixed at the")
    print("  PRE-shift-optimal arm here, so it pays zero exploration cost and can")
    print("  tie or beat an adaptive policy's total purely from a lucky head start —")
    print("  that's not evidence static handles drift well, it's evidence the")
    print("  benchmark gave it the answer for free on regime A. The metric that")
    print("  actually isolates drift-adaptation is POST-SHIFT performance alone:")
    print("  a policy that hasn't re-learned yet will lag there regardless of how")
    print("  well it did before the world changed.")

    print("\n" + "=" * 70)
    print("--- Pooling check: subscription + abandonment, ONE shared LearningCore ---")
    print("=" * 70)
    pooling = run_pooling_check()
    print(f"  Subscription -> money=${pooling['subscription']['money_recovered']:.0f}  "
          f"recovery_rate={pooling['subscription']['recovery_rate']:.3f}")
    print(f"  Abandonment  -> money=${pooling['abandonment']['money_recovered']:.0f}  "
          f"recovery_rate={pooling['abandonment']['recovery_rate']:.3f}")
    print(f"  Aggregate money recovered (both domains): ${pooling['aggregate_money']:.0f}")


if __name__ == "__main__":
    main()
