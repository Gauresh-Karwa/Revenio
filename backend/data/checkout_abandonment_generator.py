from __future__ import annotations

import random
from dataclasses import dataclass

RECOVERABLE_SIGNALS = [
    "shipping_cost_surprise",
    "forced_account_creation",
    "payment_method_unavailable",
    "checkout_form_friction",
    "checkout_page_error",
    "distracted_high_intent",
]
NON_RECOVERABLE_SIGNALS = ["low_purchase_intent"]

# NOT sourced — see module docstring. Chosen to be directionally plausible
# (technical errors and distraction recover better than deep friction/policy
# issues like forced account creation) but not backed by a specific study.
SIGNAL_BASE_RECOVERY_RATE = {
    "shipping_cost_surprise": 0.30,
    "forced_account_creation": 0.20,
    "payment_method_unavailable": 0.25,
    "checkout_form_friction": 0.28,
    "checkout_page_error": 0.45,
    "distracted_high_intent": 0.35,
}

NUDGE_DECAY = [1.0, 0.65, 0.35]  # same sourced direction as subscription retries: diminishing returns


@dataclass(frozen=True)
class CheckoutAbandonmentRecord:
    case_id: str
    customer_id: str
    reached_checkout: bool
    abandonment_signal: str
    opt_in: bool
    amount: float
    nudge_number: int
    recovered: bool


def generate_checkout_abandonment_dataset(
    n_customers: int = 8000,
    max_abandonments_per_customer: int = 3,
    seed: int = 7,
    non_checkout_starter_rate: float = 0.55,  # most cart abandonment never reaches checkout (arch doc 5.2)
    opt_in_rate: float = 0.60,
) -> list[CheckoutAbandonmentRecord]:
    rng = random.Random(seed)
    records: list[CheckoutAbandonmentRecord] = []
    case_counter = 0

    for customer_index in range(n_customers):
        customer_id = f"cust-{customer_index:05d}"
        n_events = rng.randint(0, max_abandonments_per_customer)

        for _ in range(n_events):
            reached_checkout = rng.random() > non_checkout_starter_rate
            opt_in = rng.random() < opt_in_rate
            amount = round(rng.lognormvariate(mu=5.0, sigma=0.7), 2)
            nudge_number = rng.randint(1, 2)

            if not reached_checkout:
                signal = "n/a"
                recovered = False  # never diagnosed as recoverable, never nudged
            else:
                bucket_recoverable = rng.random() < 0.85  # matches doc 5.2: most checkout-starters are recoverable categories
                if bucket_recoverable:
                    signal = rng.choice(RECOVERABLE_SIGNALS)
                else:
                    signal = rng.choice(NON_RECOVERABLE_SIGNALS)

                if signal in NON_RECOVERABLE_SIGNALS or not opt_in:
                    recovered = False
                else:
                    p = SIGNAL_BASE_RECOVERY_RATE[signal]
                    p *= NUDGE_DECAY[min(nudge_number - 1, len(NUDGE_DECAY) - 1)]
                    p = max(0.02, min(0.90, p))
                    recovered = rng.random() < p

            case_counter += 1
            records.append(
                CheckoutAbandonmentRecord(
                    case_id=f"case-{case_counter:06d}",
                    customer_id=customer_id,
                    reached_checkout=reached_checkout,
                    abandonment_signal=signal,
                    opt_in=opt_in,
                    amount=amount,
                    nudge_number=nudge_number,
                    recovered=recovered,
                )
            )

    return records