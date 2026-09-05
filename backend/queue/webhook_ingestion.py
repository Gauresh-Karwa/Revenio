
"""
Architecture doc 9.4: "The webhook handler does one fast thing: validate
and durably enqueue the event... webhook delivery is deduplicated on
transaction ID to stay idempotent against Razorpay's own retry behavior."

This is that one fast thing. It does NOT call orchestrator.process_case()
directly â€” that's process_case_task's job, running on a worker, at
whatever rate the worker pool can sustain. This function's only job is:
is this transaction_id new? If so, enqueue and return immediately. If not
(Razorpay re-delivering the same webhook after a slow/missing ack), skip
the enqueue â€” a duplicate case_id/domain_type/payload appended twice would
otherwise double-process a real payment event.

Dedup mechanism: Redis SET ... NX EX, a single atomic round-trip (no
separate "check, then set" race window). TTL matches Razorpay's own
documented webhook-retry window (24h) with margin â€” old enough delivery
attempts fall out of the dedup window naturally rather than growing this
key set unboundedly forever.
"""

from __future__ import annotations

import os
from typing import Any

import redis

from backend.config import load_environment
from backend.queue.tasks import process_case_task

load_environment()
REDIS_URL = os.environ.get("REVENIO_REDIS_URL", "redis://localhost:6379/0")

# Sourced direction only (Razorpay retries webhooks for up to 24 hours on
# failure per their own docs); the exact TTL margin here is a judgment
# call, flagged the same way MAX_NUDGES is â€” not independently re-verified
# against Razorpay's current retry-window documentation in this pass.
DEDUP_TTL_SECONDS = 26 * 60 * 60  # 26h: 24h retry window + 2h margin

_redis_client: redis.Redis | None = None


def _get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def enqueue_webhook_event(
    transaction_id: str,
    case_id: str,
    domain_type: str,
    case_payload: dict[str, Any],
    redis_client: redis.Redis | None = None,
) -> bool:
    """
    Returns True if this delivery was new and got enqueued, False if it
    was a duplicate delivery of a transaction_id already seen within the
    dedup window (nothing enqueued in that case).
    """
    r = redis_client or _get_redis_client()
    dedup_key = f"webhook:seen:{transaction_id}"

    was_new = r.set(dedup_key, "1", nx=True, ex=DEDUP_TTL_SECONDS)
    if not was_new:
        return False

    process_case_task.delay(case_id, domain_type, case_payload)
    return True
