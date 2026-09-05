
"""
Durable storage for LearningCore bandit policies, keyed by domain_type.
Reuses BanditPolicy.to_dict()/from_dict() (built in Step 6, unchanged here)
as the ONLY serialization format â€” no second format to keep in sync.

SINGLE-WRITER GUARANTEE, per architecture doc 9.5: enforced at the database
layer via SELECT ... FOR UPDATE inside apply_update()'s transaction, not
merely by convention (e.g. "just run one Celery worker"). This matters
because trusting worker-count=1 as the sole safety mechanism is fragile â€”
someone scales the worker pool for throughput six months from now and the
single-writer guarantee silently disappears with no error. A row lock is
correct regardless of how many workers are consuming the queue
concurrently: two workers racing to update the same domain_type's policy
will have one of them block at FOR UPDATE until the first transaction
commits, then read the POST-update state â€” never a lost update.
"""

from __future__ import annotations

from typing import Any

from psycopg_pool import ConnectionPool

from backend.core.learning_core import (
    BanditPolicy,
    DriftAwareThompsonSampling,
    StaticHeuristicPolicy,
    StationaryThompsonSampling,
)

# Registry keyed on exactly the string each policy's own to_dict() writes
# into "policy_type" â€” StaticHeuristicPolicy writes its own class name
# literally; ThompsonSamplingBandit's to_dict() uses type(self).__name__,
# which for the two real subclasses is "StationaryThompsonSampling" and
# "DriftAwareThompsonSampling". No guessing, no second mapping to drift
# out of sync with learning_core.py's own to_dict() implementations.
_POLICY_REGISTRY: dict[str, type[BanditPolicy]] = {
    "StaticHeuristicPolicy": StaticHeuristicPolicy,
    "StationaryThompsonSampling": StationaryThompsonSampling,
    "DriftAwareThompsonSampling": DriftAwareThompsonSampling,
}


class PostgresBanditStore:
    def __init__(self, conninfo: str, min_size: int = 1, max_size: int = 10) -> None:
        self._pool = ConnectionPool(conninfo, min_size=min_size, max_size=max_size, open=True)

    def close(self) -> None:
        self._pool.close()

    def save_policy(self, domain_type: str, policy: BanditPolicy) -> None:
        """
        Used once, at setup time, to seed durable state for a freshly
        constructed policy (e.g. a bandit that has never run before). NOT
        used by apply_update() below, which does its own locked
        read-modify-write in a single transaction.
        """
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO bandit_policy_state (domain_type, state, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (domain_type) DO UPDATE SET
                    state = EXCLUDED.state, updated_at = now()
                """,
                (domain_type, _to_jsonb(policy.to_dict())),
            )
            conn.commit()

    def load_policy(self, domain_type: str) -> BanditPolicy | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT state FROM bandit_policy_state WHERE domain_type = %s",
                (domain_type,),
            ).fetchone()
        if row is None:
            return None
        state = row[0]
        policy_class = _POLICY_REGISTRY[state["policy_type"]]
        return policy_class.from_dict(state)

    def apply_update(self, domain_type: str, arm: int, reward: float) -> None:
        """
        THE single-writer operation, per architecture doc 9.5. Locks the
        domain_type's row for the duration of the transaction (SELECT ...
        FOR UPDATE), so concurrent callers serialize correctly regardless
        of how many Celery workers are running this task concurrently.

        Raises KeyError if no policy has been seeded for this domain_type
        yet (via save_policy) â€” never silently no-ops, matching the
        project's "flag rather than guess" discipline; a missing policy
        here means setup was skipped, not that there's nothing to do.
        """
        with self._pool.connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    "SELECT state FROM bandit_policy_state WHERE domain_type = %s FOR UPDATE",
                    (domain_type,),
                ).fetchone()
                if row is None:
                    raise KeyError(
                        f"No bandit policy seeded for domain_type '{domain_type}' â€” "
                        "call save_policy() once at setup time before applying updates."
                    )

                state = row[0]
                policy_class = _POLICY_REGISTRY[state["policy_type"]]
                policy = policy_class.from_dict(state)
                policy.update(arm, reward)

                conn.execute(
                    "UPDATE bandit_policy_state SET state = %s, updated_at = now() WHERE domain_type = %s",
                    (_to_jsonb(policy.to_dict()), domain_type),
                )


def _to_jsonb(data: dict[str, Any]):
    from psycopg.types.json import Jsonb

    return Jsonb(data)

