
"""
Architecture doc 9.4 + 9.5, as real Celery tasks.

process_case_task: what a worker in the "case_processing" queue runs. Each
invocation loads the CURRENT durable bandit state for every domain that has
one seeded (PostgresBanditStore), builds a fresh Orchestrator against real
PostgreSQL (backend.wiring.build_orchestrator), subscribes a
QueuedBanditUpdateObserver (not the synchronous BanditUpdateObserver â€” see
that file), and runs the case.

HONEST CONSISTENCY NOTE: the bandit snapshot a worker loads at the start of
process_case_task is a point-in-time read, not a live connection to
whatever the single-writer consumer is doing concurrently on other cases'
updates. This is eventual consistency, standard for this queued-worker
pattern and NOT the same guarantee PostgresBanditStore.apply_update()
itself provides (that one is a genuine, tested, no-lost-writes guarantee
via row locking â€” see test_concurrent_updates_never_lose_a_write). A case
decided a few milliseconds after another case's reward update commits may
occasionally use a slightly stale policy snapshot. This is called out
explicitly rather than left implicit, matching the project's "flag rather
than guess" discipline.

apply_bandit_update_task: what a worker in the "bandit_updates" queue runs.
Thin wrapper around PostgresBanditStore.apply_update() â€” all the actual
single-writer correctness lives there (row lock), not here.
"""

from __future__ import annotations

from typing import Any

from backend.queue.celery_app import celery_app


@celery_app.task(name="backend.queue.tasks.process_case_task")
def process_case_task(
    case_id: str, domain_type: str, case_payload: dict[str, Any], max_iterations: int = 20
) -> dict[str, Any]:
    from backend.core.learning_core import LearningCore
    from backend.queue.queued_bandit_observer import QueuedBanditUpdateObserver
    from backend.storage.postgres_bandit_store import PostgresBanditStore
    from backend.storage.postgres_event_store import PostgresEventStore
    from backend.wiring import DATABASE_URL, build_orchestrator

    KNOWN_DOMAINS = ("subscription", "checkout_abandonment", "b2b_receivables", "mandate_retry")

    # The worker can receive the very first webhook before a web process has
    # constructed an orchestrator.  Initialize the idempotent schema before
    # reading bandit_policy_state; otherwise a clean deployment fails before
    # it can process its first task.  init_schema() uses CREATE ... IF NOT
    # EXISTS, so concurrent workers safely converge on the same schema.
    schema_store = PostgresEventStore(DATABASE_URL)
    try:
        schema_store.init_schema()
    finally:
        schema_store.close()

    bandit_store = PostgresBanditStore(DATABASE_URL)
    learning_core = LearningCore()
    for known_domain in KNOWN_DOMAINS:
        policy = bandit_store.load_policy(known_domain)
        if policy is not None:
            learning_core.register_policy(known_domain, policy)
    bandit_store.close()

    orchestrator, store = build_orchestrator(learning_core=learning_core)
    store.subscribe(QueuedBanditUpdateObserver())
    try:
        return orchestrator.process_case(case_id, domain_type, case_payload, max_iterations=max_iterations)
    finally:
        store.close()


@celery_app.task(name="backend.queue.tasks.apply_bandit_update_task")
def apply_bandit_update_task(domain_type: str, arm: int, reward: float) -> None:
    from backend.storage.postgres_bandit_store import PostgresBanditStore
    from backend.wiring import DATABASE_URL

    bandit_store = PostgresBanditStore(DATABASE_URL)
    try:
        bandit_store.apply_update(domain_type, arm, reward)
    finally:
        bandit_store.close()
