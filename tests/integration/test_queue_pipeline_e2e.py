
"""
Requires a running Celery worker consuming "case_processing" and
"bandit_updates" (e.g. `celery -A backend.queue.celery_app worker -Q
case_processing,bandit_updates`), a running Redis, and a running Postgres
with REVENIO_DATABASE_URL pointing at it. Skips cleanly if the worker
doesn't respond within the timeout, rather than hanging CI forever.
"""

import os

import pytest
import redis as redis_lib

from backend.core.learning_core import StationaryThompsonSampling
from backend.queue.tasks import apply_bandit_update_task, process_case_task
from backend.queue.webhook_ingestion import enqueue_webhook_event
from backend.storage.postgres_bandit_store import PostgresBanditStore
from backend.storage.postgres_event_store import PostgresEventStore

DATABASE_URL = os.environ.get(
    "REVENIO_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/revenio"
)
REDIS_URL = os.environ.get("REVENIO_REDIS_URL", "redis://localhost:6379/0")
WORKER_TIMEOUT = 20


def _worker_available() -> bool:
    try:
        import socket
        from urllib.parse import urlparse
        p = urlparse(REDIS_URL)
        host = p.hostname or "localhost"
        port = p.port or 6379
        with socket.create_connection((host, port), timeout=0.5):
            pass
    except OSError:
        return False

    try:
        result = process_case_task.delay("healthcheck-case", "subscription", {"decline_code": "51"}, 1)
        result.get(timeout=WORKER_TIMEOUT)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _worker_available(), reason="No Celery worker consuming case_processing/bandit_updates is running"
)


@pytest.fixture
def event_store():
    store = PostgresEventStore(DATABASE_URL)
    store.init_schema()
    with store._pool.connection() as conn:
        conn.execute("TRUNCATE events, case_state")
        conn.commit()
    yield store
    store.close()


@pytest.fixture
def bandit_store():
    s = PostgresBanditStore(DATABASE_URL)
    with s._pool.connection() as conn:
        conn.execute("TRUNCATE bandit_policy_state")
        conn.commit()
    yield s
    s.close()


def test_process_case_task_runs_on_a_real_worker_and_persists_to_postgres(event_store):
    result = process_case_task.delay(
        "e2e-case-1", "subscription",
        {"decline_code": "51", "amount": 400.0, "simulated_retry_result": "recovered"},
        1,
    )
    final_state = result.get(timeout=WORKER_TIMEOUT)
    assert final_state["terminal_status"] == "RECOVERED"

    # Independent, fresh read â€” proves it's durably in Postgres, not just
    # in the task result payload.
    confirmed = event_store.derive_state("e2e-case-1")
    assert confirmed["terminal_status"] == "RECOVERED"


def test_apply_bandit_update_task_runs_on_a_real_worker(bandit_store):
    bandit_store.save_policy("subscription", StationaryThompsonSampling(n_arms=4, seed=1))

    r1 = apply_bandit_update_task.delay("subscription", 2, 1.0)
    r1.get(timeout=WORKER_TIMEOUT)
    r2 = apply_bandit_update_task.delay("subscription", 2, 0.0)
    r2.get(timeout=WORKER_TIMEOUT)

    policy = bandit_store.load_policy("subscription")
    assert policy.snapshot()["arms"][2]["pull_count"] == 2


def test_full_pipeline_bandit_arm_selected_and_reward_queued_back(event_store, bandit_store):
    """
    The complete loop: a case is processed on a real worker, which selects
    a bandit arm using a snapshot loaded from Postgres, and â€” because the
    worker subscribes QueuedBanditUpdateObserver, not the synchronous
    observer â€” the reward update is enqueued back onto bandit_updates and
    applied by (possibly a different) worker, landing durably.
    """
    bandit_store.save_policy("subscription", StationaryThompsonSampling(n_arms=4, seed=1))

    result = process_case_task.delay(
        "e2e-case-2", "subscription",
        {"decline_code": "51", "amount": 400.0, "simulated_retry_result": "recovered"},
        1,
    )
    final_state = result.get(timeout=WORKER_TIMEOUT)
    assert final_state["terminal_status"] == "RECOVERED"

    decisions = [p for p in final_state["history"] if "action_type" in p and p["action_type"] == "RETRY"]
    assert decisions, "expected a RETRY decision in the case history"
    assert "bandit_arm" in decisions[0]["action_params"]

    # The apply_bandit_update_task this triggers is itself async â€” give the
    # worker a moment to consume it, then confirm the arm was credited.
    import time

    for _ in range(20):
        policy = bandit_store.load_policy("subscription")
        if policy is not None and sum(a["pull_count"] for a in policy.snapshot()["arms"]) > 0:
            break
        time.sleep(0.5)

    assert policy is not None
    assert sum(a["pull_count"] for a in policy.snapshot()["arms"]) == 1


def test_webhook_dedup_prevents_double_processing(event_store):
    rc = redis_lib.Redis.from_url(REDIS_URL, decode_responses=True)
    rc.delete("webhook:seen:test-txn-999")

    first = enqueue_webhook_event(
        "test-txn-999", "wh-dedup-case", "subscription", {"decline_code": "51"}, redis_client=rc
    )
    second = enqueue_webhook_event(
        "test-txn-999", "wh-dedup-case", "subscription", {"decline_code": "51"}, redis_client=rc
    )

    assert first is True
    assert second is False

