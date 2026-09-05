
import os
import socket
import threading
from urllib.parse import urlparse

import pytest

from backend.core.learning_core import StationaryThompsonSampling
from backend.storage.postgres_bandit_store import PostgresBanditStore

TEST_DB_URL = os.environ.get(
    "REVENIO_TEST_DB_URL", "postgresql://postgres:postgres@localhost:5432/revenio_test"
)


def _postgres_available(url: str = TEST_DB_URL) -> bool:
    try:
        p = urlparse(url)
        host = p.hostname or "localhost"
        port = p.port or 5432
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(), reason="PostgreSQL is not reachable on REVENIO_TEST_DB_URL"
)


@pytest.fixture
def store():
    s = PostgresBanditStore(TEST_DB_URL, min_size=2, max_size=8)
    with s._pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bandit_policy_state (
                domain_type TEXT PRIMARY KEY,
                state JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.commit()
    yield s
    with s._pool.connection() as conn:
        conn.execute("TRUNCATE bandit_policy_state")
        conn.commit()
    s.close()


def test_save_then_load_round_trips_exactly(store):
    policy = StationaryThompsonSampling(n_arms=4, seed=1)
    policy.update(arm=2, reward=1.0)
    store.save_policy("subscription", policy)

    loaded = store.load_policy("subscription")
    assert loaded.to_dict() == policy.to_dict()


def test_load_unknown_domain_returns_none(store):
    assert store.load_policy("nonexistent") is None


def test_apply_update_raises_for_unseeded_domain(store):
    with pytest.raises(KeyError):
        store.apply_update("never_seeded", arm=0, reward=1.0)


def test_apply_update_persists_and_is_reloadable(store):
    store.save_policy("subscription", StationaryThompsonSampling(n_arms=4, seed=1))
    store.apply_update("subscription", arm=1, reward=1.0)
    store.apply_update("subscription", arm=1, reward=0.0)

    loaded = store.load_policy("subscription")
    arm_1 = loaded.snapshot()["arms"][1]
    assert arm_1["pull_count"] == 2


def test_concurrent_updates_never_lose_a_write():
    """
    The real point of this file. 8 threads, each opening its OWN
    connection via the pool (simulating 8 concurrent Celery worker
    processes all consuming the same queue), each calling apply_update
    N times on the SAME domain_type. Without the FOR UPDATE lock, two
    threads reading the same stale state and writing back would silently
    lose one thread's updates (last-writer-wins on the whole blob, not a
    true increment). With the lock, every single update must land â€”
    total pull_count across all arms must equal N_THREADS * N_UPDATES_EACH
    exactly, not less.
    """
    store = PostgresBanditStore(TEST_DB_URL, min_size=8, max_size=16)
    with store._pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bandit_policy_state (
                domain_type TEXT PRIMARY KEY,
                state JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute("TRUNCATE bandit_policy_state")
        conn.commit()

    store.save_policy("concurrency_test", StationaryThompsonSampling(n_arms=3, seed=1))

    N_THREADS = 8
    N_UPDATES_EACH = 15
    errors = []

    def worker():
        try:
            for i in range(N_UPDATES_EACH):
                store.apply_update("concurrency_test", arm=i % 3, reward=float(i % 2))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []

    final = store.load_policy("concurrency_test")
    total_pulls = sum(a["pull_count"] for a in final.snapshot()["arms"])
    assert total_pulls == N_THREADS * N_UPDATES_EACH

    with store._pool.connection() as conn:
        conn.execute("TRUNCATE bandit_policy_state")
        conn.commit()
    store.close()

