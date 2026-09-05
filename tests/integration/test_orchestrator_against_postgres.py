
"""
The whole point of this file: zero changes to Orchestrator, BanditUpdateObserver,
or any domain module were needed to swap storage backends. These are the SAME
scenarios as tests/core/test_orchestrator.py and
tests/integration/test_bandit_observer_wiring.py, run against
PostgresEventStore instead of the in-memory EventStore.
"""

import os
import socket
from urllib.parse import urlparse

import pytest

from backend.core.bandit_observer import BanditUpdateObserver
from backend.core.learning_core import LearningCore, StationaryThompsonSampling
from backend.core.orchestrator import Orchestrator
from backend.modules.dummy.module import DummyModule
from backend.modules.subscription.module import SubscriptionModule
from backend.storage.postgres_event_store import PostgresEventStore

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
def pg():
    store = PostgresEventStore(TEST_DB_URL, min_size=1, max_size=4)
    store.init_schema()
    yield store
    with store._pool.connection() as conn:
        conn.execute("TRUNCATE events, case_state")
        conn.commit()
    store.close()


def test_dummy_module_runs_full_loop_and_stops_against_postgres(pg):
    orchestrator = Orchestrator(pg)
    orchestrator.register_module(DummyModule())

    final_state = orchestrator.process_case("case-1", "dummy", {})

    assert final_state["exists"] is True
    assert final_state["terminal"] is True
    assert final_state["terminal_status"] == "STOPPED:RESOLVED"

    events = pg.get_events("case-1")
    event_types = [e.event_type for e in events]
    assert "StopDecision" in event_types
    assert "Diagnosis" in event_types
    assert "Decision" in event_types
    assert "ExecutionResult" in event_types
    assert "Outcome" in event_types


def test_hard_decline_stops_immediately_no_retry_attempted_against_postgres(pg):
    orchestrator = Orchestrator(pg)
    orchestrator.register_module(SubscriptionModule())

    final_state = orchestrator.process_case("case-1", "subscription", {"decline_code": "43"})

    assert final_state["terminal_status"] == "STOPPED:COMPLIANCE_LIMIT"
    executions = [e for e in pg.get_events("case-1") if e.event_type == "ExecutionResult"]
    assert executions == []


def test_soft_decline_recovers_when_simulated_against_postgres(pg):
    orchestrator = Orchestrator(pg)
    orchestrator.register_module(SubscriptionModule())

    case = {"decline_code": "51", "amount": 499.0}
    orchestrator.process_case("case-1", "subscription", case, max_iterations=1)

    outcome_events = [e for e in pg.get_events("case-1") if e.event_type == "Outcome"]
    assert outcome_events[-1].payload["status"] == "PENDING"

    case["simulated_retry_result"] = "recovered"
    final_state = orchestrator.process_case("case-1", "subscription", case, max_iterations=1)
    assert final_state["terminal_status"] == "RECOVERED"


def test_cross_case_customer_pressure_survives_a_real_database_against_postgres(pg):
    """
    This one specifically exercises get_customer_case_history â€” a real
    SQL query, not an in-memory list comprehension â€” proving
    customer_recent_failure_pressure still flows correctly end-to-end.
    """
    orchestrator = Orchestrator(pg)
    orchestrator.register_module(SubscriptionModule())

    orchestrator.process_case(
        "case-1", "subscription",
        {"decline_code": "51", "customer_id": "cust-beta", "simulated_retry_result": "lost"},
    )
    orchestrator.process_case(
        "case-2", "subscription",
        {"decline_code": "05", "customer_id": "cust-beta", "simulated_retry_result": "lost"},
    )

    diag_2 = [e for e in pg.get_events("case-2") if e.event_type == "Diagnosis"][0]
    assert diag_2.payload["raw_signal"]["n_past_cases_considered"] == 1
    assert diag_2.payload["raw_signal"]["customer_recent_failure_pressure"] > 0.0


def test_bandit_observer_wiring_against_postgres(pg):
    core = LearningCore()
    core.register_policy("subscription", StationaryThompsonSampling(n_arms=4, seed=1))
    pg.subscribe(BanditUpdateObserver(core))

    orchestrator = Orchestrator(pg)
    orchestrator.register_module(SubscriptionModule(learning_core=core))

    before = core.snapshot()["subscription"]["arms"]
    assert all(a["pull_count"] == 0 for a in before)

    orchestrator.process_case(
        "case-1", "subscription",
        {"decline_code": "51", "simulated_retry_result": "recovered"},
    )

    after = core.snapshot()["subscription"]["arms"]
    assert sum(a["pull_count"] for a in after) == 1

    decisions = [e for e in pg.get_events("case-1") if e.event_type == "Decision"]
    assert "bandit_arm" in decisions[0].payload["action_params"]


def test_process_case_recovers_correctly_after_a_simulated_restart(pg):
    """
    Architecture doc 9.3: 'on restart, any case whose last logged event
    isn't terminal is resumed from exactly that point.' Simulated here by
    building a BRAND NEW Orchestrator instance (as a real process restart
    would) pointed at the same PostgresEventStore, and confirming it picks
    the case back up correctly using only what's durably stored.
    """
    orchestrator_before_restart = Orchestrator(pg)
    orchestrator_before_restart.register_module(SubscriptionModule())

    case = {"decline_code": "51", "amount": 250.0}
    orchestrator_before_restart.process_case("case-1", "subscription", case, max_iterations=1)
    mid_flight_state = pg.derive_state("case-1")
    assert mid_flight_state["terminal"] is False

    # "Restart": a fresh Orchestrator, fresh SubscriptionModule, same store.
    orchestrator_after_restart = Orchestrator(pg)
    orchestrator_after_restart.register_module(SubscriptionModule())

    case["simulated_retry_result"] = "recovered"
    final_state = orchestrator_after_restart.process_case("case-1", "subscription", case, max_iterations=1)
    assert final_state["terminal_status"] == "RECOVERED"

