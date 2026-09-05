
"""
Runs against a REAL Postgres instance (not mocked) â€” set REVENIO_TEST_DB_URL
or these default to postgresql://postgres:postgres@localhost:5432/revenio_test.
Each test truncates the tables it touches via the `pg` fixture's teardown so
tests stay independent despite append-only tables (TRUNCATE, not DELETE-per-row,
since this is dev/test cleanup, not an application-layer operation â€” the
application layer itself never issues UPDATE/DELETE against events, per
schema.sql's append-only comment).
"""

import os
import socket
from urllib.parse import urlparse

import pytest

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


def test_append_returns_event_with_real_id_and_timestamp(pg):
    event = pg.append("case-1", "dummy", "diagnose", "Diagnosis", {"root_cause": "x"})
    assert event.event_id is not None
    assert event.case_id == "case-1"
    assert event.created_at  # a real ISO timestamp, not empty


def test_get_events_returns_in_insertion_order(pg):
    pg.append("case-1", "dummy", "diagnose", "Diagnosis", {"n": 1})
    pg.append("case-1", "dummy", "decide", "Decision", {"n": 2})
    pg.append("case-1", "dummy", "execute", "ExecutionResult", {"n": 3})

    events = pg.get_events("case-1")
    assert [e.payload["n"] for e in events] == [1, 2, 3]


def test_events_are_scoped_per_case(pg):
    pg.append("case-1", "dummy", "diagnose", "Diagnosis", {})
    pg.append("case-2", "dummy", "diagnose", "Diagnosis", {})

    assert len(pg.get_events("case-1")) == 1
    assert len(pg.get_events("case-2")) == 1


def test_payload_round_trips_as_a_real_dict_not_a_json_string(pg):
    """
    A real risk with JSONB columns: getting back a str you have to
    json.loads() yourself instead of a dict. psycopg's jsonb adapter
    should handle this transparently â€” proven here, not assumed.
    """
    payload = {"nested": {"a": 1, "b": [1, 2, 3]}, "flag": True}
    pg.append("case-1", "dummy", "diagnose", "Diagnosis", payload)
    event = pg.get_events("case-1")[0]
    assert isinstance(event.payload, dict)
    assert event.payload == payload


def test_derive_state_for_unknown_case_returns_not_exists(pg):
    state = pg.derive_state("no-such-case")
    assert state["exists"] is False


def test_derive_state_detects_stop_as_terminal(pg):
    pg.append(
        "case-1", "dummy", "check_stop", "StopDecision",
        {"should_stop": True, "stop_reason": "RESOLVED"},
    )
    state = pg.derive_state("case-1")
    assert state["terminal"] is True
    assert state["terminal_status"] == "STOPPED:RESOLVED"


def test_derive_state_detects_recovered_outcome_as_terminal(pg):
    pg.append(
        "case-1", "dummy", "track_outcome", "Outcome",
        {"status": "RECOVERED", "amount_recovered": 500.0},
    )
    state = pg.derive_state("case-1")
    assert state["terminal"] is True
    assert state["terminal_status"] == "RECOVERED"


def test_derive_state_non_terminal_event_does_not_flip_status(pg):
    pg.append("case-1", "dummy", "check_stop", "StopDecision", {"should_stop": False})
    pg.append("case-1", "dummy", "diagnose", "Diagnosis", {"root_cause": "x"})
    state = pg.derive_state("case-1")
    assert state["terminal"] is False
    assert state["terminal_status"] is None


def test_derive_state_last_terminal_event_wins_if_multiple(pg):
    """
    Mirrors the in-memory EventStore's replay semantics: if a case somehow
    produces more than one terminal-triggering event, the LAST one in
    order determines the final status â€” not the first.
    """
    pg.append(
        "case-1", "dummy", "check_stop", "StopDecision",
        {"should_stop": True, "stop_reason": "COST_THRESHOLD"},
    )
    pg.append(
        "case-1", "dummy", "track_outcome", "Outcome",
        {"status": "RECOVERED", "amount_recovered": 100.0},
    )
    state = pg.derive_state("case-1")
    assert state["terminal_status"] == "RECOVERED"


def test_derive_state_stage_count_and_last_stage_are_correct(pg):
    pg.append("case-1", "dummy", "check_stop", "StopDecision", {"should_stop": False})
    pg.append("case-1", "dummy", "diagnose", "Diagnosis", {})
    pg.append("case-1", "dummy", "decide", "Decision", {})
    state = pg.derive_state("case-1")
    assert state["stage_count"] == 3
    assert state["last_stage"] == "decide"
    assert state["last_event_type"] == "Decision"


def test_derive_state_history_matches_full_payload_list(pg):
    pg.append("case-1", "dummy", "diagnose", "Diagnosis", {"root_cause": "x"})
    pg.append("case-1", "dummy", "decide", "Decision", {"action_type": "RETRY"})
    state = pg.derive_state("case-1")
    assert state["history"] == [{"root_cause": "x"}, {"action_type": "RETRY"}]


def test_get_customer_case_history_returns_only_this_customers_events(pg):
    pg.append("case-1", "subscription", "diagnose", "Diagnosis", {}, customer_id="cust-1")
    pg.append("case-2", "subscription", "diagnose", "Diagnosis", {}, customer_id="cust-2")
    history = pg.get_customer_case_history("cust-1")
    assert len(history) == 1
    assert history[0].case_id == "case-1"


def test_get_customer_case_history_excludes_current_case(pg):
    pg.append("case-1", "subscription", "diagnose", "Diagnosis", {}, customer_id="cust-1")
    pg.append("case-2", "subscription", "diagnose", "Diagnosis", {}, customer_id="cust-1")
    history = pg.get_customer_case_history("cust-1", exclude_case_id="case-2")
    assert len(history) == 1
    assert history[0].case_id == "case-1"


def test_get_customer_case_history_spans_multiple_past_cases_in_order(pg):
    pg.append("case-1", "subscription", "track_outcome", "Outcome", {"status": "LOST"}, customer_id="cust-1")
    pg.append("case-2", "subscription", "track_outcome", "Outcome", {"status": "RECOVERED"}, customer_id="cust-1")
    history = pg.get_customer_case_history("cust-1", exclude_case_id="case-3")
    assert [e.payload["status"] for e in history] == ["LOST", "RECOVERED"]


def test_events_without_customer_id_are_never_returned(pg):
    pg.append("case-1", "subscription", "diagnose", "Diagnosis", {})
    assert pg.get_customer_case_history("cust-1") == []


def test_subscribe_notifies_observer_only_after_commit(pg):
    seen = []

    class Spy:
        def on_event(self, event):
            # If this fires, the event must already be readable via a
            # fresh query â€” proves notification happens post-commit, not
            # from an in-memory buffer that might not have landed yet.
            seen.append(event)
            assert len(pg.get_events(event.case_id)) >= 1

    pg.subscribe(Spy())
    pg.append("case-1", "dummy", "diagnose", "Diagnosis", {})
    assert len(seen) == 1


def test_rebuild_case_state_from_events_matches_incremental_result(pg):
    """
    Proves the recovery path (full replay) computes the SAME final state
    as the normal incremental append() path â€” the honest answer to "what
    if case_state ever drifted," verified rather than assumed.
    """
    pg.append("case-1", "subscription", "check_stop", "StopDecision", {"should_stop": False})
    pg.append(
        "case-1", "subscription", "track_outcome", "Outcome",
        {"status": "LOST", "amount_recovered": 0.0},
    )
    incremental_state = pg.derive_state("case-1")

    # Corrupt case_state directly to prove rebuild actually recomputes,
    # rather than trivially matching because nothing changed.
    with pg._pool.connection() as conn:
        conn.execute("UPDATE case_state SET status = 'ACTIVE', terminal = FALSE WHERE case_id = %s", ("case-1",))
        conn.commit()
    assert pg.derive_state("case-1")["terminal"] is False  # confirm corruption took effect

    pg.rebuild_case_state_from_events("case-1")
    rebuilt_state = pg.derive_state("case-1")

    assert rebuilt_state["terminal"] == incremental_state["terminal"]
    assert rebuilt_state["terminal_status"] == incremental_state["terminal_status"]
    assert rebuilt_state["stage_count"] == incremental_state["stage_count"]

