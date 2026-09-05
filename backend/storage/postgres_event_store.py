
"""
PostgresEventStore â€” the production swap for backend.core.events.EventStore,
promised since Step 1 ("this is NOT the production store... tracked as
TODO(step-1-followup), not forgotten") and specified in architecture doc 9.1
/ 9.6.

DROP-IN REPLACEMENT, BY DESIGN: implements the exact same five methods as
EventStore (append, get_events, get_customer_case_history, derive_state,
subscribe) with the same signatures and the same return shapes (Event
objects, and derive_state's dict). Orchestrator, BanditUpdateObserver, and
every domain module were built against that interface without ever
importing EventStore's concrete class â€” they only ever call methods on
whatever `event_store` object they were constructed with. That's what
makes this swap possible with ZERO changes to orchestrator.py,
bandit_observer.py, or any of the four domain modules. Confirmed, not
assumed: tests/integration/test_orchestrator_against_postgres.py runs the
EXACT SAME orchestrator test scenarios from tests/core/test_orchestrator.py
and tests/integration/test_bandit_observer_wiring.py against a real
Postgres instance instead of the in-memory store.

ARCHITECTURE DOC 9.1 â€” ATOMICITY: every state transition (an events insert
PLUS the derived case_state update) is written as one atomic database
transaction. This is the concrete implementation of "removes the classic
'executed but not logged' failure mode by construction." Either both
writes land or neither does; there is no window where events shows a
StopDecision but case_state still says ACTIVE.

WHY case_state EXISTS AND IS NOT A SECOND SOURCE OF TRUTH: architecture
doc 9.1 warns against dual writes where two representations of state can
silently drift. case_state is not that â€” it is a materialized, DERIVED
projection of events, written in the same transaction as the event that
produced it, and is fully reconstructable by replaying events at any time
(see rebuild_case_state_from_events below) if it were ever found to have
drifted. This is the standard event-sourcing "read model" pattern, not an
exception to 9.1's rule â€” it exists because the in-memory EventStore's
derive_state() (a full replay of every event on every call) does not hold
up against 9.6's "every list-facing query is paginated, never a full
scan" once event volume is real.

OBSERVER NOTIFICATION TIMING: exactly like the in-memory EventStore,
observers are notified synchronously, in append order, AFTER the
transaction commits â€” never before. An observer must never see an event
that didn't actually get durably recorded. This store does NOT yet
implement architecture doc 9.5's queued/async single-writer dispatch for
bandit updates â€” that is the Redis/Celery follow-up (see
backend/queue/queued_bandit_observer.py). BanditUpdateObserver itself is
unaffected either way, since it only depends on the EventObserver
Protocol, not on how notification is scheduled.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from backend.core.events import Event, EventObserver


def _terminal_effect(event_type: str, payload: dict[str, Any]) -> tuple[bool, str | None]:
    """
    THE single implementation of "does this event make a case terminal, and
    what's the resulting status" â€” used both by append() (incremental,
    per-event) and rebuild_case_state_from_events() (full replay, for
    recovery/audit). One implementation, not two that could drift apart,
    same discipline as subscription_generator.py's update_causal_pressure.

    Mirrors backend.core.events.EventStore.derive_state's replay loop
    EXACTLY: a StopDecision with should_stop=True makes it terminal with
    status STOPPED:<stop_reason>; an Outcome with status in
    (RECOVERED, LOST) makes it terminal with that status. Anything else
    has no terminal effect (returns (False, None) â€” caller decides whether
    to overwrite existing state based on this).
    """
    if event_type == "StopDecision" and payload.get("should_stop"):
        return True, f"STOPPED:{payload.get('stop_reason')}"
    if event_type == "Outcome" and payload.get("status") in ("RECOVERED", "LOST"):
        return True, payload.get("status")
    return False, None


class PostgresEventStore:
    def __init__(self, conninfo: str, min_size: int = 1, max_size: int = 10) -> None:
        self._pool = ConnectionPool(conninfo, min_size=min_size, max_size=max_size, open=True)
        self._observers: list[EventObserver] = []

    def close(self) -> None:
        self._pool.close()

    # --- schema setup -------------------------------------------------------

    def init_schema(self, schema_path: str | None = None) -> None:
        """
        Applies backend/storage/schema.sql. Idempotent (CREATE TABLE/INDEX
        IF NOT EXISTS throughout) â€” safe to call on every app startup, the
        same way a migration tool's "up to head" is safe to re-run.
        """
        import pathlib

        path = pathlib.Path(schema_path) if schema_path else (
            pathlib.Path(__file__).parent / "schema.sql"
        )
        sql = path.read_text(encoding="utf-8")
        with self._pool.connection() as conn:
            conn.execute(sql)
            conn.commit()

    # --- EventStore-compatible interface ------------------------------------

    def subscribe(self, observer: EventObserver) -> None:
        self._observers.append(observer)

    def append(
        self,
        case_id: str,
        domain_type: str,
        stage: str,
        event_type: str,
        payload: dict[str, Any],
        customer_id: str | None = None,
    ) -> Event:
        is_terminal_event, new_status = _terminal_effect(event_type, payload)
        # Non-terminal events don't change status â€” pass the SAME sentinel
        # 'ACTIVE' as new_status; the SQL only applies new_status when
        # is_terminal_event is true (see CASE WHEN below), so this value is
        # inert on non-terminal appends. Kept explicit rather than NULL to
        # avoid a nullable-column edge case in the UPSERT.
        effective_new_status = new_status if is_terminal_event else "ACTIVE"

        with self._pool.connection() as conn:
            with conn.transaction():  # architecture doc 9.1: one atomic transaction
                row = conn.execute(
                    """
                    INSERT INTO events (case_id, domain_type, stage, event_type, payload, customer_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING event_id, created_at
                    """,
                    (case_id, domain_type, stage, event_type, Jsonb(payload), customer_id),
                ).fetchone()
                event_id, created_at = row

                conn.execute(
                    """
                    INSERT INTO case_state
                        (case_id, domain_type, status, terminal, stage_count, last_stage, last_event_type, updated_at)
                    VALUES
                        (%(case_id)s, %(domain_type)s, %(status)s, %(terminal)s, 1, %(stage)s, %(event_type)s, now())
                    ON CONFLICT (case_id) DO UPDATE SET
                        stage_count = case_state.stage_count + 1,
                        last_stage = %(stage)s,
                        last_event_type = %(event_type)s,
                        updated_at = now(),
                        status = CASE WHEN %(is_terminal)s THEN %(status)s ELSE case_state.status END,
                        terminal = CASE WHEN %(is_terminal)s THEN %(terminal)s ELSE case_state.terminal END
                    """,
                    {
                        "case_id": case_id,
                        "domain_type": domain_type,
                        "status": effective_new_status,
                        "terminal": is_terminal_event,
                        "stage": stage,
                        "event_type": event_type,
                        "is_terminal": is_terminal_event,
                    },
                )

        event = Event(
            event_id=event_id,
            case_id=case_id,
            domain_type=domain_type,
            stage=stage,
            event_type=event_type,
            payload=payload,
            created_at=created_at.isoformat() if isinstance(created_at, datetime) else str(created_at),
            customer_id=customer_id,
        )

        # Notified AFTER commit, in order â€” an observer must never see an
        # event that didn't actually get durably recorded.
        for observer in self._observers:
            observer.on_event(event)

        return event

    def get_events(self, case_id: str) -> list[Event]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT event_id, case_id, domain_type, stage, event_type, payload, created_at, customer_id
                FROM events
                WHERE case_id = %s
                ORDER BY event_id
                """,
                (case_id,),
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    def get_customer_case_history(
        self, customer_id: str, exclude_case_id: str | None = None
    ) -> list[Event]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT event_id, case_id, domain_type, stage, event_type, payload, created_at, customer_id
                FROM events
                WHERE customer_id = %s AND case_id IS DISTINCT FROM %s
                ORDER BY event_id
                """,
                (customer_id, exclude_case_id),
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    def derive_state(self, case_id: str) -> dict[str, Any]:
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT domain_type, status, terminal, stage_count, last_stage, last_event_type
                FROM case_state
                WHERE case_id = %s
                """,
                (case_id,),
            ).fetchone()

        if row is None:
            return {"case_id": case_id, "exists": False}

        domain_type, status, terminal, stage_count, last_stage, last_event_type = row

        # Bounded by this one case's own event count, not a full table
        # scan â€” the same "history" list the in-memory store returns, kept
        # for exact interface parity (audit view needs the full per-stage
        # payload trace, not just the summary row).
        history = [e.payload for e in self.get_events(case_id)]

        return {
            "case_id": case_id,
            "exists": True,
            "domain_type": domain_type,
            "stage_count": stage_count,
            "last_stage": last_stage,
            "last_event_type": last_event_type,
            "history": history,
            "terminal": terminal,
            "terminal_status": status if status != "ACTIVE" else None,
        }

    # --- recovery / audit -----------------------------------------------------

    def rebuild_case_state_from_events(self, case_id: str) -> None:
        """
        Full replay recovery path, per architecture doc 9.3: "on restart,
        any case whose last logged event isn't terminal is resumed from
        exactly that point" â€” and more generally, if case_state were ever
        suspected to have drifted from events (it shouldn't, given the
        single-transaction write in append(), but this is the honest
        answer to 'what if it did'), this rebuilds it from the only real
        source of truth. Uses the SAME _terminal_effect function append()
        uses, so recovery and live writes can never compute different
        answers to "is this case terminal" for the same event.
        """
        events = self.get_events(case_id)
        if not events:
            return

        status = "ACTIVE"
        terminal = False
        for e in events:
            is_terminal_event, new_status = _terminal_effect(e.event_type, e.payload)
            if is_terminal_event:
                status, terminal = new_status, True

        last = events[-1]
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO case_state
                    (case_id, domain_type, status, terminal, stage_count, last_stage, last_event_type, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (case_id) DO UPDATE SET
                    domain_type = EXCLUDED.domain_type,
                    status = EXCLUDED.status,
                    terminal = EXCLUDED.terminal,
                    stage_count = EXCLUDED.stage_count,
                    last_stage = EXCLUDED.last_stage,
                    last_event_type = EXCLUDED.last_event_type,
                    updated_at = now()
                """,
                (case_id, last.domain_type, status, terminal, len(events), last.stage, last.event_type),
            )
            conn.commit()


def _row_to_event(row: tuple) -> Event:
    event_id, case_id, domain_type, stage, event_type, payload, created_at, customer_id = row
    return Event(
        event_id=event_id,
        case_id=case_id,
        domain_type=domain_type,
        stage=stage,
        event_type=event_type,
        payload=payload,  # psycopg adapts jsonb -> dict automatically
        created_at=created_at.isoformat() if isinstance(created_at, datetime) else str(created_at),
        customer_id=customer_id,
    )

