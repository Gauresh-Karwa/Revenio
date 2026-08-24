"""
Event schema and event store.

Architecture doc section 9.1: the audit log IS the source of truth. A case's
current state is derived by replaying its events, never stored redundantly.

IMPORTANT — honest scope note, not a silent shortcut: this file implements an
in-memory event store so the orchestrator loop (this checkpoint) can be built
and tested in complete isolation, with zero infrastructure dependencies. It is
NOT the production store. Section 9.6 requires PostgreSQL with the given
indexes once concurrent case processing is real — that swap is a follow-up
checkpoint, tracked here as TODO(step-1-followup), not forgotten.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Event:
    event_id: int
    case_id: str
    domain_type: str
    stage: str  # e.g. "check_stop", "diagnose", "decide", "execute", "track_outcome"
    event_type: str  # e.g. "StopDecision", "Diagnosis", "Decision", ...
    payload: dict[str, Any]
    created_at: str  # ISO 8601


class EventStore:
    """
    Append-only. No update, no delete — matches architecture doc 9.6's note
    that the audit log is append-only, which sidesteps update contention.
    """

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._id_counter = itertools.count(1)

    def append(
        self,
        case_id: str,
        domain_type: str,
        stage: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> Event:
        event = Event(
            event_id=next(self._id_counter),
            case_id=case_id,
            domain_type=domain_type,
            stage=stage,
            event_type=event_type,
            payload=payload,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._events.append(event)
        return event

    def get_events(self, case_id: str) -> list[Event]:
        return [e for e in self._events if e.case_id == case_id]

    def derive_state(self, case_id: str) -> dict[str, Any]:
        """
        Reconstructs a case's current state by replaying its events.
        This is what "the audit log is the source of truth" means concretely:
        nothing about a case's status is stored anywhere except as a
        replayable consequence of these events.
        """
        events = self.get_events(case_id)
        if not events:
            return {"case_id": case_id, "exists": False}

        state: dict[str, Any] = {
            "case_id": case_id,
            "exists": True,
            "domain_type": events[0].domain_type,
            "stage_count": len(events),
            "last_stage": events[-1].stage,
            "last_event_type": events[-1].event_type,
            "history": [e.payload for e in events],
            "terminal": False,
            "terminal_status": None,
        }

        for e in events:
            if e.event_type == "StopDecision" and e.payload.get("should_stop"):
                state["terminal"] = True
                state["terminal_status"] = f"STOPPED:{e.payload.get('stop_reason')}"
            if e.event_type == "Outcome" and e.payload.get("status") in (
                "RECOVERED",
                "LOST",
            ):
                state["terminal"] = True
                state["terminal_status"] = e.payload.get("status")

        return state