
"""
Single place that wires a real, production-backed Orchestrator: PostgreSQL
event store + all real domain modules + an optional durably-backed
LearningCore. Architecture doc section 2's "one implementation, not
multiple copies" discipline applied to wiring itself â€” Celery tasks
(backend/queue/tasks.py) and any future FastAPI service both call
build_orchestrator() rather than each hand-assembling their own copy of
"which modules get registered, in what order, with what config."
"""

from __future__ import annotations

import os

from backend.config import load_environment
from backend.core.learning_core import LearningCore
from backend.core.orchestrator import Orchestrator
from backend.modules.b2b_receivables.module import B2BReceivablesModule
from backend.modules.checkout_abandonment.module import CheckoutAbandonmentModule
from backend.modules.mandate_retry.module import MandateRetryModule
from backend.modules.subscription.module import SubscriptionModule
from backend.storage.postgres_event_store import PostgresEventStore

load_environment()

DATABASE_URL = os.environ.get(
    "REVENIO_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/revenio"
)


def build_orchestrator(learning_core: LearningCore | None = None) -> tuple[Orchestrator, PostgresEventStore]:
    """
    Returns (orchestrator, event_store) callers that need to query
    events/state directly (a future FastAPI read endpoint, a Celery task
    finishing up) get the store handle without a second construction path.
    Every real domain module is registered; the dummy module is
    deliberately excluded here (it's a Step-1 test fixture, never meant to
    run in production).
    """
    store = PostgresEventStore(DATABASE_URL)
    store.init_schema()

    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule(learning_core=learning_core))
    orchestrator.register_module(CheckoutAbandonmentModule(learning_core=learning_core))
    orchestrator.register_module(B2BReceivablesModule(learning_core=learning_core))
    orchestrator.register_module(MandateRetryModule(learning_core=learning_core))

    return orchestrator, store
