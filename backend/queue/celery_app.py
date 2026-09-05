
"""
Celery application â€” architecture doc 9.4: "A pool of worker processes
consumes the queue at a sustainable rate (Redis + Celery/RQ)."

Two queues, kept SEPARATE on purpose rather than one shared queue:
  - "case_processing": webhook-triggered case ingestion (9.4). Can be
    scaled to many concurrent workers freely â€” process_case_task's
    correctness does not depend on ordering across different case_ids,
    and even same-case_id reprocessing is safe because Orchestrator
    replays from the durable event log every time (9.1/9.3).
  - "bandit_updates": the single-writer consumer (9.5). Correctness here
    depends on PostgresBanditStore's row-level lock, NOT on running only
    one worker â€” see that file's docstring for why relying on worker
    count alone would be fragile. Kept on its own queue anyway so it can
    be reasoned about and monitored independently of case-processing
    throughput.
"""

from __future__ import annotations

import os
import ssl

from celery import Celery

from backend.config import load_environment

load_environment()
REDIS_URL = os.environ.get("REVENIO_REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "revenio",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.queue.tasks"],
)

settings = {
    "task_serializer": "json",
    "result_serializer": "json",
    "accept_content": ["json"],
    "task_routes": {
        "backend.queue.tasks.process_case_task": {"queue": "case_processing"},
        "backend.queue.tasks.apply_bandit_update_task": {"queue": "bandit_updates"},
    },
}

# redis-py accepts rediss:// without extra configuration, but Celery/Kombu
# requires the verification policy to be explicit for BOTH the broker and its
# Redis result backend.  CERT_REQUIRED prevents a managed Redis endpoint from
# being silently downgraded to an unverified TLS connection.
if REDIS_URL.startswith("rediss://"):
    tls_options = {"ssl_cert_reqs": ssl.CERT_REQUIRED}
    settings["broker_use_ssl"] = tls_options
    settings["redis_backend_use_ssl"] = tls_options

celery_app.conf.update(
    **settings,
)
