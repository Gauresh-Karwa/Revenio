
"""
Queued counterpart to backend.core.bandit_observer.BanditUpdateObserver â€”
same event-watching logic (credits the most-recent bandit_arm chosen for a
case_id at that case's terminal Outcome), but ENQUEUES the update onto
Celery's "bandit_updates" queue instead of calling learning_core.update()
synchronously in-process. This is the concrete realization of architecture
doc 9.5's "outcomes are queued... a single dedicated consumer applies them
to the policy sequentially" for a real deployed worker pool â€” the
in-process BanditUpdateObserver from Step 6 already demonstrates the same
credit-assignment logic against the in-memory EventStore in tests; this
class is that same logic pointed at a real queue instead of a direct call.

Deliberately does NOT hold a LearningCore reference â€” the whole point is
decoupling the calling process (wherever orchestrator.process_case() just
ran) from whichever worker process eventually applies the update. It only
knows how to enqueue, never how to apply.

CREDIT ASSIGNMENT: identical simplification to BanditUpdateObserver's
own â€” credits the MOST RECENT arm chosen for a case_id, not full
per-attempt credit assignment across a multi-attempt case. Same
open-item status as that file's docstring already states; not repeated
in full here.
"""

from __future__ import annotations

from backend.core.events import Event
from backend.queue.tasks import apply_bandit_update_task


class QueuedBanditUpdateObserver:
    def __init__(self) -> None:
        self._pending_arm_by_case: dict[str, tuple[str, int]] = {}

    def on_event(self, event: Event) -> None:
        if event.event_type == "Decision":
            arm = event.payload.get("action_params", {}).get("bandit_arm")
            if arm is not None:
                self._pending_arm_by_case[event.case_id] = (event.domain_type, int(arm))
            return

        if event.event_type == "Outcome":
            status = event.payload.get("status")
            if status not in ("RECOVERED", "LOST"):
                return  # PENDING outcomes are not terminal â€” nothing to credit yet

            pending = self._pending_arm_by_case.pop(event.case_id, None)
            if pending is None:
                return  # this case never had a bandit-chosen arm

            domain_type, arm = pending
            reward = 1.0 if status == "RECOVERED" else 0.0
            apply_bandit_update_task.delay(domain_type, arm, reward)

