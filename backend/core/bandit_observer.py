"""
BanditUpdateObserver: the concrete single-writer consumer architecture doc
9.5 describes ("outcomes are queued... a single dedicated consumer applies
them to the policy sequentially — one writer, no races"), implemented as a
real EventObserver (backend/core/events.py) rather than left as prose.

WHAT IT DOES: watches every event flowing through an EventStore. When a
Decision event carries a `bandit_arm` in its action_params (meaning the
module consulted a LearningCore to pick that arm — see
SubscriptionModule/CheckoutAbandonmentModule's decide()), it remembers
which arm was chosen for that case_id. When that SAME case_id later
produces a terminal Outcome event (RECOVERED or LOST), it looks up the
remembered arm and calls learning_core.update(domain_type, arm, reward).

WHY THIS DECOUPLES CLEANLY: neither the orchestrator nor the modules need
to know a learning core exists in order to produce Decision/Outcome events
— they always have. This observer is what turns those already-existing
events into bandit updates, entirely from the outside. Removing the
observer (or never subscribing it) means the bandit simply never learns —
nothing else breaks, matching the "domain independence, graceful
degradation" principle used everywhere else in this project.

CREDIT ASSIGNMENT, STATED HONESTLY: a case can go through multiple
decide() calls (multiple retry/nudge attempts) before reaching a terminal
outcome, each potentially choosing a different arm. This observer credits/
blames the MOST RECENT arm chosen for that case_id — the arm active on the
attempt that immediately preceded the terminal outcome. This is a
deliberate simplification, not full per-attempt credit assignment (which
would require attributing partial credit across a sequence of pulls for
one eventual outcome — a real, harder problem, tracked as an open item,
not silently assumed away).
"""

from __future__ import annotations

from backend.core.events import Event
from backend.core.learning_core import LearningCore


class BanditUpdateObserver:
    def __init__(self, learning_core: LearningCore) -> None:
        self._learning_core = learning_core
        # case_id -> (domain_type, arm) for the most recently chosen arm
        # on that case, cleared once consumed by a terminal Outcome.
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
                return  # PENDING outcomes are not terminal — nothing to credit yet

            pending = self._pending_arm_by_case.pop(event.case_id, None)
            if pending is None:
                return  # this case never had a bandit-chosen arm (e.g. no learning_core wired)

            domain_type, arm = pending
            if not self._learning_core.has_policy(domain_type):
                return  # domain not registered — degrade gracefully, per project convention

            reward = 1.0 if status == "RECOVERED" else 0.0
            self._learning_core.update(domain_type, arm, reward)
