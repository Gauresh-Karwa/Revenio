from __future__ import annotations

from dataclasses import asdict
from enum import Enum
from typing import Any

from backend.core.contract import DomainModule
from backend.core.events import EventStore


def _enum_safe(value: Any) -> Any:

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _enum_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enum_safe(v) for v in value]
    return value


def _to_payload(obj: Any) -> dict[str, Any]:
    return _enum_safe(asdict(obj))


class UnknownDomainError(Exception):
    pass


class Orchestrator:
    def __init__(self, event_store: EventStore) -> None:
        self._event_store = event_store
        self._modules: dict[str, DomainModule] = {}

    def register_module(self, module: DomainModule) -> None:
        self._modules[module.domain_type] = module

    def process_case(
        self, case_id: str, domain_type: str, case: dict[str, Any], max_iterations: int = 20
    ) -> dict[str, Any]:

        if domain_type not in self._modules:
            raise UnknownDomainError(f"No module registered for domain '{domain_type}'")

        module = self._modules[domain_type]
        customer_id = case.get("customer_id")

        for _ in range(max_iterations):
            history = self._event_store.get_events(case_id)
            history_payloads = [{"_event_type": e.event_type, **e.payload} for e in history]

            stop = module.check_stop(case, history_payloads)
            self._event_store.append(
                case_id, domain_type, "check_stop", "StopDecision", _to_payload(stop),
                customer_id=customer_id,
            )
            if stop.should_stop:
                return self._event_store.derive_state(case_id)

            # customer_history is a DIFFERENT query than history above: this
            # customer's OTHER, past cases (cross-case), not this case's own
            # retry chain (within-case). The orchestrator only fetches raw
            # events here — it does NOT compute any domain-specific
            # interpretation (e.g. an EWMA) of them, keeping "zero domain-
            # specific logic in the orchestrator" (architecture doc section 2)
            # intact. Whatever a module does with this raw history is its
            # own business.
            customer_history_payloads = None
            if customer_id is not None:
                customer_events = self._event_store.get_customer_case_history(
                    customer_id, exclude_case_id=case_id
                )
                customer_history_payloads = [
                    {"_event_type": e.event_type, "_case_id": e.case_id, **e.payload}
                    for e in customer_events
                ]

            diagnosis = module.diagnose(case, customer_history_payloads)
            self._event_store.append(
                case_id, domain_type, "diagnose", "Diagnosis", _to_payload(diagnosis),
                customer_id=customer_id,
            )

            decision = module.decide(case, diagnosis, history_payloads)
            self._event_store.append(
                case_id, domain_type, "decide", "Decision", _to_payload(decision),
                customer_id=customer_id,
            )

            if decision.requires_human_review:
                self._event_store.append(
                    case_id,
                    domain_type,
                    "decide",
                    "PendingHumanReview",
                    {"reason": decision.reasoning},
                    customer_id=customer_id,
                )
                return self._event_store.derive_state(case_id)

            exec_result = module.execute(case, decision)
            self._event_store.append(
                case_id, domain_type, "execute", "ExecutionResult", _to_payload(exec_result),
                customer_id=customer_id,
            )

            outcome = module.track_outcome(case)
            self._event_store.append(
                case_id, domain_type, "track_outcome", "Outcome", _to_payload(outcome),
                customer_id=customer_id,
            )

            if outcome.status.value in ("RECOVERED", "LOST"):
                return self._event_store.derive_state(case_id)

        return self._event_store.derive_state(case_id)

    def submit_human_review(
        self, case_id: str, confirmed: bool, case: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Records a human's decision on a case sitting in PendingHumanReview,
        and — if the module implements it — notifies the module so it can
        act on the confirmation (e.g. SubscriptionModule growing its
        hardship anchor bank; see that module's on_human_review_confirmed).

        PRIVACY-PRESERVING DURABLE/EPHEMERAL SPLIT, stated explicitly:
        the DURABLE event appended here carries only `confirmed` (bool) —
        no raw case data, consistent with every other privacy rule in this
        project (raw email text is never persisted). `case` itself IS
        passed to the module's optional callback, but only ephemerally,
        in-memory, for this one call — the review-queue UI that already
        held this case's data (to show a human what they're reviewing,
        architecture doc 7.3) is the source of that data, not something
        reconstructed from the privacy-scrubbed audit log. The module's
        gating logic (see on_human_review_confirmed) checks the DURABLE
        hardship_confidence_tier field before ever touching case["email_text"],
        so a confirmation on a case with no prior hardship flag can never
        accidentally leak into anchor growth.

        Orchestrator itself stays domain-agnostic here too: it does not
        know what "growing an anchor bank" means, it just calls an
        optional, duck-typed method if the module defines one.
        """
        events = self._event_store.get_events(case_id)
        if not events:
            raise ValueError(f"No case found for case_id '{case_id}'")

        domain_type = events[0].domain_type
        customer_id = case.get("customer_id")

        self._event_store.append(
            case_id, domain_type, "human_review", "HumanReviewDecision",
            {"confirmed": confirmed}, customer_id=customer_id,
        )

        module = self._modules.get(domain_type)
        callback = getattr(module, "on_human_review_confirmed", None)
        if callback is not None:
            diagnosis_events = [e for e in events if e.event_type == "Diagnosis"]
            last_diagnosis_payload = diagnosis_events[-1].payload if diagnosis_events else {}
            callback(case, confirmed, last_diagnosis_payload)

        return self._event_store.derive_state(case_id)