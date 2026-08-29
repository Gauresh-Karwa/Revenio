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