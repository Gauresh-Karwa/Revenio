from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.core.contract import (
    ActionType,
    Decision,
    Diagnosis,
    ExecutionResult,
    Outcome,
    OutcomeStatus,
    PromiseOutcome,
    StopDecision,
    StopReason,
)
from backend.ml.features import FEATURE_NAMES, build_feature_vector_from_case

SOFT_DECLINE_CODES: dict[str, str] = {
    "51": "insufficient_funds",
    "05": "do_not_honor",
    "91": "issuer_unavailable",
    "96": "system_malfunction",
    "65": "activity_limit_exceeded",
    "61": "exceeds_withdrawal_limit",
}

HARD_DECLINE_CODES: dict[str, str] = {
    "04": "pickup_card",
    "07": "pickup_card_special",
    "12": "invalid_transaction",
    "14": "invalid_card_number",
    "15": "invalid_issuer",
    "41": "lost_card",
    "43": "stolen_card",
    "46": "closed_account",
    "57": "transaction_not_permitted",
}

STOP_INSTRUCTION_CODES: dict[str, str] = {
    "R0": "customer_stopped_specific_payment",
    "R1": "customer_stopped_all_recurring",
    "R3": "authorization_revoked",
}

MAX_RETRY_ATTEMPTS = 15

RETRY_BACKOFF_HOURS = [1, 6, 24, 72]

DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "ml" / "models" / "subscription_winner.joblib"


def _load_model_bundle(model_path: Path) -> dict | None:
    """
    Loads the offline-trained bundle (model_type + pipeline + feature
    schema) if it exists. Returns None if it doesn't — the normal state
    before train_subscription_model.py has ever been run — and must never
    raise, since SubscriptionModule() is instantiated no-arg throughout the
    orchestrator and test suite.

    Also refuses a bundle whose feature schema doesn't match the module's
    live FEATURE_NAMES: this is the concrete guard against the "model
    trained on one feature set, module builds a different one, predictions
    are silently wrong" failure mode.
    """
    if not model_path.exists():
        return None
    try:
        import joblib

        bundle = joblib.load(model_path)
    except Exception:
        return None

    if bundle.get("feature_names") != FEATURE_NAMES:
        warnings.warn(
            "Loaded model bundle's feature_names does not match the current "
            "build_feature_vector schema — refusing to use it. Re-run "
            "train_subscription_model.py to retrain against the current "
            "feature set. Falling back to rule-based confidence only.",
            stacklevel=2,
        )
        return None

    return bundle


class SubscriptionModule:
    domain_type = "subscription"

    def __init__(self, model_path: Path | str | None = None) -> None:
        path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self._model_bundle = _load_model_bundle(path)

    def check_stop(
        self, case: dict[str, Any], history: list[dict[str, Any]]
    ) -> StopDecision:
        code = case.get("decline_code")

        if code in HARD_DECLINE_CODES:
            return StopDecision(should_stop=True, stop_reason=StopReason.COMPLIANCE_LIMIT)

        if code in STOP_INSTRUCTION_CODES:
            return StopDecision(should_stop=True, stop_reason=StopReason.OPT_OUT)

        retry_count = sum(1 for h in history if h.get("_event_type") == "ExecutionResult")
        if retry_count >= MAX_RETRY_ATTEMPTS:
            return StopDecision(should_stop=True, stop_reason=StopReason.COMPLIANCE_LIMIT)

        return StopDecision(should_stop=False)

    def _predict_recovery_probability(self, case: dict[str, Any], code: str) -> float | None:
        """
        Only ever called for known soft codes. Model-agnostic: whichever
        model_type won at training time (GBM, MLP, or a future candidate),
        this call is identical — the winning Pipeline owns its own
        preprocessing, so this method never needs to know what's inside it.
        """
        if self._model_bundle is None:
            return None
        pipeline = self._model_bundle["pipeline"]
        features = [build_feature_vector_from_case({**case, "decline_code": code})]
        return float(pipeline.predict_proba(features)[0][1])

    def diagnose(self, case: dict[str, Any]) -> Diagnosis:
        code = case.get("decline_code")

        if code in SOFT_DECLINE_CODES:
            raw_signal = {"decline_code": code, "source": "iso8583_soft_lookup"}
            if self._model_bundle is not None:
                # Audit visibility only (architecture doc 7.2 developer/audit
                # view) — never branched on for behavior.
                raw_signal["model_type"] = self._model_bundle["model_type"]

            return Diagnosis(
                root_cause=SOFT_DECLINE_CODES[code],
                is_recoverable=True,
                confidence=0.95,
                raw_signal=raw_signal,
                predicted_recovery_probability=self._predict_recovery_probability(case, code),
            )

        if code in HARD_DECLINE_CODES:
            return Diagnosis(
                root_cause=HARD_DECLINE_CODES[code],
                is_recoverable=False,
                confidence=0.95,
                raw_signal={"decline_code": code, "source": "visa_category1_lookup"},
                # No prediction: the model was never trained on hard-decline
                # cases (they never reach a real retry), so scoring one
                # would be meaningless, not just unavailable.
            )

        if code in STOP_INSTRUCTION_CODES:
            return Diagnosis(
                root_cause=STOP_INSTRUCTION_CODES[code],
                is_recoverable=False,
                confidence=0.99,
                raw_signal={"decline_code": code, "source": "stop_instruction_lookup"},
            )

        return Diagnosis(
            root_cause="unmapped_decline_code",
            is_recoverable=False,
            confidence=0.2,
            raw_signal={"decline_code": code, "source": "unmapped"},
        )

    def decide(
        self, case: dict[str, Any], diagnosis: Diagnosis, history: list[dict[str, Any]]
    ) -> Decision:
        if diagnosis.confidence < 0.5:
            return Decision(
                action_type=ActionType.ESCALATE,
                reasoning=(
                    f"Decline code '{case.get('decline_code')}' is not in the known "
                    "taxonomy — confidence too low for automated handling."
                ),
                requires_human_review=True,
            )

        if diagnosis.is_recoverable:
            retry_count = sum(1 for h in history if h.get("_event_type") == "ExecutionResult")
            backoff_index = min(retry_count, len(RETRY_BACKOFF_HOURS) - 1)
            backoff_hours = RETRY_BACKOFF_HOURS[backoff_index]

            reasoning = f"Soft decline ({diagnosis.root_cause}), retry #{retry_count + 1}"
            if diagnosis.predicted_recovery_probability is not None:
                model_type = diagnosis.raw_signal.get("model_type", "unknown")
                reasoning += (
                    f" (diagnosis conf={diagnosis.confidence:.2f}, "
                    f"{model_type} predicts {diagnosis.predicted_recovery_probability:.2f} "
                    "recovery probability)"
                )
            else:
                reasoning += " (no trained recovery model applied — rule-based confidence only)"

            return Decision(
                action_type=ActionType.RETRY,
                action_params={"retry_in_hours": backoff_hours},
                reasoning=reasoning,
                requires_human_review=False,
            )

        return Decision(
            action_type=ActionType.STOP,
            reasoning="Diagnosis marked unrecoverable outside the known stop-code set.",
            requires_human_review=True,
        )

    def execute(self, case: dict[str, Any], decision: Decision) -> ExecutionResult:
        return ExecutionResult(
            success=True,
            compliance_check_passed=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def track_outcome(self, case: dict[str, Any]) -> Outcome:
        simulated = case.get("simulated_retry_result")
        if simulated == "recovered":
            return Outcome(
                status=OutcomeStatus.RECOVERED, amount_recovered=case.get("amount", 0.0)
            )
        if simulated == "lost":
            return Outcome(status=OutcomeStatus.LOST, amount_recovered=0.0)
        return Outcome(status=OutcomeStatus.PENDING, amount_recovered=0.0)

    def on_promise_due(self, case: dict[str, Any]) -> PromiseOutcome:
        return PromiseOutcome(kept=True)