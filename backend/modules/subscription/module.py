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
from backend.data.subscription_generator import compute_pressure_from_customer_history
from backend.ml.features import (
    FEATURE_NAMES_WITH_HISTORY_AND_TEXT,
    build_feature_vector_with_history_and_text,
)
from backend.ml.text_signals import (
    HardshipExtractor,
    add_confirmed_hardship_anchor,
    add_confirmed_neutral_anchor,
    extract_hardship_signal_embedding,
    hash_email_reference,
)

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
    if not model_path.exists():
        return None
    try:
        import joblib

        bundle = joblib.load(model_path)
    except Exception:
        return None

    if bundle.get("feature_names") != FEATURE_NAMES_WITH_HISTORY_AND_TEXT:
        warnings.warn(
            "Loaded model bundle's feature_names does not match the current "
            "(enriched, with customer_recent_failure_pressure and "
            "hardship_signal_detected) schema — refusing to use it. Re-run "
            "train_subscription_model.py to retrain against the current "
            "feature set. Falling back to rule-based confidence only.",
            stacklevel=2,
        )
        return None

    return bundle


class SubscriptionModule:
    domain_type = "subscription"

    def __init__(
        self,
        model_path: Path | str | None = None,
        hardship_extractor: HardshipExtractor = extract_hardship_signal_embedding,
        learning_core: Any = None,
        anchor_growth_callback: Any = add_confirmed_hardship_anchor,
        neutral_anchor_growth_callback: Any = add_confirmed_neutral_anchor,
        channel_gateway: Any = None,
    ) -> None:
        """
        hardship_extractor: swap point for the signal extractor.

        learning_core: optional LearningCore. When provided AND it has a
        policy for "subscription", decide() asks it which retry-backoff
        arm to pull instead of the fixed escalation schedule. check_stop
        also consults it for a bandit-informed DIMINISHING_RETURNS stop —
        see check_stop's docstring.

        anchor_growth_callback / neutral_anchor_growth_callback: the
        step-6 feedback-loop swap points. anchor_growth_callback fires
        when a human confirms an `uncertain`-tier case WAS hardship
        (grows the hardship bank). neutral_anchor_growth_callback fires
        when a human confirms an `uncertain`-tier case was NOT hardship
        (grows the neutral bank) — this symmetric path previously did not
        exist. Both default to the embedding extractor's own anchor banks
        — if you swap hardship_extractor, pass both as None (or your own
        compatible functions) too.
        """
        path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self._model_bundle = _load_model_bundle(path)
        self._hardship_extractor = hardship_extractor
        self._learning_core = learning_core
        self._anchor_growth_callback = anchor_growth_callback
        self._neutral_anchor_growth_callback = neutral_anchor_growth_callback
        self._channel_gateway = channel_gateway

    def on_human_review_confirmed(
        self, case: dict[str, Any], confirmed: bool, last_diagnosis_payload: dict[str, Any]
    ) -> None:
        """
        THE step-6 feedback loop, concretely. Called by
        Orchestrator.submit_human_review.

        Both directions of feedback are handled, symmetrically: confirmed
        WAS hardship grows the hardship bank; confirmed WAS NOT hardship
        grows the neutral bank. Only `uncertain`-tier cases feed either
        bank — `high` was already confidently correct, `none` was never
        flagged for hardship review in the first place.
        """
        raw_signal = last_diagnosis_payload.get("raw_signal", {})
        if raw_signal.get("hardship_confidence_tier") != "uncertain":
            return

        email_text = case.get("email_text")
        if email_text is None:
            return

        if confirmed:
            if self._anchor_growth_callback is not None:
                self._anchor_growth_callback(email_text)
        else:
            if self._neutral_anchor_growth_callback is not None:
                self._neutral_anchor_growth_callback(email_text)

    # Bandit-informed diminishing-returns thresholds. Only meaningful once
    # a learning_core is wired AND it has genuinely learned something —
    # both magnitudes below are estimated judgment calls, not sourced.
    DIMINISHING_RETURNS_MIN_PULLS_PER_ARM = 20
    DIMINISHING_RETURNS_PROBABILITY_FLOOR = 0.10
    DIMINISHING_RETURNS_MIN_CASE_RETRIES = 2

    def check_stop(
        self, case: dict[str, Any], history: list[dict[str, Any]]
    ) -> StopDecision:
        """
        DIMINISHING_RETURNS is a real StopReason defined in contract.py but,
        before this, never actually fired by any module — the only way to
        know "further retries are unlikely to help" is a learned per-arm
        success-rate estimate, which didn't exist before the learning
        core. When a learning_core is wired and has accumulated enough
        data (every arm pulled at least DIMINISHING_RETURNS_MIN_PULLS_PER_ARM
        times) and even its best-estimated arm's recovery odds are below
        DIMINISHING_RETURNS_PROBABILITY_FLOOR, this case stops early —
        smarter than grinding to the blunt MAX_RETRY_ATTEMPTS ceiling
        regardless of whether continuing is worth it. Without a
        learning_core (or before it has enough data), this never fires —
        behavior is identical to before.
        """
        code = case.get("decline_code")

        if code in HARD_DECLINE_CODES:
            return StopDecision(should_stop=True, stop_reason=StopReason.COMPLIANCE_LIMIT)

        if code in STOP_INSTRUCTION_CODES:
            return StopDecision(should_stop=True, stop_reason=StopReason.OPT_OUT)

        retry_count = sum(1 for h in history if h.get("_event_type") == "ExecutionResult")

        if (
            self._learning_core is not None
            and self._learning_core.has_policy(self.domain_type)
            and retry_count >= self.DIMINISHING_RETURNS_MIN_CASE_RETRIES
        ):
            arms = self._learning_core.snapshot()[self.domain_type]["arms"]
            if all(a["pull_count"] >= self.DIMINISHING_RETURNS_MIN_PULLS_PER_ARM for a in arms):
                best_estimate = max(
                    (a.get("mean_estimate", a.get("mean_reward")) or 0.0) for a in arms
                )
                if best_estimate < self.DIMINISHING_RETURNS_PROBABILITY_FLOOR:
                    return StopDecision(should_stop=True, stop_reason=StopReason.DIMINISHING_RETURNS)

        if retry_count >= MAX_RETRY_ATTEMPTS:
            return StopDecision(should_stop=True, stop_reason=StopReason.COMPLIANCE_LIMIT)

        return StopDecision(should_stop=False)

    def _predict_recovery_probability(
        self,
        case: dict[str, Any],
        code: str,
        customer_recent_failure_pressure: float,
        hardship_signal_detected: bool,
    ) -> float | None:
        if self._model_bundle is None:
            return None
        pipeline = self._model_bundle["pipeline"]
        features = [
            build_feature_vector_with_history_and_text(
                decline_code=code,
                attempt_number=case.get("attempt_number", 1),
                hour_of_day=case.get("hour_of_day", 12),
                is_near_payday=case.get("is_near_payday", False),
                amount=case.get("amount", 0.0),
                customer_recent_failure_pressure=customer_recent_failure_pressure,
                hardship_signal_detected=hardship_signal_detected,
            )
        ]
        return float(pipeline.predict_proba(features)[0][1])

    def diagnose(
        self, case: dict[str, Any], customer_history: list[dict[str, Any]] | None = None
    ) -> Diagnosis:
        code = case.get("decline_code")

        past_case_outcomes = [
            event.get("status") == "RECOVERED"
            for event in (customer_history or [])
            if event.get("_event_type") == "Outcome" and event.get("status") in ("RECOVERED", "LOST")
        ]
        customer_recent_failure_pressure = compute_pressure_from_customer_history(past_case_outcomes)

        email_text = case.get("email_text")
        # A response supplied by an approved support channel is already a
        # direct customer statement. It should not be reclassified through
        # an email-language model before the mandatory human-review gate.
        # This is distinct from an inferred text signal and keeps the source
        # clear in the durable, privacy-safe diagnosis.
        hardship_extraction = (
            {
                "hardship_signal_detected": True,
                "hardship_confidence_tier": "high",
                "extracted_reason_code": "customer_reported_hardship",
            }
            if case.get("customer_reported_hardship", False)
            else self._hardship_extractor(email_text)
        )
        hardship_signal_detected = hardship_extraction["hardship_signal_detected"]

        if code in SOFT_DECLINE_CODES:
            raw_signal = {"decline_code": code, "source": "iso8583_soft_lookup"}
            if self._model_bundle is not None:
                raw_signal["model_type"] = self._model_bundle["model_type"]
            raw_signal["customer_recent_failure_pressure"] = customer_recent_failure_pressure
            raw_signal["n_past_cases_considered"] = len(past_case_outcomes)
            raw_signal["hardship_signal_detected"] = hardship_signal_detected
            raw_signal["hardship_confidence_tier"] = hardship_extraction.get(
                "hardship_confidence_tier", "high" if hardship_signal_detected else "none"
            )
            raw_signal["extracted_reason_code"] = hardship_extraction["extracted_reason_code"]
            raw_signal["email_reference_hash"] = hash_email_reference(email_text)

            return Diagnosis(
                root_cause=SOFT_DECLINE_CODES[code],
                is_recoverable=True,
                confidence=0.95,
                raw_signal=raw_signal,
                predicted_recovery_probability=self._predict_recovery_probability(
                    case, code, customer_recent_failure_pressure, hardship_signal_detected
                ),
            )

        if code in HARD_DECLINE_CODES:
            return Diagnosis(
                root_cause=HARD_DECLINE_CODES[code],
                is_recoverable=False,
                confidence=0.95,
                raw_signal={"decline_code": code, "source": "visa_category1_lookup"},
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
        if diagnosis.raw_signal.get("hardship_signal_detected"):
            tier = diagnosis.raw_signal.get("hardship_confidence_tier", "high")
            if tier == "uncertain":
                reasoning = (
                    "Customer email could not be confidently classified by the "
                    "extractor (contrastive score in uncertain band) — routed to "
                    "human review rather than making a binary call on out-of-"
                    "distribution text."
                )
            else:
                reasoning = (
                    "Customer disclosed financial hardship — routed to human "
                    "review regardless of model confidence, per policy."
                )
            return Decision(
                action_type=ActionType.ESCALATE,
                reasoning=reasoning,
                requires_human_review=True,
            )

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

            action_params: dict[str, Any] = {}
            if self._learning_core is not None and self._learning_core.has_policy(self.domain_type):
                # Step 6 wiring: the bandit chooses which backoff arm to pull
                # for THIS retry, instead of the fixed escalating schedule.
                arm = self._learning_core.select_arm(self.domain_type)
                backoff_hours = RETRY_BACKOFF_HOURS[arm]
                action_params["bandit_arm"] = arm
            else:
                backoff_index = min(retry_count, len(RETRY_BACKOFF_HOURS) - 1)
                backoff_hours = RETRY_BACKOFF_HOURS[backoff_index]

            action_params["retry_in_hours"] = backoff_hours

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
                action_params=action_params,
                reasoning=reasoning,
                requires_human_review=False,
            )

        return Decision(
            action_type=ActionType.STOP,
            reasoning="Diagnosis marked unrecoverable outside the known stop-code set.",
            requires_human_review=True,
        )

    def execute(self, case: dict[str, Any], decision: Decision) -> ExecutionResult:
        details = self._channel_gateway.dispatch(case, decision) if self._channel_gateway else {}
        return ExecutionResult(
            success=True,
            compliance_check_passed=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details,
        )

    def track_outcome(self, case: dict[str, Any]) -> Outcome:
        if case.get("awaiting_razorpay_confirmation"):
            return Outcome(status=OutcomeStatus.PENDING, amount_recovered=0.0)
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
