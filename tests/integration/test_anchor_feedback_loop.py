"""
Tests the anchor-feedback loop's MECHANISM — the gating logic, the
privacy-preserving durable/ephemeral split, and the orchestrator wiring —
using a mock extractor and a mock growth callback. This is deliberately
independent of the real embedding model (which needs network access to
download on first use) so the mechanism itself is proven correct
regardless of environment.
"""

import pytest

from backend.core.events import EventStore
from backend.core.orchestrator import Orchestrator
from backend.modules.checkout_abandonment.module import CheckoutAbandonmentModule
from backend.modules.subscription.module import SubscriptionModule


def _uncertain_tier_extractor(email_text):
    return {
        "hardship_signal_detected": True,
        "hardship_confidence_tier": "uncertain",
        "extracted_reason_code": "financial_hardship_disclosed",
    }


def _high_tier_extractor(email_text):
    return {
        "hardship_signal_detected": True,
        "hardship_confidence_tier": "high",
        "extracted_reason_code": "financial_hardship_disclosed",
    }


def _none_tier_extractor(email_text):
    return {
        "hardship_signal_detected": False,
        "hardship_confidence_tier": "none",
        "extracted_reason_code": "no_hardship_signal_detected",
    }


def _make_system(extractor, growth_callback):
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(
        SubscriptionModule(hardship_extractor=extractor, anchor_growth_callback=growth_callback)
    )
    return store, orchestrator


def test_uncertain_tier_confirmed_true_grows_the_anchor():
    calls = []
    store, orchestrator = _make_system(_uncertain_tier_extractor, calls.append)

    case = {"decline_code": "51", "email_text": "Things have been a struggle lately, hope you understand."}
    orchestrator.process_case("case-1", "subscription", case)

    orchestrator.submit_human_review("case-1", confirmed=True, case=case)

    assert calls == [case["email_text"]]


def test_uncertain_tier_confirmed_false_does_not_grow_the_anchor():
    calls = []
    store, orchestrator = _make_system(_uncertain_tier_extractor, calls.append)

    case = {"decline_code": "51", "email_text": "Ambiguous message."}
    orchestrator.process_case("case-1", "subscription", case)
    orchestrator.submit_human_review("case-1", confirmed=False, case=case)

    assert calls == []


def test_high_tier_confirmation_does_not_grow_the_anchor():
    calls = []
    store, orchestrator = _make_system(_high_tier_extractor, calls.append)

    case = {"decline_code": "51", "email_text": "I lost my job and cannot afford this."}
    orchestrator.process_case("case-1", "subscription", case)
    orchestrator.submit_human_review("case-1", confirmed=True, case=case)

    assert calls == []


def test_none_tier_case_never_reaches_review_so_never_grows_anchor():
    calls = []
    store, orchestrator = _make_system(_none_tier_extractor, calls.append)

    case = {"decline_code": "51", "email_text": "Please update my card on file."}
    orchestrator.process_case("case-1", "subscription", case)

    diag = [e for e in store.get_events("case-1") if e.event_type == "Diagnosis"][0]
    assert diag.payload["raw_signal"]["hardship_confidence_tier"] == "none"
    review_events = [e for e in store.get_events("case-1") if e.event_type == "PendingHumanReview"]
    assert review_events == []


def test_anchor_growth_callback_none_disables_the_feedback_loop_entirely():
    store, orchestrator = _make_system(_uncertain_tier_extractor, None)
    case = {"decline_code": "51", "email_text": "Hard to say what's going on for me right now."}
    orchestrator.process_case("case-1", "subscription", case)
    orchestrator.submit_human_review("case-1", confirmed=True, case=case)


def test_durable_event_never_contains_the_raw_email_text():
    calls = []
    store, orchestrator = _make_system(_uncertain_tier_extractor, calls.append)

    sensitive_text = "I lost my job last week and things are very hard right now."
    case = {"decline_code": "51", "email_text": sensitive_text}
    orchestrator.process_case("case-1", "subscription", case)
    orchestrator.submit_human_review("case-1", confirmed=True, case=case)

    review_event = [e for e in store.get_events("case-1") if e.event_type == "HumanReviewDecision"][0]
    assert review_event.payload == {"confirmed": True}
    for value in review_event.payload.values():
        if isinstance(value, str):
            assert sensitive_text not in value


def test_submit_human_review_raises_for_unknown_case():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(SubscriptionModule(hardship_extractor=_uncertain_tier_extractor))

    with pytest.raises(ValueError):
        orchestrator.submit_human_review("nonexistent-case", confirmed=True, case={})


def test_module_without_on_human_review_confirmed_is_handled_gracefully():
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(CheckoutAbandonmentModule())

    case = {"reached_checkout": True, "opt_in": True, "abandonment_signal": "unmapped_thing"}
    orchestrator.process_case("case-1", "checkout_abandonment", case)

    orchestrator.submit_human_review("case-1", confirmed=True, case=case)
