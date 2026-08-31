from backend.core.events import EventStore
from backend.core.orchestrator import Orchestrator
from backend.modules.subscription.module import SubscriptionModule


def _uncertain_tier_extractor(email_text):
    return {
        "hardship_signal_detected": True,
        "hardship_confidence_tier": "uncertain",
        "extracted_reason_code": "financial_hardship_disclosed",
    }


def _make_system(extractor, hardship_calls, neutral_calls):
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(
        SubscriptionModule(
            hardship_extractor=extractor,
            anchor_growth_callback=hardship_calls.append,
            neutral_anchor_growth_callback=neutral_calls.append,
        )
    )
    return store, orchestrator


def test_confirmed_false_on_uncertain_tier_grows_neutral_bank_not_hardship_bank():
    hardship_calls, neutral_calls = [], []
    store, orchestrator = _make_system(_uncertain_tier_extractor, hardship_calls, neutral_calls)

    case = {"decline_code": "51", "email_text": "Ambiguous message that isn't really hardship."}
    orchestrator.process_case("case-1", "subscription", case)
    orchestrator.submit_human_review("case-1", confirmed=False, case=case)

    assert hardship_calls == []
    assert neutral_calls == [case["email_text"]]


def test_confirmed_true_still_grows_hardship_bank_not_neutral_bank():
    hardship_calls, neutral_calls = [], []
    store, orchestrator = _make_system(_uncertain_tier_extractor, hardship_calls, neutral_calls)

    case = {"decline_code": "51", "email_text": "Things have been hard lately."}
    orchestrator.process_case("case-1", "subscription", case)
    orchestrator.submit_human_review("case-1", confirmed=True, case=case)

    assert hardship_calls == [case["email_text"]]
    assert neutral_calls == []


def test_neutral_growth_callback_none_disables_that_side_only():
    hardship_calls = []
    store = EventStore()
    orchestrator = Orchestrator(store)
    orchestrator.register_module(
        SubscriptionModule(
            hardship_extractor=_uncertain_tier_extractor,
            anchor_growth_callback=hardship_calls.append,
            neutral_anchor_growth_callback=None,
        )
    )
    case = {"decline_code": "51", "email_text": "Some message."}
    orchestrator.process_case("case-1", "subscription", case)
    orchestrator.submit_human_review("case-1", confirmed=False, case=case)  # must not raise
    assert hardship_calls == []


def test_add_confirmed_neutral_anchor_appends_and_invalidates_cache():
    import backend.ml.text_signals as ts

    original_anchors = list(ts._NEUTRAL_ANCHORS)
    before_count = ts.get_neutral_anchor_count()
    ts._neutral_embeddings = "sentinel"  # simulate a populated cache

    new_sentence = "This is a brand new neutral confirmation sentence for testing."
    try:
        ts.add_confirmed_neutral_anchor(new_sentence)

        assert ts.get_neutral_anchor_count() == before_count + 1
        assert new_sentence in ts._NEUTRAL_ANCHORS
        assert ts._neutral_embeddings is None  # cache invalidated

        # duplicate add is a no-op, doesn't grow the bank twice
        ts._neutral_embeddings = "sentinel_again"
        ts.add_confirmed_neutral_anchor(new_sentence)
        assert ts.get_neutral_anchor_count() == before_count + 1
        assert ts._neutral_embeddings == "sentinel_again"  # cache NOT invalidated on a no-op duplicate
    finally:
        # Clean up global state so other tests are not contaminated
        ts._NEUTRAL_ANCHORS = original_anchors
        ts._neutral_embeddings = None
