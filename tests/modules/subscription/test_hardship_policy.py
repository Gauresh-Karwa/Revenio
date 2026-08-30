from backend.core.contract import ActionType
from backend.modules.subscription.module import SubscriptionModule
from backend.ml.text_signals import (
    extract_hardship_signal,
    extract_hardship_signal_embedding,
)


def test_diagnose_without_email_has_no_hardship():
    module = SubscriptionModule()
    diag = module.diagnose({"decline_code": "51"})
    assert diag.raw_signal["hardship_signal_detected"] is False
    assert diag.raw_signal["extracted_reason_code"] == "no_support_contact"
    assert diag.raw_signal["email_reference_hash"] is None


def test_diagnose_with_hardship_email_detects_signal():
    module = SubscriptionModule()
    case = {
        "decline_code": "51",
        "email_text": "I lost my job last week and can't cover this charge right now.",
    }
    diag = module.diagnose(case)
    assert diag.raw_signal["hardship_signal_detected"] is True
    assert diag.raw_signal["extracted_reason_code"] == "financial_hardship_disclosed"
    assert diag.raw_signal["email_reference_hash"] is not None


def test_diagnose_with_paraphrase_email_detected_by_embedding():
    """
    The embedding extractor catches paraphrases the keyword matcher would miss.
    "Things have been really rough for us financially" contains no hardship
    keyword but semantically matches the anchor sentences (scores 0.55+).
    """
    module = SubscriptionModule()  # default = embedding
    case = {
        "decline_code": "51",
        "email_text": "Things have been really rough for us financially lately, could we work something out?",
    }
    diag = module.diagnose(case)
    assert diag.raw_signal["hardship_signal_detected"] is True


def test_decide_routes_hardship_to_human_review():
    module = SubscriptionModule()
    case = {
        "decline_code": "51",
        "email_text": "I lost my job last week and can't cover this charge right now.",
    }
    diag = module.diagnose(case)
    decision = module.decide(case, diag, history=[])

    assert decision.action_type == ActionType.ESCALATE
    assert decision.requires_human_review is True
    assert "hardship" in decision.reasoning.lower()


def test_decide_with_neutral_email_retries_normally():
    module = SubscriptionModule()
    case = {
        "decline_code": "51",
        "email_text": "Please update my card on file.",
    }
    diag = module.diagnose(case)
    decision = module.decide(case, diag, history=[])

    assert decision.action_type == ActionType.RETRY
    assert decision.requires_human_review is False


def test_swappable_hardship_extractor_injection():
    """
    Prove the extractor swap actually works end-to-end: inject a mock that
    flags all text as hardship, pass neutral text the embedding extractor
    would NOT flag, confirm it is still routed to human review.
    """
    def custom_flag_all_extractor(email_text: str | None) -> dict:
        return {
            "hardship_signal_detected": True,
            "extracted_reason_code": "custom_rule_flagged",
        }

    module = SubscriptionModule(hardship_extractor=custom_flag_all_extractor)
    case = {
        "decline_code": "51",
        "email_text": "Please update my card on file.",  # neutral — would NOT trigger embedding
    }
    diag = module.diagnose(case)
    assert diag.raw_signal["hardship_signal_detected"] is True
    assert diag.raw_signal["extracted_reason_code"] == "custom_rule_flagged"

    decision = module.decide(case, diag, history=[])
    assert decision.action_type == ActionType.ESCALATE
    assert decision.requires_human_review is True


def test_keyword_extractor_swap_is_also_valid():
    """
    Keyword extractor is still a valid swap — e.g. for environments
    where sentence-transformers isn't available.
    """
    module = SubscriptionModule(hardship_extractor=extract_hardship_signal)
    case = {
        "decline_code": "51",
        "email_text": "I cannot afford this right now.",
    }
    diag = module.diagnose(case)
    assert diag.raw_signal["hardship_signal_detected"] is True
