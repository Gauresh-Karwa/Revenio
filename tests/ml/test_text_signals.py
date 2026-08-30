from backend.ml.text_signals import (
    HARDSHIP_REASON_CODE,
    NO_CONTACT_REASON_CODE,
    NO_SIGNAL_REASON_CODE,
    extract_hardship_signal,
    extract_hardship_signal_embedding,
    hash_email_reference,
)


# ------------------------------------------------------------------ #
# Keyword extractor tests                                              #
# ------------------------------------------------------------------ #

def test_keyword_extractor_none():
    res = extract_hardship_signal(None)
    assert res["hardship_signal_detected"] is False
    assert res["extracted_reason_code"] == NO_CONTACT_REASON_CODE


def test_keyword_extractor_hardship_texts():
    hardship_texts = [
        "I lost my job last week and can't cover this charge right now.",
        "Going through a medical emergency, please give me some time.",
        "I'm in a really tough financial situation at the moment.",
        "My hours got cut and money is tight this month.",
        "There's been a death in the family and finances are a mess right now.",
        "I cannot afford this right now.",
    ]
    for text in hardship_texts:
        res = extract_hardship_signal(text)
        assert res["hardship_signal_detected"] is True, f"Expected detection for: {text!r}"
        assert res["extracted_reason_code"] == HARDSHIP_REASON_CODE


def test_keyword_extractor_neutral_texts():
    neutral_texts = [
        "Can you tell me when my card will be charged again?",
        "I'd like to update my payment method on file.",
        "Please cancel my subscription for next month.",
        "Why was my payment declined? My card should be fine.",
        "I want to change my billing email address.",
    ]
    for text in neutral_texts:
        res = extract_hardship_signal(text)
        assert res["hardship_signal_detected"] is False, f"False positive for: {text!r}"
        assert res["extracted_reason_code"] == NO_SIGNAL_REASON_CODE


# ------------------------------------------------------------------ #
# Embedding extractor tests                                            #
# ------------------------------------------------------------------ #

def test_embedding_extractor_none():
    res = extract_hardship_signal_embedding(None)
    assert res["hardship_signal_detected"] is False
    assert res["extracted_reason_code"] == NO_CONTACT_REASON_CODE


def test_embedding_extractor_detects_direct_hardship():
    hardship_texts = [
        "I lost my job last week and can't cover this charge right now.",
        # Medical emergency that mentions financial impact — model can separate
        # pure medical context (0.42) from financial impact framing (0.55+).
        "I've had a medical emergency and cannot afford this payment.",
        "I cannot afford this right now.",
    ]
    for text in hardship_texts:
        res = extract_hardship_signal_embedding(text)
        assert res["hardship_signal_detected"] is True, f"Expected detection for: {text!r}"


def test_embedding_extractor_detects_paraphrase_keyword_misses():
    """
    The key advantage of the embedding extractor: paraphrases the keyword
    matcher would never catch are caught by semantic similarity.
    These score 0.55-0.61 with the richer anchor bank vs the keyword
    matcher returning False for both.
    """
    paraphrases = [
        "Things have been really rough for us lately, could we work something out?",
        "We're dealing with a hard stretch right now financially, hope you understand.",
    ]
    for text in paraphrases:
        res = extract_hardship_signal_embedding(text)
        assert res["hardship_signal_detected"] is True, (
            f"Embedding extractor should catch paraphrase: {text!r}"
        )


def test_embedding_extractor_does_not_flag_neutral():
    neutral_texts = [
        "Please update my card on file.",
        "Can you tell me when my card will be charged again?",  # problem child — now handled
        "I want to change my billing email address.",
        "Please cancel my subscription.",
    ]
    for text in neutral_texts:
        res = extract_hardship_signal_embedding(text)
        assert res["hardship_signal_detected"] is False, f"False positive for: {text!r}"


def test_embedding_extractor_includes_similarity_score():
    res = extract_hardship_signal_embedding("I lost my job and cannot pay.")
    assert "hardship_similarity" in res
    assert "neutral_similarity" in res
    assert "contrastive_score" in res
    assert "hardship_confidence_tier" in res
    assert res["contrastive_score"] > 0.25   # high-confidence hardship sentence
    assert res["hardship_confidence_tier"] == "high"


def test_embedding_extractor_uncertain_tier_escalates():
    """
    Text that falls in the uncertain band (H-N between 0.05 and 0.25) is
    treated as detected=True so the policy layer escalates it to human review
    rather than auto-retrying. This is the safe behaviour for out-of-
    distribution text the anchor bank cannot cleanly classify.
    """
    from backend.ml.text_signals import _CONTRASTIVE_MARGIN, _CONTRASTIVE_UNCERTAIN_FLOOR

    # Synthesise a text likely to land in the uncertain band by checking
    # the contrastive score at runtime rather than hardcoding a sentence
    # that might shift with future anchor updates.
    test_sentences = [
        "Things have been difficult and we need some flexibility.",
        "We are in a tough spot right now, any help would be great.",
    ]
    for text in test_sentences:
        res = extract_hardship_signal_embedding(text)
        score = res["contrastive_score"]
        tier = res["hardship_confidence_tier"]
        if _CONTRASTIVE_UNCERTAIN_FLOOR < score <= _CONTRASTIVE_MARGIN:
            # Confirmed in uncertain band — must be detected=True (escalate)
            assert res["hardship_signal_detected"] is True, (
                f"Uncertain-band text should be detected=True for safe escalation: {text!r}"
            )
            assert tier == "uncertain"
        elif score > _CONTRASTIVE_MARGIN:
            assert tier == "high"
            assert res["hardship_signal_detected"] is True
        else:
            assert tier == "none"
            assert res["hardship_signal_detected"] is False


def test_embedding_extractor_contrastive_rejects_billing_inquiry():
    """
    The billing inquiry that previously scored 0.4284 against hardship anchors
    (a false positive risk with single-anchor scoring) is correctly rejected
    with contrastive scoring: H=0.43, N=0.90, H-N=-0.47.
    """
    res = extract_hardship_signal_embedding("Can you tell me when my card will be charged again?")
    assert res["hardship_signal_detected"] is False
    assert res["contrastive_score"] < 0  # strongly negative = billing, not hardship


# ------------------------------------------------------------------ #
# Utility tests                                                        #
# ------------------------------------------------------------------ #

def test_hash_email_reference():
    assert hash_email_reference(None) is None
    h1 = hash_email_reference("I lost my job")
    h2 = hash_email_reference("I lost my job")
    h3 = hash_email_reference("Different text")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16
