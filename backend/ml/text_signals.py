"""
Extracts a structured hardship signal from free-text customer communication
(e.g. a support email) — the concrete implementation of the design decision
discussed before step 6: don't feed raw text into GBM; extract a structured
signal upstream, feed THAT into the existing flat feature pipeline exactly
like customer_recent_failure_pressure.

THE SWAPPABILITY DESIGN:
Every extractor has the SAME signature —
`(email_text: str | None) -> dict` returning
{"hardship_signal_detected": bool, "extracted_reason_code": str}.
SubscriptionModule takes one as a constructor argument, defaulting to
extract_hardship_signal_embedding. Swapping is a one-argument change —
nothing in features.py, diagnose(), or decide() changes.

THREE IMPLEMENTATIONS:
- extract_hardship_signal_embedding: DEFAULT. Uses sentence-transformers
  (all-MiniLM-L6-v2, ~80MB, downloads once, then runs fully offline)
  to compute cosine similarity against hardship anchor sentences. Catches
  paraphrases that exact keywords miss. Falls back to the keyword matcher
  if sentence-transformers is not installed. Zero API cost, zero latency
  beyond the first model load, fully offline.
- extract_hardship_signal: keyword matcher backup. Deliberately imperfect
  on paraphrases. Zero external dependencies.
- extract_hardship_signal_llm: Claude API, requires ANTHROPIC_API_KEY.
  NOT used anywhere by default — here for completeness if you want to
  swap it in explicitly.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable

HardshipExtractor = Callable[[str | None], dict]

# ------------------------------------------------------------------ #
# Shared constants                                                     #
# ------------------------------------------------------------------ #

NO_CONTACT_REASON_CODE = "no_support_contact"
HARDSHIP_REASON_CODE = "financial_hardship_disclosed"
NO_SIGNAL_REASON_CODE = "no_hardship_signal_detected"
EXTRACTION_ERROR_REASON_CODE = "extraction_failed"

# ------------------------------------------------------------------ #
# Embedding extractor — DEFAULT                                        #
# ------------------------------------------------------------------ #

_HARDSHIP_ANCHORS = [
    # Direct/explicit hardship
    "I lost my job and cannot afford this payment.",
    "We are going through a financial hardship right now.",
    "My hours got cut and I cannot cover this charge.",
    "I cannot afford this right now, things are hard.",
    "I am in a really tough financial situation at the moment.",
    # Medical emergency with financial framing
    "There has been a medical emergency and money is very tight.",
    "I am going through a medical crisis and struggling to pay.",
    "Medical bills have left me unable to cover this charge.",
    # Indirect/soft paraphrase
    "Things have been very difficult for us financially lately.",
    "We are struggling and going through a hard time right now.",
    "There was a death in the family and finances are a mess.",
]

_NEUTRAL_ANCHORS = [
    # Billing/account management inquiries — the class of text that lexically
    # overlaps with hardship anchors (words like 'charged', 'payment') but is
    # not expressing financial distress.
    "When will my card be charged?",
    "I want to update my payment method on file.",
    "Can you change my billing date?",
    "Please update my billing email address.",
    "I would like to cancel my subscription.",
    "Why was I charged twice this month?",
    "How do I pause my subscription?",
    "I want to switch to a different plan.",
]

_CONTRASTIVE_MARGIN = 0.25        # H-N above this -> high-confidence hardship
_CONTRASTIVE_UNCERTAIN_FLOOR = 0.05  # H-N in [0.05, 0.25] -> uncertain, escalate anyway
# Calibrated from measured H-N scores on all-MiniLM-L6-v2:
#   Hardship floor: +0.30 ("I lost my job...", lowest hardship sentence)
#   Neutral ceiling: -0.31 ("Please update my card on file.")
#   Billing inquiry near boundary: -0.47 ("when will my card be charged?")
#   Gap between hardship floor and neutral ceiling: 0.61
#
# THREE ZONES (the fix for out-of-distribution / unusual text):
#   H-N > 0.25          -> HIGH confidence hardship detected. Escalate.
#   0.05 < H-N <= 0.25  -> UNCERTAIN. Anchor bank cannot cleanly place this
#                          text. Escalate to human review — safer than a
#                          binary call on text we haven't seen before.
#   H-N <= 0.05         -> NONE. Not hardship. Continue normal flow.
#
# DO NOT lower _CONTRASTIVE_UNCERTAIN_FLOOR below 0.0 — the billing inquiry
# problem child lives at H-N=-0.47 and is already safely in the NONE zone.

_embedding_model = None
_hardship_embeddings = None
_neutral_embeddings = None


def _get_embedding_model():
    """
    Loads sentence-transformers model on first call, caches it globally.
    Encodes both hardship and neutral anchor banks on load.
    Raises ImportError clearly if sentence-transformers is not installed.
    """
    global _embedding_model, _hardship_embeddings, _neutral_embeddings
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        _hardship_embeddings = _embedding_model.encode(
            _HARDSHIP_ANCHORS, convert_to_tensor=True, show_progress_bar=False
        )
        _neutral_embeddings = _embedding_model.encode(
            _NEUTRAL_ANCHORS, convert_to_tensor=True, show_progress_bar=False
        )
    return _embedding_model, _hardship_embeddings, _neutral_embeddings


def extract_hardship_signal_embedding(email_text: str | None) -> dict:
    """
    DEFAULT extractor. Uses contrastive embedding scoring:

        H = max cosine similarity to hardship anchor bank
        N = max cosine similarity to neutral/billing anchor bank
        detected = (H - N) > _CONTRASTIVE_MARGIN

    This directly solves the problem with single-anchor scoring: billing
    inquiries that mention 'charged' or 'payment' score moderately against
    hardship anchors (H=0.43), but score very high against neutral anchors
    (N=0.90), giving H-N=-0.47 — correctly rejected.
    Genuine hardship sentences score H-N >= +0.30 even at the floor.

    Falls back to the keyword extractor if sentence-transformers is not
    installed, with '_keyword_fallback' appended to the reason code so the
    fallback is always visible in the audit log.
    """
    if email_text is None:
        return {
            "hardship_signal_detected": False,
            "extracted_reason_code": NO_CONTACT_REASON_CODE,
        }

    try:
        from sentence_transformers import util

        model, h_anchors, n_anchors = _get_embedding_model()
        text_emb = model.encode(email_text, convert_to_tensor=True, show_progress_bar=False)
        h_score = float(util.cos_sim(text_emb, h_anchors).max())
        n_score = float(util.cos_sim(text_emb, n_anchors).max())
        contrastive = round(h_score - n_score, 4)

        if contrastive > _CONTRASTIVE_MARGIN:
            tier = "high"
            detected = True
        elif contrastive > _CONTRASTIVE_UNCERTAIN_FLOOR:
            # Unusual or out-of-distribution text: the anchor bank can't place
            # it cleanly. We expose this as a distinct tier so the policy layer
            # can escalate to human review instead of making a binary call.
            tier = "uncertain"
            detected = True  # treat as potential hardship — escalate is safer
        else:
            tier = "none"
            detected = False

        return {
            "hardship_signal_detected": detected,
            "hardship_confidence_tier": tier,
            "extracted_reason_code": HARDSHIP_REASON_CODE if detected else NO_SIGNAL_REASON_CODE,
            "hardship_similarity": round(h_score, 4),
            "neutral_similarity": round(n_score, 4),
            "contrastive_score": contrastive,
        }
    except ImportError:
        result = extract_hardship_signal(email_text)
        result["extracted_reason_code"] = result["extracted_reason_code"] + "_keyword_fallback"
        return result


# ------------------------------------------------------------------ #
# Keyword extractor — backup / fallback                               #
# ------------------------------------------------------------------ #

HARDSHIP_KEYWORDS: list[str] = [
    "lost my job",
    "lost his job",
    "lost her job",
    "medical emergency",
    "financial hardship",
    "financial situation",
    "can't afford",
    "cannot afford",
    "hours got cut",
    "death in the family",
    "emergency",
]


def extract_hardship_signal(email_text: str | None) -> dict:
    """
    BACKUP keyword extractor. Returns
    {"hardship_signal_detected": bool, "extracted_reason_code": str}.
    Used as the fallback inside extract_hardship_signal_embedding if
    sentence-transformers is not installed.
    """
    if email_text is None:
        return {
            "hardship_signal_detected": False,
            "extracted_reason_code": NO_CONTACT_REASON_CODE,
        }

    lowered = email_text.lower()
    detected = any(keyword in lowered for keyword in HARDSHIP_KEYWORDS)

    return {
        "hardship_signal_detected": detected,
        "extracted_reason_code": HARDSHIP_REASON_CODE if detected else NO_SIGNAL_REASON_CODE,
    }


# ------------------------------------------------------------------ #
# LLM extractor — explicit opt-in only, never used by default         #
# ------------------------------------------------------------------ #

def extract_hardship_signal_llm(
    email_text: str | None,
    model: str = "claude-haiku-4-5-20251001",
    fallback_to_keyword_on_error: bool = True,
) -> dict:
    """
    Claude API extractor. Same output shape as the other extractors.
    Requires: `pip install anthropic` and ANTHROPIC_API_KEY set.
    NOT used by default anywhere — explicit opt-in only:
        SubscriptionModule(hardship_extractor=extract_hardship_signal_llm)
    """
    if email_text is None:
        return {
            "hardship_signal_detected": False,
            "extracted_reason_code": NO_CONTACT_REASON_CODE,
        }

    try:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are classifying a single customer support message for a "
                        "payment-recovery system. Determine whether the customer is "
                        "disclosing genuine financial hardship (e.g. job loss, medical "
                        "emergency, inability to pay) as opposed to a routine billing "
                        "question or an unrelated request.\n\n"
                        f"Message:\n\"\"\"\n{email_text}\n\"\"\"\n\n"
                        "Respond with ONLY a JSON object, no other text, in exactly this "
                        'shape: {"hardship_signal_detected": true or false, '
                        '"extracted_reason_code": a short snake_case string describing '
                        "the specific reason if hardship is detected, or "
                        '"no_hardship_signal_detected" if not.}'
                    ),
                }
            ],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        parsed = json.loads(text)
        return {
            "hardship_signal_detected": bool(parsed["hardship_signal_detected"]),
            "extracted_reason_code": str(parsed["extracted_reason_code"]),
        }
    except Exception:
        if fallback_to_keyword_on_error:
            result = extract_hardship_signal(email_text)
            result["extracted_reason_code"] = result["extracted_reason_code"] + "_llm_fallback"
            return result
        raise


# ------------------------------------------------------------------ #
# Utilities                                                           #
# ------------------------------------------------------------------ #

def hash_email_reference(email_text: str | None) -> str | None:
    """
    A stable, non-reversible reference to a piece of email text, safe to
    store in the audit log without storing the text itself.
    """
    if email_text is None:
        return None
    return hashlib.sha256(email_text.encode("utf-8")).hexdigest()[:16]
