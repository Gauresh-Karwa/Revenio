# Revenio

**Demo:** https://youtu.be/KPbci4vFo6I

Razorpay Hackathon — Track 3: AI Revenue Recovery

---

Revenio is a payment recovery system for merchants dealing with failed subscriptions, abandoned checkouts, overdue B2B invoices, and broken UPI/NACH mandates.

The key distinction from most recovery tooling: communication and revenue are tracked separately. An SMS delivered or a link clicked is not recovery. Recovery is recorded only after Razorpay returns a signed webhook confirming actual payment. Everything on the dashboard is verified money.

Compliance is not an afterthought. Retrying a stolen or expired card violates Visa Category 1 rules and can result in merchant penalties. Calling outside TRAI DND windows is illegal. Sending recovery messages without DPDP consent is a liability. Revenio enforces all of this at the architecture level, not as optional configuration.

---

## How It Works

The system is built around four independent recovery modules card subscriptions, checkout abandonment, B2B receivables, and UPI/NACH mandates. Each module plugs into a shared orchestrator through the same five-method interface (`check_stop`, `diagnose`, `decide`, `execute`, `track_outcome`). The orchestrator handles the loop, the audit log, and the human review queue. It contains no domain logic of its own.

State is stored as an append-only event log in PostgreSQL. There is no separate state table to keep in sync state is always derived by replaying the log, which means the audit trail is complete by construction and there is no way for state to go silently wrong.

---

## Recovery Logic

**Card subscriptions**

The system reads the ISO 8583 decline code before doing anything. Expired, stolen, and lost cards get an immediate permanent stop no retry, no escalation, because attempting those is a card scheme violation. Soft declines (insufficient funds, issuer unavailable) go into a retry schedule driven by a calibrated XGBoost model that predicts the best timing window. The model uses 12 features including attempt number, time of day, payday proximity, and whether the customer has flagged financial hardship.

When a customer replies to an outreach message, their text is classified by a sentence-transformer model using contrastive scoring against hardship and neutral anchor banks. The classifier outputs three tiers confirmed hardship, uncertain, and no signal. Both confirmed and uncertain cases go to the Human Review Queue. Uncertain cases are never forced into a binary decision, because a missed hardship is more expensive than an unnecessary human review.

A Thompson Sampling bandit continuously adjusts retry timing across all domains based on real outcomes. Under non-stationary conditions (where the best retry window shifts over time), the drift-aware variant significantly outperforms a static schedule (p = 0.046 across 7 independent trials).

**Checkout abandonment**

The module only fires on sessions that reached the checkout page add-to-cart abandonment has a different profile and chasing it recovers less than it costs. Outreach is blocked without DPDP marketing consent. Recovery is confirmed only on payment completion via a Razorpay payment link, not on open or click.

**B2B receivables**

Overdue invoices are tracked against the MSMEDA Section 43B(h) 45-day statutory deadline, which creates a real tax-deductibility incentive for debtors to settle. The escalation path is email, then SMS, then a Twilio voice call in Hindi/Hinglish. If a debtor flags a dispute, automated outreach stops immediately and the case goes to human review collection never continues on a contested invoice.

**UPI AutoPay and NACH mandates**

NPCI caps UPI AutoPay retries at 4 total attempts. Transactions above the RBI ₹15,000 AFA threshold cannot be auto-debited and require UPI PIN re-authentication from the customer. NACH Return 8 (account closed) cancels the mandate immediately continuing to present on a closed account generates bank fees. All of this is enforced at the code level, not in documentation.

---

## ML Results

All models are trained on entity-level splits no customer appears in both training and test data.

**Subscription timing oracle**

| Model | Test AUC | Gap to Bayes Ceiling |
|:---|:---:|:---:|
| Bayes Ceiling | 0.7035 | — |
| GBM (deployed) | 0.7002 | 0.0033 |
| LSTM | 0.6982 | 0.0053 |
| MLP | 0.6920 | 0.0115 |

The LSTM was given the full chronological retry sequence; GBM only had the flat 12-feature vector per attempt. GBM still won. The reason is that the data-generating function depends on prior attempts only through `attempt_number`, which is already in the flat feature set so the sequence adds nothing once that scalar is present. The Bayes ceiling is computed directly from the generating function, not estimated from a model.

**B2B receivables**

| Model | Test AUC | Gap to Ceiling (0.8428) |
|:---|:---:|:---:|
| GBM | 0.8365 | 0.0063 |
| Baseline | 0.8312 | 0.0115 |
| MLP | 0.8297 | 0.0131 |

**Hardship classification**

The classifier scores `H - N`, where H is max cosine similarity to hardship anchor sentences and N is max cosine similarity to neutral billing inquiry sentences. The gap between the lowest genuine hardship score (+0.30) and the highest neutral score (-0.31) is 0.61, so both classification thresholds have substantial margin on either side.

---

## Stack

Backend: Python, FastAPI, Pydantic, Celery, Redis

Storage: PostgreSQL with an append-only event log

ML: XGBoost, Scikit-Learn, PyTorch, Sentence-Transformers (all-MiniLM-L6-v2, runs fully offline)

Frontend: React 18, TypeScript, Vite, Tailwind CSS

Integrations: Razorpay (payment links and HMAC-SHA256 signed webhooks), Twilio (voice IVR and SMS), Resend (email)

---

## Tests

```
python -m pytest -q
291 passed in 21.15s
```

32 test files covering all four modules, orchestrator integration, bandit observer wiring, cross-domain learning core isolation, hardship anchor feedback loops, ML evaluation pipelines, and generator statistical properties.

---

## Running It

```powershell
# API
$env:REVENIO_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/revenio"
$env:REVENIO_REDIS_URL="redis://localhost:6379/0"
python -m uvicorn backend.api.app:app --reload --port 8000

# Frontend
cd frontend
npm run dev
```

```bash
# ML
python -m backend.ml.train_subscription_model
python -m backend.ml.compare_all
python -m backend.ml.bandit_simulation
python -m backend.ml.oracle
```

For live delivery (email, SMS, voice, payment links), set `REVENIO_CHANNEL_MODE=live` and `REVENIO_LIVE_DELIVERY_ACK=I_HAVE_CONSENT` along with the relevant provider credentials. Without them, every delivery action logs a `delivery_blocked` event and nothing is sent.

---

## Structure

```
backend/
  core/
    orchestrator.py       the loop, stop-gate, audit log, human review routing
    contract.py           shared dataclasses
    events.py             append-only event store
    learning_core.py      Thompson Sampling bandit (static, stationary, drift-aware)
    bandit_observer.py    single-writer observer feeding outcomes to the bandit
  modules/
    subscription/
    checkout_abandonment/
    b2b_receivables/
    mandate_retry/
  ml/
    features.py                    12-feature Schema v3 vector, shared between training and inference
    text_signals.py                hardship classifier (contrastive embedding, three-tier output, anchor growth)
    train_subscription_model.py
    compare_all.py
    bandit_simulation.py
  data/
    *_generator.py                 synthetic data generators calibrated against published sources
    splitting.py                   entity-level splitting
  api/
    app.py

frontend/
  src/
    pages/
    components/

tests/
  core/
  modules/
  integration/
  ml/
  data/
```

---

Python 3.14.2 · `fastapi` · `celery` · `redis` · `xgboost` · `torch` · `scikit-learn` · `sentence-transformers` · `joblib` · `numpy` · `scipy` · `pandas` · `pytest`
