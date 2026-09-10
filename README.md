# Revenio

**Demo:** https://youtu.be/KPbci4vFo6I

Razorpay Hackathon — Track 3, AI Revenue Recovery

> "Build an agent that detects revenue at risk, determines the right intervention, and executes a bounded recovery workflow: from payment failures and checkout abandonment to overdue receivables."

---

## What This Is

Revenio is a modular, auditable AI recovery agent. It diagnoses why revenue was lost, decides the right intervention, executes it within documented compliance bounds, and learns from outcomes. Every decision at every stage is logged.

**Three architectural principles:**
- A shared five-method contract every domain module implements
- An orchestrator that owns the loop, stop-gate, and audit trail — containing zero domain logic
- Independent domain modules that can be tested and run in isolation before being plugged in

---

## Build Order and Status (1-8)

### 1 — Orchestrator Skeleton [DONE]

Full orchestrator loop wired to a stub module. No ML, no domain logic.

- `check_stop` enforced before every `decide` — not skippable by any module
- Human-review gate: `requires_human_review=True` routes to a review queue, not auto-execute
- Circuit breaker: kills a module that never stops itself after a configurable cap
- Append-only audit log: every stage of every case written in call order
- Event-sourced state: the log is the source of truth; state is derived by replaying it

### 2 — Subscription Module, Rule-Based [DONE]

Decline-code diagnosis and retry policy. Baseline before ML.

- ISO 8583 / Visa decline-code taxonomy: soft (retry-eligible), hard (Visa Category 1, never retry), stop-instruction (authorization revoked)
- Hard codes fire `COMPLIANCE_LIMIT`; stop-instruction codes fire `OPT_OUT` — real network rules
- Exponential backoff schedule: 1h, 6h, 24h, 72h; `MAX_RETRY_ATTEMPTS = 15`
- ML bundle integration: loads `subscription_winner.joblib` at startup; falls back to rule-based confidence if missing; schema guard validates feature names at load time

### 3 — Checkout Abandonment Module, Rule-Based [DONE]

Session-behavioral-event diagnosis. No decline code exists — the event is a dropped session.

- Signal taxonomy from Baymard Institute's 50-study meta-analysis: recoverable signals (shipping cost surprise, forced account creation, payment friction) vs non-recoverable (`low_purchase_intent`)
- Module only fires on sessions that reached checkout — add-to-cart abandonment is not a recoverable event
- Consent gate: `check_stop` refuses without explicit marketing opt-in; enforced in `execute()` as well
- `MAX_NUDGES = 3` (documented judgment call — no authoritative source like Visa's retry cap exists)
- Channel escalation: email -> SMS -> human review

### 4 — Grounded Synthetic Data [DONE]

Data calibrated against real published taxonomies, not invented.

**Subscription generator:**
- Decline-code distribution: code-51 at 45%, code-05 at 12% (anchored to published 40.5% / 7.5% figures)
- Per-code base recovery rates within the published 60-70% aggregate for recoverable soft declines
- Code-51 amount-dependence: logistic decay centered on dataset median `exp(5.5) = 244.69`, computed analytically
- Night penalty, payday boost, attempt decay — all sourced in direction, magnitudes estimated

**Checkout-abandonment generator:** recovery rates per signal type from Baymard and 2026 industry benchmarks.

**Entity-level splitting:** no customer appears in more than one split — prevents data leakage.

**Scale:** 5,000 subscription customers (~8,000+ soft-decline records); 8,000 abandonment customers (~2,900+ recoverable records). Enforced by a test that fails loudly if reduced.

### 5 — Subscription Diagnosis-Layer Model Comparison [DONE]

Baseline vs GBM vs NN vs LSTM, evaluated on the same held-out entity-level split.

#### Unified Results (`python -m backend.ml.compare_all`)

| Model | Val AUC | Test AUC | Brier | Gap to Oracle |
|:---|:---:|:---:|:---:|:---:|
| Oracle ceiling (Bayes) | — | **0.7035** | — | — |
| **GBM** (winner) | 0.7403 | **0.7002** | 0.2161 | **+0.0033** |
| LSTM | 0.7454 | 0.6982 | 0.2173 | +0.0053 |
| MLP | 0.7261 | 0.6920 | 0.2181 | +0.0115 |

**GBM wins.** Total spread across all three models: 0.0082 AUC — within run-to-run variance. GBM is within 0.0033 of the Bayes ceiling; the data is the constraint, not the model.

**LSTM finding:** LSTM received the full retry sequence; GBM received only the flat per-attempt vector. GBM still won by 0.002. Reason: the generator's recovery probability depends on prior attempts only through `attempt_number`, a scalar already in the flat feature vector. Sequence order adds zero marginal signal once that scalar is present.

**Calibration:** Sigmoid (Platt) vs isotonic — sigmoid produced lower Brier (0.2186 vs 0.2193) at ~1,235 row scale. Sigmoid kept.

**Cross-distribution generalization:** GBM trained on regime A, evaluated on shifted regime B: AUC drops 0.693 -> 0.620 (-0.073). Baseline drops 0.605 -> 0.527 (-0.078). GBM degrades less — empirically more generalizable.

**Enriched flat parity:** A flat GBM given `customer_recent_failure_pressure` tracks its oracle ceiling within 0.0055 — matching the LSTM's 0.0049 gap. Simpler, lower-latency model deployed to production.

**Production bundle (Schema v3, 12 features):**
```
code_51, code_05, code_91, code_96, code_65, code_61,
attempt_number, is_night, is_near_payday, amount,
customer_recent_failure_pressure, hardship_signal_detected
```

### 5b — Hardship Signal Extraction (Schema v3) [DONE]

Unstructured customer emails converted to a structured signal upstream — not raw text fed into GBM.

**Extractor options (swappable via constructor injection):**

| Extractor | Latency | Cost | Dependency |
|:---|:---|:---|:---|
| `extract_hardship_signal_embedding` (default) | ~10ms | Free | `sentence-transformers` (offline) |
| `extract_hardship_signal` | ~0us | Free | None (keyword fallback) |
| `extract_hardship_signal_llm` | ~500ms | Per-call | API key (explicit opt-in) |

**Contrastive scoring** (`H = max similarity to hardship anchors`, `N = max similarity to neutral anchors`, `score = H - N`) prevents false positives: billing inquiries score H=0.43, N=0.90, giving H-N=-0.47 (correctly rejected). Genuine hardship scores H-N >= +0.30.

**Three-tier confidence output:**
```
H-N > 0.25         ->  tier="high"      -> ESCALATE (confirmed hardship)
0.05 < H-N <= 0.25 ->  tier="uncertain" -> ESCALATE (human decides)
H-N <= 0.05        ->  tier="none"      -> continue normal retry flow
```

**Feedback loop:** when a human confirms an `uncertain`-tier case, `add_confirmed_hardship_anchor(email_text)` grows the anchor bank so future similar phrasing is caught at `high` confidence directly. `add_confirmed_neutral_anchor` reduces false-positive escalations over time.

### 6 — Learning Core & Bandit Policies [DONE]

Drift-aware contextual bandit over discrete action spaces, single-writer observer, and human review feedback loop.

- `StaticHeuristicPolicy`: fixed baseline, never learns
- `StationaryThompsonSampling`: standard Beta-Bernoulli, accumulates uniform history
- `DriftAwareThompsonSampling`: discounted (gamma) or sliding-window, adaptively forgets stale outcomes
- `LearningCore`: manages one policy per domain; cross-domain independence guaranteed
- `BanditUpdateObserver`: subscribes to `EventStore` via `EventObserver` protocol; applies single-writer updates on terminal `Outcome` events

**Benchmark — multi-trial summary (7 seeds, paired t-tests):**

| Policy | Post-shift mean recovery rate | vs static |
|:---|:---:|:---|
| Static | 0.258 | — |
| Stationary TS | 0.299 | p=0.127 (not significant) |
| **Drift-aware TS** | **0.326** | **p=0.046 (significant)** |

Domain pooling: subscription + abandonment + mandate_retry under one shared `LearningCore` recover $505,890 aggregate with independent policy spaces and zero arm-pull bleed across domains.

### 7 — B2B Receivables Module [DONE]

Overdue invoice recovery with statutory compliance and promise-to-pay lifecycle.

- **Section 43B(h) / MSMED Act**: tracks 45-day (written agreement) vs 15-day (no agreement) statutory deadlines for registered MSMEs
- **Channel escalation**: email -> SMS -> voice (Hinglish hi-IN locale, configurable per customer)
- **Disputed invoice protection**: `is_disputed=True` fires `StopReason.COST_THRESHOLD` and halts all automated outreach immediately
- **Promise-to-pay lifecycle**: `on_promise_due` pauses contact; broken promises after `MAX_BROKEN_PROMISES = 2` trigger `StopReason.DIMINISHING_RETURNS`
- **DND / NCPR consent**: double-enforced at `check_stop` and `execute`; fails closed

**B2B model comparison:**

| Model | Test AUC | Gap to Oracle (0.8428) |
|:---|:---:|:---:|
| GBM | 0.8365 | +0.0063 |
| Baseline | 0.8312 | +0.0115 |
| MLP | 0.8297 | +0.0131 |

Oracle AUC 0.8428 vs subscription's 0.7035: structurally simpler generating function (aging-bucket monotonic decay + MSME flag). GBM gap of 0.0063 means it is essentially learning the generating function.

### 8 — Mandate Retry Sequencer (UPI AutoPay & NACH) [DONE]

Fourth domain, expanding recovery to recurring UPI and bank debit mandates under Indian network rules.

**UPI Autopay (NPCI 2026 rules):**
- RBI AFA threshold: amounts > 15,000 INR switch to `push_notification` for manual UPI-PIN re-auth — enforced in both `decide()` and `execute()`
- NPCI ceiling: 1 main attempt + 3 retries = 4 total (`StopReason.COMPLIANCE_LIMIT`)
- Taxonomy: soft failures (U01 insufficient funds, U02-U04 system transients) vs stop codes (revoked/paused/expired -> `StopReason.OPT_OUT`)

**NACH (RBI ECS Debit Guidelines):**
- Return codes 1, 2, 3: require data correction before re-presentation (`requires_human_review=True`)
- Return code 8 (mandate not received): immediate halt (`StopReason.OPT_OUT`)
- Max 3 re-presentations (`MAX_NACH_PRESENTATIONS = 3`)

Optional 3-arm Thompson sampling bandit over `UPI_RETRY_BACKOFF_HOURS = [24, 72, 168]` — scoped to UPI; NACH runs fixed 24h cadence. Four-domain pooling test proves zero arm-pull bleed across all domains under one shared `LearningCore`.

---

## Test Suite

**291 tests, 32 files, all passing.**

```
python -m pytest -q
291 passed in 21.15s
```

| Area | Tests |
|:---|:---:|
| Core (orchestrator, events, learning core, case history) | 37 |
| Data generators & splitting | 45 |
| ML (oracle, baseline, calibration, features, text signals, b2b compare) | 37 |
| Modules (subscription, abandonment, b2b, mandate retry, hardship, diminishing returns) | 90 |
| Integration (bandit observer, anchor feedback, cross-domain pooling) | 57 |
| Misc (queue pipeline, orchestrator against postgres) | 25 |

---

## Architecture

### Shared Contract

Every domain module implements the same five-method interface:

| Method | Purpose |
|:---|:---|
| `check_stop(case, history)` | Called before every cycle — not skippable; returns `StopDecision` |
| `diagnose(case, customer_history)` | Domain-owned interpretation; returns `Diagnosis` with root cause, recoverability, confidence, optional ML prediction |
| `decide(case, diagnosis, history)` | Policy decision; returns `Decision` with action type, params, reasoning |
| `execute(case, decision)` | Takes the action; module self-certifies its own compliance check |
| `track_outcome(case)` | Ground-truth feedback; returns `Outcome` |

**Action types:** `RETRY`, `SWITCH_CHANNEL`, `ESCALATE`, `WAIT`, `STOP`
**Stop reasons:** `COMPLIANCE_LIMIT`, `OPT_OUT`, `DIMINISHING_RETURNS`, `COST_THRESHOLD`, `RESOLVED`
**Outcome statuses:** `RECOVERED`, `PROMISED`, `LOST`, `PENDING`

### Orchestrator

`backend/core/orchestrator.py` owns the loop, stop-gate, and audit trail. Contains zero domain-specific logic.

- Calls `check_stop` before every cycle — modules cannot skip it
- Queries `EventStore.get_customer_case_history()` and passes raw prior events to `diagnose()` without interpretation
- Routes `requires_human_review=True` cases to the review queue; does not call `execute`
- Circuit breaker caps runaway modules

### Event Sourcing

`backend/core/events.py` — append-only event log. State is derived by replaying it; no dual-write synchronization problem. Cross-case customer history queried via `get_customer_case_history(customer_id, exclude_case_id)`.

### Feature Construction

`backend/ml/features.py` — single source of truth for the 12-feature Schema v3 vector. Both the trainer and inference path import from here, preventing train/serve skew by construction.

### Hardship Signal Extraction

`backend/ml/text_signals.py` — contrastive embedding scoring upstream of GBM. The decision model never sees raw text. Key constants: `_CONTRASTIVE_MARGIN = 0.25`, `_CONTRASTIVE_UNCERTAIN_FLOOR = 0.05`. Gap between hardship floor (+0.30) and neutral ceiling (-0.31): 0.61 — both thresholds sit comfortably inside that gap.

---

## Project Structure

```
backend/
  core/
    contract.py           shared Diagnosis, Decision, Outcome, StopDecision dataclasses
    events.py             append-only event store with EventObserver protocol
    orchestrator.py       loop, stop-gate, audit trail, submit_human_review
    learning_core.py      static, stationary, and drift-aware Thompson Sampling
    bandit_observer.py    single-writer event observer feeding outcomes to learning core
  data/
    subscription_generator.py          grounded synthetic subscription records & retry sequences
    checkout_abandonment_generator.py  grounded synthetic abandonment records
    b2b_generator.py                   grounded synthetic B2B invoice records (AR aging curve)
    mandate_retry_generator.py         grounded synthetic UPI/NACH mandate records
    splitting.py                       entity-level train/val/test splitting
  ml/
    features.py             canonical flat & enriched feature construction (one source of truth)
    sequence_features.py    sequence per-stage feature construction
    text_signals.py         hardship extraction: contrastive embedding, keyword, LLM, anchor growth
    compare.py              flat comparison: baseline vs GBM vs NN (10 features)
    compare_sequence.py     sequence model: LSTM vs chain oracle
    compare_with_history.py flat models with customer history parity (11 features)
    compare_all.py          UNIFIED: GBM vs MLP vs LSTM, same split, schema v3
    compare_b2b.py          B2B model comparison
    bandit_simulation.py    drift & pooling benchmark over observer-driven pipeline
    train_subscription_model.py  produces subscription_winner.joblib (Schema v3)
    oracle.py               flat oracle AUC ceiling computation
    calibration.py          calibration evaluation (Platt/sigmoid)
    models/
      baseline.py           rule-based lookup baseline
      gbm.py                XGBoost search and training
      neural_net.py         PyTorch MLP search and training
      sequence.py           PyTorch LSTM sequence model
      subscription_winner.joblib          deployed bundle (generated)
      subscription_winner_metrics.json    human-readable audit copy (generated)
  modules/
    dummy/module.py                stub for orchestrator testing
    subscription/module.py         subscription recovery (cross-case memory, hardship, bandit backoff)
    checkout_abandonment/module.py checkout session recovery (bandit channel selection)
    b2b_receivables/module.py      B2B receivables (43B(h), DND, promise tracking, voice)
    mandate_retry/module.py        UPI/NACH mandate recovery (AFA threshold, NPCI/NACH rules)

tests/
  core/        orchestrator, events, learning core, customer case history
  data/        generators, splitting, retry sequences, causal pressure, hardship signal
  integration/ bandit observer wiring, anchor feedback loop, cross-domain pooling
  ml/          oracle, baseline, calibration, features, text signals, b2b compare
  modules/     per-module unit tests, hardship policy, diminishing returns
```

---

## How to Run

```bash
# All tests
python -m pytest -q

# Unified model comparison (GBM vs MLP vs LSTM, same split, schema v3)
python -m backend.ml.compare_all

# B2B model comparison
python -m backend.ml.compare_b2b

# Flat comparison (10 features)
python -m backend.ml.compare

# Sequence model comparison (LSTM only)
python -m backend.ml.compare_sequence

# Enriched flat comparison (11 features with customer history)
python -m backend.ml.compare_with_history

# Recompute oracle ceiling
python -m backend.ml.oracle

# Train and save production bundle (Schema v3, 12 features)
python -m backend.ml.train_subscription_model

# Bandit drift & pooling benchmark
python -m backend.ml.bandit_simulation
```

## Run the Merchant Workbench

```powershell
# Terminal 1 — API
$env:REVENIO_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/revenio"
$env:REVENIO_REDIS_URL="redis://localhost:6379/0"
python -m uvicorn backend.api.app:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

**Delivery modes:** Default `sandbox` adapter never contacts a real address. `live` mode requires:
- Email: `RESEND_API_KEY`, `REVENIO_EMAIL_FROM`
- SMS/Voice: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`
- Razorpay: `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`, `RAZORPAY_WEBHOOK_SECRET`
- `REVENIO_CHANNEL_MODE=live`, `REVENIO_LIVE_DELIVERY_ACK=I_HAVE_CONSENT`

Missing config yields an auditable `delivery_blocked` event — it never silently claims a message occurred.

---

## Key Design Decisions

| Decision | What was done | Why |
|:---|:---|:---|
| GBM over LSTM for production | GBM wins by 0.002 AUC on sequence data | Generator recovery probability depends on `attempt_number` (scalar already in flat vector). Sequence order adds zero marginal signal. |
| Sigmoid over isotonic calibration | Brier 0.2186 vs 0.2193 | Isotonic needs more calibration data; 2-parameter sigmoid fit is more reliable at ~1,235 row scale. |
| Contrastive not single-anchor scoring | Prevents false positives on billing inquiries | Billing inquiries (H=0.43, N=0.90) correctly rejected at H-N=-0.47. |
| Structured signal not raw text in GBM | `bool` + `enum` fed into flat pipeline | Keeps GBM as the decision layer; only feature-extraction phase changes when upgrading the extractor. |
| No Bayesian hyperparameter search | Random search adopted | GBM already within 0.003 of oracle ceiling; smarter search has no headroom left to find. |
| No scale-up to 50k+ records | Current scale kept | More data tightens estimate around ceiling but does not raise it. Ceiling is a property of feature informativeness. |

---

## Open Items (Documented, Not Silently Deferred)

- Checkout nudge cap `MAX_NUDGES = 3` is a judgment call — no authoritative source equivalent to Visa's retry cap exists for abandonment. Flagged in module source.
- `checkout_abandonment.diagnose()` accepts `customer_history` (contract-required) but does not use it — no cross-case behavioral signal built for this domain. Documented scope decision.
- Do not lower `_CONTRASTIVE_UNCERTAIN_FLOOR` below 0.0 without re-running the probe script; the boundary between hardship and neutral is 0.61 wide but the floor must stay inside it.
- Promise-to-pay cadence (`MAX_BROKEN_PROMISES = 2`) and exact `requires_human_review` confidence thresholds per domain are judgment calls, not sourced values.

---

## Technical Notes

**Python:** 3.14.2

**Key dependencies:** `xgboost`, `torch`, `scikit-learn`, `sentence-transformers` (all-MiniLM-L6-v2, ~80MB, downloads once then fully offline), `joblib`, `numpy`, `scipy`, `pandas`, `pytest`

**Windows:** All paths use forward slashes internally. Run `python -m backend.*` from the project root so it is on `sys.path`.
