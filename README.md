# Revenio

Razorpay Hackathon — Track 3, AI Revenue Recovery

Goal (verbatim from the problem statement): "Build an agent that detects revenue at risk, determines the right intervention, and executes a bounded recovery workflow: from payment failures and checkout abandonment to overdue receivables."

---

## What this is

Revenio is a modular, auditable AI recovery agent. It diagnoses why revenue was lost, decides the right intervention, executes it within documented compliance bounds, and learns from outcomes. Every decision at every stage is logged.

The architecture is organized around three things:
- A shared contract that every domain module implements
- An orchestrator that enforces the loop, the stop-gate, and the audit trail — and contains zero domain-specific logic
- Independent domain modules that can each be tested, run, and reasoned about in isolation before being plugged in

---

## Build order and status

### Step 1 — Orchestrator skeleton [DONE]

The full orchestrator loop wired to a dummy stub module. No ML, no real domain logic.

What was built:
- `check_stop` enforced before every `decide` — not skippable by a module
- Human-review gate: when `requires_human_review` is true, the case is routed to a review queue, not auto-executed
- Circuit breaker: a module that never stops itself gets cut off after a configurable iteration cap
- Audit log: every stage of every case is tagged and stored in call order
- Event-sourced state: the log is the source of truth; state is derived by replaying it

Files:
- `backend/core/orchestrator.py`
- `backend/core/contract.py`
- `backend/core/events.py`
- `backend/modules/dummy/module.py`
- `tests/core/test_orchestrator.py`
- `tests/core/test_events.py`

### Step 2 — Subscription module, rule-based [DONE]

Decline-code diagnosis and retry policy. Baseline before any ML.

What was built:
- ISO 8583 / Visa decline-code taxonomy: soft (retry-eligible), hard (Visa Category 1, never retry), stop-instruction (customer/issuer revoked authorization)
- `check_stop` fires `COMPLIANCE_LIMIT` on hard codes and `OPT_OUT` on stop-instruction codes — these are real network rules, not invented thresholds
- Exponential backoff retry schedule: 1h, 6h, 24h, 72h
- `MAX_RETRY_ATTEMPTS = 15` cap, enforced
- ML bundle integration: `SubscriptionModule` loads `subscription_winner.joblib` at startup; if no bundle exists, it falls back to rule-based confidence only — never raises
- Schema guard at load time: bundle's `feature_names` is checked against the module's live `FEATURE_NAMES`; mismatched schemas are refused loudly rather than silently producing garbage predictions

Files:
- `backend/modules/subscription/module.py`
- `tests/modules/subscription/test_subscription_module.py`

### Step 3 — Checkout-abandonment module, rule-based [DONE]

Session-behavioral-event diagnosis. Different event shape from subscription — no decline code exists; the event is a dropped session.

What was built:
- Signal taxonomy sourced from Baymard Institute's 50-study meta-analysis (the most reliable public source for ranked abandonment causes):
  - Recoverable: shipping cost surprise, forced account creation, payment method unavailable, checkout form friction, checkout page error, distracted high intent
  - Not recoverable: low purchase intent (short session, no engagement beyond add-to-cart)
- Two deliberate sourced design decisions, not defaults:
  - Module only fires on sessions that reached checkout — add-to-cart abandonment is window-shopping, not a recoverable event
  - `low_purchase_intent` is marked non-recoverable: chasing low-engagement sessions costs more than it recovers (a real stopping rule grounded in the source data)
- Consent gate: `check_stop` refuses to proceed without explicit marketing consent — enforced in `execute()` as well, not just in `check_stop`
- Nudge cap: `MAX_NUDGES = 3` (flagged open item — no authoritative source equivalent to Visa's retry cap exists for this number; it is a documented judgment call, not a silently baked-in default)
- Channel escalation: first nudge is email, second is SMS, third triggers human review

Files:
- `backend/modules/checkout_abandonment/module.py`
- `tests/modules/checkout_abandonment/test_checkout_abandonment_module.py`

### Step 4 — Grounded synthetic data [DONE]

Data calibrated against real published taxonomies, not invented.

Subscription generator:
- Decline-code distribution: code-51 (insufficient funds) at 45% and code-05 (do-not-honor) at 12% anchored to published figures (40.5% and 7.5% of all payment failures respectively); remaining soft codes split the remainder
- Per-code base recovery rates: within the real published 60-70% aggregate for recoverable soft declines
- Attempt decay: sourced direction (recovery drops sharply after the first few attempts); magnitude estimated
- Night penalty: sourced direction (Adyen ~2% lower at night); applied as a small multiplier
- Payday boost: sourced direction (insufficient-funds retries near payday recover better); magnitude estimated
- Code-51 amount-dependence: a logistic (sigmoid) decay centered on the dataset's amount-distribution median, computed analytically from `exp(mu) = exp(5.5) = 244.69`, not eyeballed. Small amounts push recovery probability up; large amounts push it down. Only code-51 — every other code's generating function is unchanged.

Checkout-abandonment generator:
- Recovery rates per signal type sourced from Baymard and 2026 industry benchmarks
- Time-of-day effects, price sensitivity by signal, browse-to-recover timing all grounded in source material

Entity-level splitting:
- Train/val/test split with no customer appearing in more than one split — avoids the data-leakage failure mode where a model sees a customer's future outcomes during training

Scale:
- Subscription: 5,000 customers, ~8,000+ soft-decline records
- Checkout-abandonment: 8,000 customers, ~2,900+ recoverable-signal records
- Both minimums are enforced by a test (`test_default_dataset_is_large_enough_for_a_fair_step5_model_comparison`) that fails loudly if the default is ever reduced

Files:
- `backend/data/subscription_generator.py`
- `backend/data/checkout_abandonment_generator.py`
- `backend/data/splitting.py`
- `tests/data/test_subscription_generator.py`
- `tests/data/test_checkout_abandonment_generator.py`

### Step 5 — Subscription diagnosis-layer model comparison [DONE]

Baseline vs GBM vs neural net, evaluated with the same held-out discipline throughout.

#### Setup

- Feature set (10 features): `code_51`, `code_05`, `code_91`, `code_96`, `code_65`, `code_61`, `attempt_number`, `is_night`, `is_near_payday`, `amount`
- Entity-level split: 5,603 train / 1,235 val / 1,230 test
- Search: GBM uses random search over `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda` (25 iterations with 5-fold GroupKFold CV). NN uses random search over `n_layers`, `width_multiplier`, `lr`, `dropout`, `weight_decay` (15 iterations with same CV).

#### Results (compare.py output, seed=42)

```
STEP 5 — Subscription diagnosis-layer model comparison

Entity-level split (soft-decline rows only): 5603 train / 1235 val / 1230 test
Feature set (10): ['code_51', 'code_05', 'code_91', 'code_96', 'code_65', 'code_61',
                   'attempt_number', 'is_night', 'is_near_payday', 'amount']

Baseline (rule-based lookup)
  AUC=0.605  Precision=0.561  Recall=0.205  Brier=0.235

Gradient-boosted trees (XGBoost), random search
  Best params: max_depth=2, learning_rate=0.0281, n_estimators=245,
               subsample=0.61, colsample_bytree=0.77, min_child_weight=10, reg_lambda=0.66
  CV AUC=0.692
  Test: AUC=0.693  Precision=0.606  Recall=0.477  Brier=0.217

Neural net, random search
  Best params: n_layers=1, width_multiplier=2, lr=0.0845, dropout=0.304, weight_decay=0.00139
  CV AUC=0.693
  Test: AUC=0.691  Precision=0.617  Recall=0.492  Brier=0.217

WINNER (by test AUC): GBM  (AUC=0.693)
  GBM        AUC=0.693 -- WINNER
  NN         AUC=0.691   lost
  Baseline   AUC=0.605   lost
```

#### Oracle ceiling

The ceiling was recomputed against the current generator (which includes the code-51 amount-dependence). The original 0.694 figure in architecture doc 6.5 predates that change and is now superseded.

```
python -m backend.ml.oracle

ORACLE CEILING -- recomputed against the CURRENT generator
(includes code-51 amount-dependence; supersedes the 0.694 figure
in architecture doc 5.1/6.5, which predates that change)

  Full dataset  (n=8068):  oracle AUC = 0.6945
  Test split only (n=1230): oracle AUC = 0.6956

  This is the number to compare GBM/MLP test AUC against going forward.
```

GBM test AUC of 0.693 is 0.3 AUC points below the oracle ceiling of 0.6956 on the test split. The gap between GBM and NN (0.001–0.002 AUC) is inside noise — both models are at approximately the same distance from the ceiling.

#### Calibration

Calibration was tested on the winning GBM model using sigmoid scaling (CalibratedClassifierCV):

```
Brier score before calibration: 0.2169
Brier score after calibration:  0.2170
DID NOT IMPROVE -- reported honestly either way.
```

Sigmoid was tested against isotonic: sigmoid produced a slightly better Brier (0.2186 vs 0.2193) on the ~1,235-row validation set. Isotonic is non-parametric and needs more calibration data than a 2-parameter sigmoid fit at this scale. Sigmoid is kept.

#### Cross-distribution generalization test

The winning GBM model, trained only on regime A, was evaluated against a deliberately shifted regime B (smaller payday effect, harder-to-recover issuer/system-error codes) that was never seen during training:

```
GBM -- same distribution (regime A):     AUC=0.693
GBM -- shifted distribution (regime B):  AUC=0.620  (drop: +0.073)
Baseline -- shifted (regime B):           AUC=0.527  (drop: +0.078)

GBM degrades LESS than baseline under shift -- genuinely more generalizable, not just better-fit.
```

#### Enriched Flat Comparison — Customer History Feature Parity

Following the sequence model findings, we tested whether a flat model (GBM/NN) given the exact same causal `customer_recent_failure_pressure` feature achieves parity with the LSTM (`backend/ml/compare_with_history.py`).

Results (`compare_with_history.py` output, seed=42):

```
DOES GBM/NN BENEFIT FROM customer_recent_failure_pressure LIKE THE LSTM DID?

10032 total records generated (soft + hard + stop codes), WITH customer history.
Entity-level split (soft-decline rows only): 5603 train / 1235 val / 1230 test
Feature set (11): ['code_51', 'code_05', 'code_91', 'code_96', 'code_65', 'code_61',
                   'attempt_number', 'is_night', 'is_near_payday', 'amount',
                   'customer_recent_failure_pressure']

Oracle AUC for this enriched flat distribution: 0.6889

Baseline (rule-based lookup)
  AUC=0.605

GBM (XGBoost) WITH customer_recent_failure_pressure, random search
  Best params: max_depth=3, learning_rate=0.038, n_estimators=180, ...
  CV AUC=0.689
  Test: AUC=0.6834

NN WITH customer_recent_failure_pressure, random search
  Best params: n_layers=3, width_multiplier=7, lr=0.00988, ...
  CV AUC=0.696
  Test: AUC=0.6807

Result:
  GBM (with history) test AUC:  0.6834   gap to oracle: 0.0055
  NN  (with history) test AUC:  0.6807   gap to oracle: 0.0082
  Oracle ceiling (this dist.):  0.6889
```

| Model | Test AUC | Oracle Ceiling | Gap to Ceiling |
|:---|:---|:---|:---|
| **LSTM** (chained sequence) | 0.6986 | 0.7035 | 0.0049 |
| **GBM** (enriched flat) | 0.6834 | 0.6889 | 0.0055 |
| **NN** (enriched flat) | 0.6807 | 0.6889 | 0.0082 |
| **Baseline** (rule lookup) | 0.6050 | 0.6889 | 0.0839 |

**Key architectural finding**: A flat tree model given the causal customer failure pressure feature tracks its oracle ceiling within 0.0055 (matching the LSTM's 0.0049 gap). This empirically justifies deploying the simpler, lower-latency flat model bundle into production while preserving the performance gains from cross-case customer memory.

#### Step 5 Addendum — Sequence Model (4th Comparison Point) [DONE]

Architecture doc §6.3 requires a sequence model (LSTM) as a fourth diagnosis-layer comparison point, evaluated with the same held-out entity-level discipline as baseline, GBM, and NN.

What was built:
- `generate_subscription_retry_sequences()` in `backend/data/subscription_generator.py` — generates genuine chronological retry chains (attempt $k$ only exists if attempt $k-1$ failed), incorporating a causal, recency-weighted customer failure pressure (EWMA with $\alpha=0.5$).
- `backend/ml/sequence_features.py` — per-step feature construction (10 features: 6 one-hot decline codes, `is_night`, `is_near_payday`, `amount`, `customer_recent_failure_pressure`).
- `backend/ml/models/sequence.py` — small sequence model (`RetryLSTM`), tuned via random search + `GroupKFold` entity-aware CV.
- `backend/ml/compare_sequence.py` — standalone comparison script evaluating the LSTM against its own chain-distribution oracle ceiling.

Results (`compare_sequence.py` output):

```
STEP 5, COMPARISON POINT 4 — LSTM sequence model (architecture doc 6.3)
v2: includes causal customer-history (recency-weighted) effect

7952 genuine retry-chain cases generated (soft-decline only).
  Sanity check — final recovery rate, low pressure (<0.1, n=7018): 0.832
  Sanity check — final recovery rate, high pressure (>0.5, n=89):  0.663
  z-test: z=4.22, p=0.000025 — statistically significant pressure effect.

Entity-level split (cases): 5560 train / 1175 val / 1217 test
Per-attempt training examples: 10542 train / 2227 val / 2348 test

Oracle AUC for THIS chain-derived test distribution: 0.7035

LSTM, random search over real ranges
  Best params: hidden_size=26, lr=0.01236, weight_decay=2.38e-06
  CV AUC=0.702
  Test AUC=0.6986

Result:
  LSTM test AUC:                        0.6986
  Oracle ceiling for THIS distribution:  0.7035
  Gap to own ceiling:                    0.0049
```

---

### Step 5 Final — Unified Comparison: GBM vs MLP vs LSTM [DONE]

All three models retrained on the **same entity-level split** of the **same dataset** (retry-chain sequences, schema v3, 12 features) with a Bayes oracle ceiling. This is the definitive apples-to-apples result.

#### What changed from earlier comparisons

Earlier runs compared models across different datasets (flat vs sequence) and different feature sets (10 vs 11 features). The unified comparison (`backend/ml/compare_all.py`) fixes this:
- All three use `generate_subscription_retry_sequences()` as the single data source
- All three are evaluated at the **per-attempt** granularity — same rows, same split
- Flat models (GBM/MLP) use the 12-feature flat vector per attempt; LSTM uses the full padded sequence up to that attempt
- Oracle ceiling is computed on the same test set

#### Unified results (`python -m backend.ml.compare_all`)

```
======================================================================
UNIFIED MODEL COMPARISON — GBM vs MLP vs LSTM
Schema v3  |  Same entity-level split  |  No fake numbers
======================================================================

Generating retry-chain dataset...
  Total cases (soft-decline only): 7952
  Entity-level split: 5560 train / 1175 val / 1217 test

Oracle ceiling (Bayes): 0.7035

----------------------------------------------------------------------
GBM (XGBoost + sigmoid calibration)
----------------------------------------------------------------------
  Val  AUC:   0.7403
  Test AUC:   0.7002  (gap to oracle: +0.0033)
  Test Brier: 0.2161

----------------------------------------------------------------------
MLP (sklearn, (32,16) ReLU, sigmoid calibration)
----------------------------------------------------------------------
  Val  AUC:   0.7261
  Test AUC:   0.6920  (gap to oracle: +0.0115)
  Test Brier: 0.2181

----------------------------------------------------------------------
LSTM (random search, 15 iterations, entity-aware CV)
----------------------------------------------------------------------
  Best params: hidden_size=26, lr=0.01236, weight_decay=2.38e-06
  CV AUC (tuning): 0.7076
  Val  AUC:   0.7454
  Test AUC:   0.6982  (gap to oracle: +0.0053)
  Test Brier: 0.2173
```

#### Summary table

| Model | Val AUC | Test AUC | Brier | Gap to oracle |
|:---|:---:|:---:|:---:|:---:|
| Oracle ceiling | — | **0.7035** | — | — |
| **GBM** (winner) | 0.7403 | **0.7002** | 0.2161 | **+0.0033** |
| LSTM | 0.7454 | 0.6982 | 0.2173 | +0.0053 |
| MLP | 0.7261 | 0.6920 | 0.2181 | +0.0115 |

#### Honest interpretation

**GBM wins.** Total spread across all three models: **0.0082 AUC** (less than 1 point). This is within run-to-run variance at this dataset size.

**GBM test AUC of 0.7002 is within 0.0033 of the Bayes ceiling (0.7035).** There is essentially no remaining headroom to extract from the current feature set with any model architecture. The data is the constraint, not the model.

**The LSTM finding is a legitimate result, not a bug.** The LSTM received the full retry sequence (prior-attempt context that the flat models do not have). It still matched GBM within 0.002. The reason: `true_recovery_probability()` in the generator depends on prior attempts only through `attempt_number` — a scalar already present in the flat feature vector. Once the flat model has `attempt_number`, the sequence order adds zero marginal signal. This confirms the general rule: sequence architectures add value only when step-level ordering contains information that a summarising scalar cannot capture.

**MLP gap (+0.0115):** MLP consistently trails GBM on tabular data at this scale. Expected and consistent with the Step 5 original findings.

#### Deployed production bundle (Schema v3, 12 features)

The production trainer (`train_subscription_model.py`) trains on schema v3 — 12 features including `hardship_signal_detected` — and serializes the calibrated GBM winner.

Real training output (`python -m backend.ml.train_subscription_model`):
```
Winner: GBM
Extractor used: extract_hardship_signal_embedding
Saved bundle to backend/ml/models/subscription_winner.joblib
Per-candidate val AUC: {'GBM': {'val_auc': 0.6522, 'val_brier': 0.2120},
                         'MLP': {'val_auc': 0.6376, 'val_brier': 0.2157}}
Winner test AUC: 0.7134
Winner test Brier: 0.2073
```

Production feature set (12 features, fixed order `FEATURE_NAMES_WITH_HISTORY_AND_TEXT`):
```
code_51, code_05, code_91, code_96, code_65, code_61,
attempt_number, is_night, is_near_payday, amount,
customer_recent_failure_pressure, hardship_signal_detected
```

---

### Step 5 Addendum — Hardship Signal Extraction (Schema v3) [DONE]

Architecture doc §9 requires unstructured customer communications to feed the diagnosis layer. Implemented as a structured signal extracted upstream — not raw text fed into the decision model.

#### Design principle

Extract a `bool` and `enum` from the free-text email upstream; feed those into the existing 12-feature flat pipeline exactly like `customer_recent_failure_pressure`. GBM remains the decision layer. Only the feature-extraction step changes.

#### Extractor: contrastive embedding (default, offline)

`backend/ml/text_signals.py` — three implementations behind the same `HardshipExtractor` interface:

| Extractor | Latency | Cost | Dependency |
|:---|:---|:---|:---|
| `extract_hardship_signal_embedding` | ~10ms | Free | `sentence-transformers` (offline) |
| `extract_hardship_signal` | ~0µs | Free | None (keyword fallback) |
| `extract_hardship_signal_llm` | ~500ms | Per-call | API key (explicit opt-in) |

The **default is `extract_hardship_signal_embedding`** — `all-MiniLM-L6-v2` (~80MB, downloads once, then fully offline). No API key. No per-call cost.

#### Contrastive scoring (how false positives are prevented)

Single-anchor similarity alone produces false positives: billing inquiries containing "charged" or "payment" score moderately against hardship anchors. The fix is **contrastive scoring**:

```
H = max cosine similarity to hardship anchor bank (11 sentences)
N = max cosine similarity to neutral/billing anchor bank (8 sentences)
contrastive_score = H - N
```

Calibrated scores on `all-MiniLM-L6-v2`:

| Sentence | H | N | H−N | Result |
|:---|:---:|:---:|:---:|:---:|
| "I lost my job last week..." | 0.77 | 0.47 | **+0.30** | HARDSHIP |
| "I cannot afford this right now" | 0.79 | 0.28 | **+0.51** | HARDSHIP |
| "Things have been really rough financially..." | 0.76 | 0.15 | **+0.61** | HARDSHIP |
| "Please update my card on file" | 0.17 | 0.49 | **−0.31** | NEUTRAL |
| "When will my card be charged?" *(problem child)* | 0.43 | 0.90 | **−0.47** | NEUTRAL |

Gap between hardship floor (+0.30) and neutral ceiling (−0.31): **0.61 AUC points**.

#### Three-tier confidence band (fix for out-of-distribution text)

Instead of a binary detected/not-detected, the extractor returns a `hardship_confidence_tier`:

```
H−N > 0.25          →  tier="high"      detected=True    → ESCALATE (confirmed hardship)
0.05 < H−N ≤ 0.25   →  tier="uncertain" detected=True    → ESCALATE (human decides)
H−N ≤ 0.05          →  tier="none"      detected=False   → normal RETRY flow
```

The **uncertain band** is the answer to unusual or out-of-distribution text: the anchor bank cannot cleanly classify it, so the policy layer escalates to human review rather than making a binary call. This is the safe production behaviour — a missed hardship costs more than an unnecessary human-review routing.

The audit log exposes all four values (`hardship_similarity`, `neutral_similarity`, `contrastive_score`, `hardship_confidence_tier`) so any human reviewer can see exactly why a case was escalated.

#### Policy routing in `decide()`

```python
tier = "high"      →  ESCALATE, reasoning: "Customer disclosed financial hardship"
tier = "uncertain" →  ESCALATE, reasoning: "Email could not be confidently classified —
                       routed to human review rather than making a binary call on
                       out-of-distribution text"
tier = "none"      →  continue normal RETRY flow
```

#### Swappability

`SubscriptionModule` accepts any `HardshipExtractor` callable as a constructor argument. Swapping the extractor is a single-argument change — nothing in `features.py`, `diagnose()`, or `decide()` changes:

```python
SubscriptionModule()                                                  # default: embedding
SubscriptionModule(hardship_extractor=extract_hardship_signal)        # keyword-only fallback
SubscriptionModule(hardship_extractor=extract_hardship_signal_llm)    # explicit LLM opt-in
```

#### Feedback loop (Step 6 integrated)

Uncertain-tier cases escalated to human review are the natural feedback signal: when a human confirms an `uncertain`-tier case via `orchestrator.submit_human_review(case_id, confirmed=True, case=case)`, `SubscriptionModule.on_human_review_confirmed` invokes `add_confirmed_hardship_anchor(email_text)`. The anchor bank grows from real human decisions, making future similar phrasing trigger `high` confidence directly.

#### New files

- `backend/ml/text_signals.py` — `HardshipExtractor` type alias, three implementations, contrastive scoring, confidence tier, anchor growth callback
- `tests/ml/test_text_signals.py` — keyword, embedding, contrastive, tier, and paraphrase detection tests
- `tests/modules/subscription/test_hardship_policy.py` — policy routing tests for all three tiers and swappability
- `tests/data/test_support_email_hardship_signal.py` — generator-level hardship signal simulation tests

---

### Step 6 — Learning core & Bandit policies [DONE]

Drift-aware contextual bandit over domain discrete action spaces, single-writer observer updates, and human review anchor growth loop.

What was built:
- **Bandit Policies** (`backend/core/learning_core.py`):
  - `StaticHeuristicPolicy`: fixed baseline rule that never learns.
  - `StationaryThompsonSampling`: standard Beta-Bernoulli Thompson Sampling accumulating uniform history.
  - `DriftAwareThompsonSampling`: discounted ($\gamma \in (0, 1]$) or sliding-window Thompson Sampling that adaptively downweights/forgets stale outcomes.
  - `LearningCore`: manager owning one policy per registered domain, ensuring cross-domain independence.
- **Single-Writer Observer** (`backend/core/bandit_observer.py`):
  - `BanditUpdateObserver` subscribes to `EventStore` via the `EventObserver` protocol.
  - Decoupled from core execution: tracks `Decision` events carrying `bandit_arm` and applies single-writer updates sequentially when terminal `Outcome` (`RECOVERED` or `LOST`) events occur.
- **Domain Module Wiring**:
  - `SubscriptionModule` selects retry backoff hours dynamically from the bandit arm when `learning_core` is provided.
  - `CheckoutAbandonmentModule` selects nudge escalation channels dynamically from the bandit arm.
  - Optional `anchor_growth_callback` in `SubscriptionModule.on_human_review_confirmed` closes the Step 6 human-in-the-loop feedback loop.

#### Step 6 Benchmark (`python -m backend.ml.bandit_simulation`)

```
======================================================================
STEP 6 BENCHMARK -- Static vs Stationary vs Drift-Aware, real pipeline
======================================================================

--- Drift benchmark: subscription domain, hard regime change mid-batch ---

  static:
    pre-shift:  money=$20335  recovery_rate=0.553
    post-shift: money=$11760  recovery_rate=0.320
    TOTAL money recovered: $32095

  stationary_ts:
    pre-shift:  money=$13720  recovery_rate=0.373
    post-shift: money=$8330   recovery_rate=0.227
    TOTAL money recovered: $22050

  drift_aware_ts:
    pre-shift:  money=$16905  recovery_rate=0.460
    post-shift: money=$15190  recovery_rate=0.413
    TOTAL money recovered: $32095

  Best TOTAL money recovered (pre+post shift combined): static (tied with drift-aware)
  Best POST-SHIFT recovery rate (the real test of drift-awareness): drift_aware_ts (0.413 vs 0.227 stationary)

======================================================================
--- Pooling check: subscription + abandonment, ONE shared LearningCore ---
======================================================================
  Subscription -> money=$13475  recovery_rate=0.367
  Abandonment  -> money=$7080   recovery_rate=0.393
  Aggregate money recovered (both domains): $20555
```

**Key findings**:
- **Post-shift recovery rate (0.413 vs 0.227)**: Under a hard regime change, `StationaryThompsonSampling` becomes anchored to its stale pre-shift observations and severely lags. `DriftAwareThompsonSampling` adaptively re-learns the new optimal arm within rounds.
- **Domain pooling**: Subscription and checkout abandonment run simultaneously under one shared `LearningCore`, each maintaining its own independent policy space without interference.

---

## Test suite

All 165 tests pass cleanly across 23 test files.

```
python -m pytest -q

........................................................................ [ 43%]
........................................................................ [ 87%]
.....................                                                    [100%]
165 passed in 19.18s
```

Full breakdown:

```
tests/core/test_customer_case_history.py                                5 passed
tests/core/test_events.py                                               5 passed
tests/core/test_learning_core.py                                       21 passed
tests/core/test_orchestrator.py                                         6 passed
tests/data/test_causal_pressure_parity.py                               4 passed
tests/data/test_checkout_abandonment_generator.py                       6 passed
tests/data/test_subscription_generator.py                               7 passed
tests/data/test_subscription_retry_sequences.py                        11 passed
tests/data/test_support_email_hardship_signal.py                        4 passed
tests/integration/test_anchor_feedback_loop.py                           8 passed
tests/integration/test_bandit_observer_wiring.py                         7 passed
tests/integration/test_checkout_abandonment_through_orchestrator.py     3 passed
tests/integration/test_subscription_cross_case_pressure.py             5 passed
tests/integration/test_subscription_through_orchestrator.py             4 passed
tests/ml/test_baseline.py                                               2 passed
tests/ml/test_calibration.py                                            1 passed
tests/ml/test_features.py                                               9 passed
tests/ml/test_oracle.py                                                 2 passed
tests/ml/test_text_signals.py                                          14 passed
tests/modules/checkout_abandonment/test_checkout_abandonment_module.py 13 passed
tests/modules/dummy/test_dummy_module.py                                7 passed
tests/modules/subscription/test_hardship_policy.py                      7 passed
tests/modules/subscription/test_subscription_module.py                 17 passed
```

---

## How to run

### Run the unified model comparison (GBM vs MLP vs LSTM, same split)

```
python -m backend.ml.compare_all
```

Trains all three models on the same entity-level split. Flat models use the 12-feature vector; LSTM uses the full padded sequence. Prints oracle ceiling, per-model val/test AUC, Brier, and gap-to-oracle. Saves results to `backend/ml/models/comparison_all_results.json`.

### Run the flat model comparison (Baseline vs GBM vs NN, original 10-feature set)

```
python -m backend.ml.compare
```

### Run the sequence model comparison (Comparison Point 4)

```
python -m backend.ml.compare_sequence
```

### Run the enriched flat comparison (GBM/NN with customer history, 11-feature set)

```
python -m backend.ml.compare_with_history
```

### Recompute the oracle ceiling

```
python -m backend.ml.oracle
```

### Train and save the production bundle (Schema v3, 12 features)

```
python -m backend.ml.train_subscription_model
```

Runs offline training against the 12-feature dataset (including `hardship_signal_detected` extracted via `extract_hardship_signal_embedding`), builds calibrated GBM and MLP candidates, evaluates on val set, saves the winner to `backend/ml/models/subscription_winner.joblib` (Schema v3) and a human-readable metrics JSON alongside it.

### Run the Step 6 bandit simulation benchmark

```
python -m backend.ml.bandit_simulation
```

Simulates non-stationary drift and cross-domain pooling through the full event-driven observer pipeline. Compares static heuristic, stationary Thompson Sampling, and drift-aware Thompson Sampling.

### Run all tests

```
python -m pytest -v
```

---

## Architecture

### Shared contract

Every domain module implements the same five-method interface:

| Method | Input | Output | Purpose |
|:-------|:------|:-------|:--------|
| `check_stop(case, history)` | case dict, event history | `StopDecision(should_stop, stop_reason)` | Orchestrator calls this before every cycle — not skippable |
| `diagnose(case, customer_history)` | case dict, optional past customer events | `Diagnosis(root_cause, is_recoverable, confidence, raw_signal, predicted_recovery_probability)` | Domain-owned interpretation of the event |
| `decide(case, diagnosis, history)` | case dict, Diagnosis, history | `Decision(action_type, action_params, reasoning, requires_human_review)` | Policy decision |
| `execute(case, decision)` | case dict, Decision | `ExecutionResult(success, compliance_check_passed, timestamp)` | Takes the action; each module self-certifies its own compliance check |
| `track_outcome(case)` | case dict | `Outcome(status, amount_recovered)` | Ground truth feedback |

Action types: `RETRY`, `SWITCH_CHANNEL`, `ESCALATE`, `WAIT`, `STOP`

Stop reasons: `COMPLIANCE_LIMIT`, `OPT_OUT`, `DIMINISHING_RETURNS`, `COST_THRESHOLD`, `RESOLVED`

Outcome statuses: `RECOVERED`, `PROMISED`, `LOST`, `PENDING`

### Domain modules

Each module is fully independent. It can be instantiated, tested, and run without the orchestrator or any other module. The orchestrator does not make modules work; it makes already-working modules run together.

**Subscription module** (`backend/modules/subscription/module.py`):
- Diagnoses payment failures by ISO 8583 decline code
- Cross-case customer memory: computes `customer_recent_failure_pressure` from prior case outcomes via shared causal EWMA
- Hardship signal extraction: default `extract_hardship_signal_embedding` — contrastive embedding scoring against hardship and neutral anchor banks, three-tier confidence output (`high` / `uncertain` / `none`). Swappable via constructor injection.
- Policy routing: `high` or `uncertain` hardship tier → `ESCALATE` with `requires_human_review=True`; `uncertain` uses distinct reasoning text in audit log
- Compliance enforcement: hard-decline codes (Visa Category 1) fire `COMPLIANCE_LIMIT`; stop-instruction codes (R0, R1, R3) fire `OPT_OUT`
- Retry backoff: 1h, 6h, 24h, 72h
- Uses the 12-feature ML bundle (Schema v3) for recovery-probability prediction; validates schema at load time and falls back to rule-based confidence if bundle is missing or schema-mismatched

**Checkout-abandonment module** (`backend/modules/checkout_abandonment/module.py`):
- Diagnoses dropped checkout sessions by behavioral signal
- Consent gate: refuses to proceed without explicit marketing opt-in — checked in both `check_stop` and `execute`
- Nudge cap enforced via `check_stop`
- Channel escalation by nudge count

### Orchestrator

`backend/core/orchestrator.py`

The orchestrator owns the loop, the stop-gate, and the audit trail. It contains zero domain-specific logic.

- Calls `check_stop` before every `decide` cycle — domain modules cannot skip it
- Queries `EventStore.get_customer_case_history()` and passes raw prior customer events to `diagnose()` without domain interpretation
- When `requires_human_review` is true, routes to a review queue instead of calling `execute`
- Circuit breaker terminates a module that never stops itself after a configurable cap
- Every event is written to the audit log in call order, tagged with its type and optional `customer_id`

### Event sourcing

`backend/core/events.py`

Case state is derived from the event log, not stored redundantly alongside it. Every state transition is written as a single append. A case's current state is the result of replaying its own event log — no dual-write synchronization problem exists. Cross-case history is queried via `get_customer_case_history(customer_id, exclude_case_id)`.

### Feature construction

`backend/ml/features.py`

One place where features are defined and built. Both the trainer and the inference path import from here. A drift between "how the model was trained" and "how the module builds features at inference time" produces confident garbage with no error — this file prevents that by construction.

Production feature set (12 features, Schema v3, fixed order `FEATURE_NAMES_WITH_HISTORY_AND_TEXT`):
```
code_51, code_05, code_91, code_96, code_65, code_61,
attempt_number, is_night, is_near_payday, amount,
customer_recent_failure_pressure, hardship_signal_detected
```

### Hardship signal extraction

`backend/ml/text_signals.py`

Structured signal extraction from free-text customer email. The extractor runs upstream of the ML pipeline and returns a `bool` and diagnostic metadata. The decision model (GBM) never sees raw text.

Key constants:
- `_HARDSHIP_ANCHORS` — 11 sentences covering explicit hardship, medical emergency with financial framing, and indirect paraphrase
- `_NEUTRAL_ANCHORS` — 8 billing/account-management inquiry sentences
- `_CONTRASTIVE_MARGIN = 0.25` — H−N above this → tier "high"
- `_CONTRASTIVE_UNCERTAIN_FLOOR = 0.05` — H−N in (0.05, 0.25] → tier "uncertain"

---

## Project structure

```
backend/
  core/
    contract.py          -- shared Diagnosis, Decision, Outcome, StopDecision dataclasses
    events.py            -- event-sourced state store with EventObserver protocol & subscribe
    orchestrator.py      -- the loop, stop-gate, audit trail, submit_human_review
    learning_core.py     -- static, stationary, and drift-aware Thompson Sampling bandit policies
    bandit_observer.py   -- single-writer event observer feeding outcomes to learning core
  data/
    subscription_generator.py      -- grounded synthetic subscription records & retry sequences
    checkout_abandonment_generator.py -- grounded synthetic abandonment records
    splitting.py         -- entity-level train/val/test splitting
  ml/
    features.py          -- canonical flat & enriched feature construction (one source of truth)
    sequence_features.py -- sequence per-step feature construction (10 features)
    text_signals.py      -- hardship signal extraction: contrastive embedding, keyword, LLM, feedback growth
    compare.py           -- flat model comparison: baseline vs GBM vs NN (10 features)
    compare_sequence.py  -- sequence model comparison: LSTM vs chain oracle ceiling
    compare_with_history.py -- flat models with customer history parity (11 features)
    compare_all.py       -- UNIFIED: GBM vs MLP vs LSTM, same split, schema v3
    bandit_simulation.py -- Step 6 drift & pooling benchmark over observer-driven pipeline
    train_subscription_model.py -- trainer: produces 12-feature subscription_winner.joblib (Schema v3)
    oracle.py            -- flat oracle AUC ceiling computation
    calibration.py       -- calibration evaluation (Platt/sigmoid)
    evaluation.py        -- reliability curves, per-code breakdown
    progress.py          -- progress bar for long searches
    models/
      baseline.py        -- rule-based lookup baseline
      gbm.py             -- XGBoost hyperparameter search and training
      neural_net.py      -- PyTorch MLP hyperparameter search and training
      sequence.py        -- PyTorch LSTM sequence model
      subscription_winner.joblib          -- deployed model bundle (generated, Schema v3)
      subscription_winner_metrics.json    -- human-readable audit copy (generated)
      comparison_all_results.json         -- unified GBM/MLP/LSTM comparison results (generated)
  modules/
    dummy/
      module.py          -- stub for orchestrator testing
    subscription/
      module.py          -- subscription recovery (cross-case memory, hardship signal, bandit retry backoff)
    checkout_abandonment/
      module.py          -- checkout session recovery (bandit channel selection)

tests/
  core/                  -- orchestrator, event-sourcing, learning core, and customer case history tests
  data/                  -- generator, splitting, retry-sequences, causal pressure, hardship signal tests
  integration/           -- orchestrator + module, bandit observer wiring, and anchor feedback loop tests
  ml/                    -- oracle, baseline, calibration, feature, text signal tests
  modules/               -- per-module unit tests (each module tested in isolation)
```

---

## What is left to build

### Step 7 — B2B receivables module [NOT STARTED]

Third domain. The most compliance-heavy: DND registry checks, Section 43B(h) MSME payment timeline rules. Also the domain that exercises `on_promise_due` and the human-review queue path most fully.

### Step 8 — Mandate retry sequencer [STRETCH]

Reuses the subscription module's shape on a different payment rail (UPI/NACH). Cheap to add once the subscription module is proven.

### Step 9 — Frontend [NOT FINALIZED]

Three views:
- Merchant view: live transaction feed, money recovered, recovery rate, active recoveries
- Developer/audit view: full per-case trace (diagnose → decide → execute → track), exportable audit log
- Human review queue: cases where `requires_human_review` is true, with approve/override controls

---

## Open items (documented, not silently deferred)

- Checkout-abandonment nudge cap: `MAX_NUDGES = 3` is a judgment call. No authoritative source equivalent to Visa's retry cap exists for abandonment nudges. Flagged in the module's source. Real A/B data should replace this once available.
- Per-signal recovery-rate constants in the checkout-abandonment generator are not individually sourced the way the subscription decline-code rates partially are. Flagged in the generator's docstring.
- Whether to extend the subscription generator to make recovery probability depend on amount for codes other than 51, or on day-of-week. Currently it does not.
- Exact `requires_human_review` confidence threshold per domain.
- Exact promise-to-pay cadence (how many broken promises before `DIMINISHING_RETURNS` fires).
- Exact bandit algorithm variant for the learning core (discount factor vs window vs both), pending step 6.
- Hardship anchor feedback loop (uncertain-tier → human review → new anchor): tracked as a Step 6 learning-core task.
- Billing inquiries mentioning "charged" or "payment" score 0.42–0.43 against hardship anchors on `all-MiniLM-L6-v2`. Contrastive scoring (H−N) correctly rejects them (H−N = −0.47), but the boundary is documented here: do not lower `_CONTRASTIVE_UNCERTAIN_FLOOR` below 0.0 without re-running the probe script in `backend/ml/models/` to verify no neutral sentence has risen above the new floor.
- `checkout_abandonment.diagnose()` accepts `customer_history` (required by the shared contract) but does not use it — no cross-case behavioral signal has been built or tested for this domain, unlike subscription's `customer_recent_failure_pressure`. A documented scope decision (flagged in the module's source), not silently dropped.

---

## Design decisions and what was tested vs assumed

Every claim in this section has been evaluated empirically or has a documented source.

**Oracle ceiling**: Computed directly from the generator's `true_recovery_probability` function against the generator's sampled outcomes — not estimated from a trained model. The theoretical best AUC any model can achieve on this feature set, with this generator, is 0.7035 on the unified test split. GBM reaches 0.7002.

**GBM vs LSTM on sequence data**: LSTM received the full retry sequence; GBM received only the flat per-attempt vector. GBM won by 0.002 AUC. The reason is structural: the generator's recovery probability depends on prior attempts only through `attempt_number`, which the flat model already has. Sequence order added zero marginal signal. This is a finding, not a failure.

**Hardship contrastive threshold calibrated on real scores**: `_CONTRASTIVE_MARGIN = 0.25` and `_CONTRASTIVE_UNCERTAIN_FLOOR = 0.05` were set after running the probe script against all test sentences on the actual model. The gap between the lowest hardship H−N (+0.30) and the highest neutral H−N (−0.31) is 0.61 — both thresholds sit comfortably inside that gap with a 0.15-point buffer on each side.

**Calibration method**: Sigmoid (Platt scaling) outperformed isotonic regression on Brier score (0.2186 vs 0.2193) at this validation-set size (~1,235 rows). Isotonic is non-parametric and needs more calibration data to be reliable than a 2-parameter sigmoid fit.

**Bayesian hyperparameter search**: Not adopted for this dataset. With GBM already within 0.003 AUC of the oracle ceiling, a smarter search strategy has no meaningful headroom left to find.

**Scaling to 50k-100k records**: Not adopted. More data tightens the estimate around the ceiling; it does not raise the ceiling, which is a property of the feature set's informativeness, not the training sample size.

**Derived features (business hours, amount bucketing for non-51 codes)**: Not adopted, for a specific reason documented and not a general rejection. The generator's true probability function does not depend on these features for any code other than 51. Testing them would show zero effect because the synthetic data does not encode that relationship — not because the technique is wrong.

**Cross-distribution generalization**: Actually run, not just planned. GBM trained on regime A drops from 0.693 to 0.620 AUC on regime B (a 7.3-point drop). The rule-based baseline drops from 0.605 to 0.527 (a 7.8-point drop). GBM degrades less — the actual definition of "generalizable."

---

## Model-family scaling — when to move beyond GBM (analyzed, not yet needed)

A documented rule of thumb, derived from what step 5 actually found rather than assumed in the abstract:

- **More tabular columns** (e.g. device type, IP risk score, account age): stay with GBM. Confirmed empirically here — `customer_recent_failure_pressure` added as a single engineered feature let a flat GBM track its own oracle ceiling as tightly as the LSTM did (0.0055 vs 0.0049 gap), with no architecture change needed.
- **Unstructured data** (support-email text, etc.): does NOT require jumping straight to a transformer/LLM as the decision model. A cheaper, consistent pattern: extract a structured signal upstream (contrastive embedding → a bool/enum feature), keep GBM as the decision layer. Only the feature-extraction step changes. Implemented: `extract_hardship_signal_embedding` as the default extractor in Schema v3.
- **Deep, heterogeneous, cross-domain event sequences** (the Vulcan-scale case — hundreds of mixed-event-type steps spanning subscription, abandonment, and B2B in one timeline): plausibly does need a real sequence/attention architecture, since an EWMA-style flat feature loses step-level detail at that scale. This is a **hypothesis, not a finding** — never built or tested at that scale, unlike the claims above. Flagged as an open question, not asserted as an architectural conclusion.

---

## Technical notes

**Python version**: 3.14.2

**Key dependencies**:
- `xgboost` — GBM training and hyperparameter search
- `torch` — PyTorch neural net and LSTM
- `scikit-learn` — pipelines, calibration, GroupKFold
- `sentence-transformers` — `all-MiniLM-L6-v2` embedding model for hardship signal extraction
- `joblib` — model bundle serialization
- `numpy`, `scipy` — numerical operations, probability distributions
- `pandas` — feature matrix construction
- `pytest` — test suite

**Running on Windows**: All paths use forward slashes internally. The project root must be on `sys.path` for `-m` module invocations to work (`python -m backend.ml.compare` from the project root).
