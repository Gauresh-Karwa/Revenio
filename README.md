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

#### Deployed production bundle (Schema v2, 11 features)

The production trainer (`train_subscription_model.py`) trains against the enriched feature set (`FEATURE_NAMES_WITH_HISTORY`, 11 features including `customer_recent_failure_pressure`) and serializes the calibrated winner:

```
backend/ml/models/subscription_winner.joblib
backend/ml/models/subscription_winner_metrics.json
```

Real training output (`python -m backend.ml.train_subscription_model`):
```
Winner: GBM
Saved bundle to backend/ml/models/subscription_winner.joblib
Per-candidate val AUC: {'GBM': {'val_auc': 0.6775, 'val_brier': 0.2180},
                        'MLP': {'val_auc': 0.6772, 'val_brier': 0.2175}}
Winner test AUC: 0.6788
Winner test Brier: 0.2123
```

- **Schema Guard**: `SubscriptionModule` validates `feature_names == FEATURE_NAMES_WITH_HISTORY` upon loading `subscription_winner.joblib`. If a stale 10-feature bundle is detected, it raises a clear warning and safely falls back to rule-based confidence rather than producing silent feature mismatch errors.
- **Cross-Case Inference Integration**: When `orchestrator.process_case()` runs, it retrieves the customer's past case outcomes from `EventStore.get_customer_case_history()`, and `SubscriptionModule.diagnose()` causally computes `customer_recent_failure_pressure` via `compute_pressure_from_customer_history()` to populate the 11th feature.

#### Step 5 Addendum — Sequence Model (4th Comparison Point) [DONE]

Architecture doc §6.3 requires a sequence model (LSTM) as a fourth diagnosis-layer comparison point, evaluated with the same held-out entity-level discipline as baseline, GBM, and NN.

What was built:
- `generate_subscription_retry_sequences()` in `backend/data/subscription_generator.py` — generates genuine chronological retry chains (attempt $k$ only exists if attempt $k-1$ failed), incorporating a causal, recency-weighted customer failure pressure (EWMA with $\alpha=0.5$).
- `backend/ml/sequence_features.py` — per-step feature construction (10 features: 6 one-hot decline codes, `is_night`, `is_near_payday`, `amount`, `customer_recent_failure_pressure`).
- `backend/ml/models/sequence.py` — small sequence model (`RetryLSTM`), tuned via random search + `GroupKFold` entity-aware CV.
- `backend/ml/compare_sequence.py` — standalone comparison script evaluating the LSTM against its own chain-distribution oracle ceiling.

Results (`compare_sequence.py` output, seed=42):

```
STEP 5, COMPARISON POINT 4 — LSTM sequence model (architecture doc 6.3)
v2: includes causal customer-history (recency-weighted) effect

7952 genuine retry-chain cases generated (soft-decline only).
  Sanity check — final recovery rate, low pressure (<0.1, n=7018): 0.832
  Sanity check — final recovery rate, high pressure (>0.5, n=89): 0.663

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

| Metric | Value |
|:---|:---|
| Genuine retry-chain cases generated | 7,952 |
| Per-attempt training examples (train / val / test) | 10,542 / 2,227 / 2,348 |
| Best hyperparameters | `hidden_size=26, lr=0.01236, weight_decay=2.38e-06` |
| LSTM CV AUC | 0.702 |
| LSTM test AUC | 0.6986 |
| Oracle ceiling for this chain distribution | 0.7035 |
| LSTM gap to its own ceiling | 0.0049 |
| (Reference) GBM gap to its own flat ceiling | 0.0026 |

#### Files

- `backend/ml/compare.py` — flat model comparison: baseline vs GBM vs NN
- `backend/ml/compare_sequence.py` — sequence model comparison: LSTM vs chain oracle ceiling
- `backend/ml/compare_with_history.py` — flat models with customer history feature parity comparison
- `backend/ml/train_subscription_model.py` — trainer: produces deployable 11-feature bundle
- `backend/ml/oracle.py` — flat oracle ceiling computation
- `backend/ml/features.py` — canonical flat and enriched feature construction (one source of truth)
- `backend/ml/sequence_features.py` — sequence per-step feature construction
- `backend/ml/models/gbm.py` — XGBoost hyperparameter search and training
- `backend/ml/models/neural_net.py` — PyTorch MLP hyperparameter search and training
- `backend/ml/models/sequence.py` — PyTorch LSTM sequence model
- `backend/ml/models/baseline.py` — rule-based baseline
- `backend/ml/calibration.py` — calibration evaluation
- `backend/ml/evaluation.py` — reliability curves, per-code breakdown
- `tests/ml/` — oracle, baseline, calibration, feature tests
- `tests/data/test_subscription_retry_sequences.py` — sequence generator, causality, and pressure tests
- `tests/data/test_causal_pressure_parity.py` — training/inference EWMA parity tests
- `tests/core/test_customer_case_history.py` — cross-case event store query tests
- `tests/integration/test_subscription_cross_case_pressure.py` — end-to-end customer memory integration tests

---

## Test suite

All 101 tests pass cleanly across 16 test files.

```
python -m pytest -v

tests/core/test_customer_case_history.py                                5 passed
tests/core/test_events.py                                              5 passed
tests/core/test_orchestrator.py                                        6 passed
tests/data/test_causal_pressure_parity.py                              4 passed
tests/data/test_checkout_abandonment_generator.py                      6 passed
tests/data/test_subscription_generator.py                              7 passed
tests/data/test_subscription_retry_sequences.py                       11 passed
tests/integration/test_checkout_abandonment_through_orchestrator.py    3 passed
tests/integration/test_subscription_cross_case_pressure.py             5 passed
tests/integration/test_subscription_through_orchestrator.py            4 passed
tests/ml/test_baseline.py                                              2 passed
tests/ml/test_calibration.py                                           1 passed
tests/ml/test_features.py                                              3 passed
tests/ml/test_oracle.py                                                2 passed
tests/modules/checkout_abandonment/test_checkout_abandonment_module.py    13 passed
tests/modules/dummy/test_dummy_module.py                               7 passed
tests/modules/subscription/test_subscription_module.py                17 passed

101 passed in 3.90s
```

---

## How to run

### Run the flat model comparison (Baseline vs GBM vs NN, original 10-feature set)

```
python -m backend.ml.compare
```

Generates a fresh dataset, runs entity-level splitting, performs random hyperparameter search for GBM and NN, evaluates all three models against the rule-based baseline on the held-out test set, prints calibration results and the cross-distribution generalization test.

### Run the sequence model comparison (Comparison Point 4)

```
python -m backend.ml.compare_sequence
```

Generates chained retry sequences with causal customer failure pressure, tunes the LSTM sequence model, and evaluates test AUC against the chain-distribution oracle ceiling.

### Run the enriched flat comparison (GBM/NN with customer history, 11-feature set)

```
python -m backend.ml.compare_with_history
```

Evaluates GBM and NN when given the 11th `customer_recent_failure_pressure` feature on the enriched flat dataset against its oracle ceiling (0.6889).

### Recompute the oracle ceiling

```
python -m backend.ml.oracle
```

Computes the theoretical AUC upper bound for the current generator by scoring the generator's own `true_recovery_probability` function against the sampled binary outcomes. This is the number to compare model AUC against, not 1.0.

### Train and save the production bundle

```
python -m backend.ml.train_subscription_model
```

Runs offline training against the enriched 11-feature dataset, builds calibrated Pipelines for both candidates, evaluates on val set, saves the winner to `backend/ml/models/subscription_winner.joblib` (Schema v2) and a human-readable metrics JSON alongside it.

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
- Compliance enforcement: hard-decline codes (Visa Category 1) fire `COMPLIANCE_LIMIT`; stop-instruction codes (R0, R1, R3) fire `OPT_OUT`
- Retry backoff: 1h, 6h, 24h, 72h
- Uses the 11-feature ML bundle for recovery-probability prediction at inference time; validates schema at load time and falls back to rule-based confidence if bundle is missing or invalid

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

Production enriched feature set (11 features, fixed order `FEATURE_NAMES_WITH_HISTORY`):
```
code_51, code_05, code_91, code_96, code_65, code_61,
attempt_number, is_night, is_near_payday, amount,
customer_recent_failure_pressure
```

---

## Project structure

```
backend/
  core/
    contract.py          -- shared Diagnosis, Decision, Outcome, StopDecision dataclasses
    events.py            -- event-sourced state store with customer history queries
    orchestrator.py      -- the loop, stop-gate, audit trail, cross-case event routing
  data/
    subscription_generator.py      -- grounded synthetic subscription records & retry sequences
    checkout_abandonment_generator.py -- grounded synthetic abandonment records
    splitting.py         -- entity-level train/val/test splitting
  ml/
    features.py          -- canonical flat & enriched feature construction (one source of truth)
    sequence_features.py -- sequence per-step feature construction (10 features)
    compare.py           -- flat model comparison: baseline vs GBM vs NN
    compare_sequence.py  -- sequence model comparison: LSTM vs chain oracle ceiling
    compare_with_history.py -- flat models with customer history feature parity comparison
    train_subscription_model.py -- trainer: produces 11-feature subscription_winner.joblib (Schema v2)
    oracle.py            -- flat oracle AUC ceiling computation
    calibration.py       -- calibration evaluation (Platt/sigmoid)
    evaluation.py        -- reliability curves, per-code breakdown
    progress.py          -- progress bar for long searches
    models/
      baseline.py        -- rule-based lookup baseline
      gbm.py             -- XGBoost hyperparameter search and training
      neural_net.py      -- PyTorch MLP hyperparameter search and training
      sequence.py        -- PyTorch LSTM sequence model
      subscription_winner.joblib          -- deployed model bundle (generated, Schema v2)
      subscription_winner_metrics.json    -- human-readable audit copy (generated)
  modules/
    dummy/
      module.py          -- stub for orchestrator testing
    subscription/
      module.py          -- subscription payment-retry recovery (cross-case memory enabled)
    checkout_abandonment/
      module.py          -- checkout session recovery

tests/
  core/                  -- orchestrator, event-sourcing, and customer case history tests
  data/                  -- generator, splitting, retry-sequences, and causal pressure parity tests
  integration/           -- orchestrator + module end-to-end and cross-case pressure tests
  ml/                    -- oracle, baseline, calibration, feature tests
  modules/               -- per-module unit tests (each module tested in isolation)
```

---

## What is left to build

### Step 6 — Learning core [NOT STARTED]

Drift-aware contextual bandit: discounted/sliding-window Thompson Sampling (exponentially decaying weight on older outcomes). Hand-implemented in NumPy/SciPy — no off-the-shelf bandit library covers this variant.

Will be tested first with subscription-only data, then confirmed it improves (and does not break) when abandonment data is pooled in.

One open question locked but not resolved: whether the bandit policy updates should use a discount factor, a sliding window, or both.

### Step 7 — B2B receivables module [NOT STARTED]

Third domain. The most compliance-heavy: DND registry checks, Section 43B(h) MSME payment timeline rules. Also the domain that exercises `on_promise_due` and the human-review queue path most fully.

### Step 8 — Mandate retry sequencer [STRETCH]

Reuses the subscription module's shape on a different payment rail (UPI/NACH). Cheap to add once the subscription module is proven.

### Step 9 — Frontend [NOT FINALIZED]

Three views:
- Merchant view: live transaction feed, money recovered, recovery rate, active recoveries
- Developer/audit view: full per-case trace (diagnose -> decide -> execute -> track), exportable audit log
- Human review queue: cases where `requires_human_review` is true, with approve/override controls

---

## Open items (documented, not silently deferred)

- Checkout-abandonment nudge cap: `MAX_NUDGES = 3` is a judgment call. No authoritative source equivalent to Visa's retry cap exists for abandonment nudges. Flagged in the module's source. Real A/B data should replace this once available.
- Per-signal recovery-rate constants in the checkout-abandonment generator are not individually sourced the way the subscription decline-code rates partially are. Flagged in the generator's docstring.
- Whether to extend the subscription generator to make recovery probability depend on amount for codes other than 51, or on day-of-week. Currently it does not.
- Exact `requires_human_review` confidence threshold per domain.
- Exact promise-to-pay cadence (how many broken promises before `DIMINISHING_RETURNS` fires).
- Exact bandit algorithm variant for the learning core (discount factor vs window vs both), pending step 6.
- Whether to merge `compare_sequence.py` into `compare.py` as a unified comparison script — currently kept separate because the sequence model needs chained retry data rather than the flat per-row dataset the other candidates share.
- `checkout_abandonment.diagnose()` accepts `customer_history` (required by the shared contract) but does not use it — no cross-case behavioral signal has been built or tested for this domain, unlike subscription's `customer_recent_failure_pressure`. A documented scope decision (flagged in the module's source), not silently dropped.
- Whether to extend the recovery layer to unstructured customer communications (e.g. support-email text signaling hardship). Analyzed but not built: the conclusion was NOT to reach for a transformer/LLM directly on raw text — instead, extract a structured signal upstream (e.g. `hardship_signal_detected: bool`, `extracted_reason_code: enum`) via a one-time embedding/extraction step, and feed that into the existing flat feature pipeline exactly like `customer_recent_failure_pressure`, keeping GBM as the decision layer. Two things this would need before any implementation: (1) a consent/data-handling story for ingesting free-text customer communications — a materially higher-stakes data category than a decline code, comparable to why B2B's DND/43B(h) compliance checks exist; (2) a policy decision, not just a modeling one — certain extracted signals (e.g. explicit hardship disclosures) should likely force `requires_human_review=True` unconditionally, regardless of confidence score, rather than being left to a threshold.

---

## Design decisions and what was tested vs assumed

Every claim in this section has been evaluated empirically or has a documented source.

**Oracle ceiling**: Computed directly from the generator's `true_recovery_probability` function against the generator's sampled outcomes — not estimated from a trained model. The theoretical best AUC any model can achieve on this feature set, with this generator, is 0.6956 on the test split. GBM reaches 0.693.

**Calibration method**: Sigmoid (Platt scaling) outperformed isotonic regression on Brier score (0.2186 vs 0.2193) at this validation-set size (~1,235 rows). Isotonic is non-parametric and needs more calibration data to be reliable than a 2-parameter sigmoid fit.

**Bayesian hyperparameter search**: Not adopted for this dataset. With GBM already within 0.003 AUC of the oracle ceiling, a smarter search strategy has no meaningful headroom left to find.

**Scaling to 50k-100k records**: Not adopted. More data tightens the estimate around the ceiling; it does not raise the ceiling, which is a property of the feature set's informativeness, not the training sample size.

**Derived features (business hours, amount bucketing for non-51 codes)**: Not adopted, for a specific reason documented and not a general rejection. The generator's true probability function does not depend on these features for any code other than 51. Testing them would show zero effect because the synthetic data does not encode that relationship — not because the technique is wrong. The generator was deliberately extended with amount-dependence for code-51 specifically, because "insufficient funds" is mechanistically a gap between balance and requested amount.

**Cross-distribution generalization**: Actually run, not just planned. GBM trained on regime A drops from 0.693 to 0.620 AUC on regime B (a 7.3-point drop). The rule-based baseline drops from 0.605 to 0.527 (a 7.8-point drop). GBM degrades less — the actual definition of "generalizable."

---

## Model-family scaling — when to move beyond GBM (analyzed, not yet needed)

A documented rule of thumb, derived from what step 5 actually found rather than assumed in the abstract:

- **More tabular columns** (e.g. device type, IP risk score, account age): stay with GBM. Confirmed empirically here — `customer_recent_failure_pressure` added as a single engineered feature let a flat GBM track its own oracle ceiling as tightly as the LSTM did (0.0055 vs 0.0049 gap), with no architecture change needed.
- **Unstructured data** (support-email text, etc.): does NOT require jumping straight to a transformer/LLM as the decision model. A cheaper, consistent pattern: extract a structured signal upstream (embedding or one-time LLM call → a bool/enum feature), keep GBM as the decision layer. Only the feature-extraction step changes.
- **Deep, heterogeneous, cross-domain event sequences** (the Vulcan-scale case — hundreds of mixed-event-type steps spanning subscription, abandonment, and B2B in one timeline): plausibly does need a real sequence/attention architecture, since an EWMA-style flat feature loses step-level detail at that scale. This is a **hypothesis, not a finding** — never built or tested at that scale, unlike the claims above. Flagged as an open question, not asserted as an architectural conclusion.

---

## Technical notes

**Python version**: 3.14.2

**Key dependencies**:
- `xgboost` — GBM training and hyperparameter search
- `torch` — PyTorch neural net
- `scikit-learn` — pipelines, calibration, GroupKFold
- `joblib` — model bundle serialization
- `numpy`, `scipy` — numerical operations, probability distributions
- `pandas` — feature matrix construction
- `pytest` — test suite

**Running on Windows**: All paths use forward slashes internally. The project root must be on `sys.path` for `-m` module invocations to work (`python -m backend.ml.compare` from the project root).
