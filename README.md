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

#### Deployed bundle

The trainer (`train_subscription_model.py`) runs the same random hyperparameter search as `compare.py` and saves the winning model as a self-contained bundle:

```
backend/ml/models/subscription_winner.joblib
backend/ml/models/subscription_winner_metrics.json
```

The bundle contains the full sklearn Pipeline (preprocessing + calibrated classifier), `feature_names` for schema verification, `best_params` for the winning configuration, per-candidate metrics, and `auc_margin_vs_runner_up` to document how close the runner-up was.

The winner is whichever candidate has the higher val AUC — no tie-break rule. Results are reproducible run-to-run because both `tune_gbm` and `tune_nn` use a fixed seed.

#### Winner selection in practice

On a given run with seed=42, the val AUC gap between GBM and MLP is 0.0007–0.0016 — inside noise at this dataset scale. Which one wins varies slightly depending on the specific tuned hyperparameters found for each. The bundle records `auc_margin_vs_runner_up` for audit visibility. The oracle ceiling confirms both are at approximately the same distance from the theoretical best possible on this feature set.

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

Key takeaways:
1. **The customer-history effect is real and learnable**: Customers with recent failure pressure (>0.5) recover at 66.3% compared to 83.2% for low-pressure (<0.1) customers.
2. **Tight tracking to oracle ceiling**: The LSTM achieves an AUC of 0.6986, landing within 0.0049 of the 0.7035 ceiling.
3. **Caveat on flat vs sequence comparison**: The LSTM has access to `customer_recent_failure_pressure`, which the flat generator (`generate_subscription_dataset`) does not encode. The fair metric is each model's **gap to its own distribution's oracle ceiling** (0.0049 for LSTM vs 0.0026 for GBM).

#### Files

- `backend/ml/compare.py` — flat model comparison: baseline vs GBM vs NN
- `backend/ml/compare_sequence.py` — sequence model comparison: LSTM vs chain oracle ceiling
- `backend/ml/train_subscription_model.py` — trainer (produces deployable bundle)
- `backend/ml/oracle.py` — flat oracle ceiling computation
- `backend/ml/features.py` — canonical flat feature construction
- `backend/ml/sequence_features.py` — sequence per-step feature construction
- `backend/ml/models/gbm.py` — XGBoost hyperparameter search and training
- `backend/ml/models/neural_net.py` — PyTorch MLP hyperparameter search and training
- `backend/ml/models/sequence.py` — PyTorch LSTM sequence model
- `backend/ml/models/baseline.py` — rule-based baseline
- `backend/ml/calibration.py` — calibration evaluation
- `backend/ml/evaluation.py` — reliability curves, per-code breakdown
- `tests/ml/` — oracle, baseline, calibration, feature tests
- `tests/data/test_subscription_retry_sequences.py` — sequence generator, causality, and pressure tests

---

## Test suite

All 87 tests pass as of step 5 completion.

```
python -m pytest -v

tests/core/test_events.py                                              5 passed
tests/core/test_orchestrator.py                                        6 passed
tests/data/test_checkout_abandonment_generator.py                      6 passed
tests/data/test_subscription_generator.py                              7 passed
tests/data/test_subscription_retry_sequences.py                       11 passed
tests/integration/test_checkout_abandonment_through_orchestrator.py    3 passed
tests/integration/test_subscription_through_orchestrator.py            4 passed
tests/ml/test_baseline.py                                              2 passed
tests/ml/test_calibration.py                                           1 passed
tests/ml/test_features.py                                              3 passed
tests/ml/test_oracle.py                                                2 passed
tests/modules/checkout_abandonment/test_checkout_abandonment_module.py    13 passed
tests/modules/dummy/test_dummy_module.py                               7 passed
tests/modules/subscription/test_subscription_module.py                17 passed

87 passed in 3.85s
```

---

## How to run

### Run the flat model comparison (Baseline vs GBM vs NN)

```
python -m backend.ml.compare
```

Generates a fresh dataset, runs entity-level splitting, performs random hyperparameter search for GBM and NN, evaluates all three models against the rule-based baseline on the held-out test set, prints calibration results and the cross-distribution generalization test.

### Run the sequence model comparison (Comparison Point 4)

```
python -m backend.ml.compare_sequence
```

Generates chained retry sequences with causal customer failure pressure, tunes the LSTM sequence model, and evaluates test AUC against the chain-distribution oracle ceiling.

### Recompute the oracle ceiling

```
python -m backend.ml.oracle
```

Computes the theoretical AUC upper bound for the current generator by scoring the generator's own `true_recovery_probability` function against the sampled binary outcomes. This is the number to compare model AUC against, not 1.0.

### Train and save the production bundle

```
python -m backend.ml.train_subscription_model
```

Runs the same hyperparameter search as `compare.py`, builds calibrated Pipelines for both candidates, evaluates on val set, saves the winner to `backend/ml/models/subscription_winner.joblib` and a human-readable metrics JSON alongside it.

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
| `diagnose(case)` | case dict | `Diagnosis(root_cause, is_recoverable, confidence, raw_signal)` | Domain-owned interpretation of the event |
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
- Compliance enforcement: hard-decline codes (Visa Category 1) fire `COMPLIANCE_LIMIT`; stop-instruction codes (R0, R1, R3) fire `OPT_OUT`
- Retry backoff: 1h, 6h, 24h, 72h
- Uses the ML bundle for recovery-probability prediction at inference time; falls back to rule-based confidence if no bundle exists

**Checkout-abandonment module** (`backend/modules/checkout_abandonment/module.py`):
- Diagnoses dropped checkout sessions by behavioral signal
- Consent gate: refuses to proceed without explicit marketing opt-in — checked in both `check_stop` and `execute`
- Nudge cap enforced via `check_stop`
- Channel escalation by nudge count

### Orchestrator

`backend/core/orchestrator.py`

The orchestrator owns the loop, the stop-gate, and the audit trail. It contains zero domain-specific logic.

- Calls `check_stop` before every `decide` cycle — domain modules cannot skip it
- When `requires_human_review` is true, routes to a review queue instead of calling `execute`
- Circuit breaker terminates a module that never stops itself after a configurable cap
- Every event is written to the audit log in call order, tagged with its type

### Event sourcing

`backend/core/events.py`

Case state is derived from the event log, not stored redundantly alongside it. Every state transition is written as a single append. A case's current state is the result of replaying its own event log — no dual-write synchronization problem exists.

### Feature construction

`backend/ml/features.py`

One place where features are defined and built. Both the trainer and the inference path import from here. A drift between "how the model was trained" and "how the module builds features at inference time" produces confident garbage with no error — this file prevents that by construction.

Feature set (10 features, fixed order):
```
code_51, code_05, code_91, code_96, code_65, code_61,
attempt_number, is_night, is_near_payday, amount
```

---

## Project structure

```
backend/
  core/
    contract.py          -- shared Diagnosis, Decision, Outcome, StopDecision dataclasses
    events.py            -- event-sourced state store
    orchestrator.py      -- the loop, stop-gate, audit trail, human-review routing
  data/
    subscription_generator.py      -- grounded synthetic subscription records
    checkout_abandonment_generator.py -- grounded synthetic abandonment records
    splitting.py         -- entity-level train/val/test splitting
  ml/
    features.py          -- canonical flat feature construction (one source of truth)
    sequence_features.py -- sequence per-step feature construction (10 features)
    compare.py           -- flat model comparison: baseline vs GBM vs NN
    compare_sequence.py  -- sequence model comparison: LSTM vs chain oracle ceiling
    train_subscription_model.py -- trainer: produces subscription_winner.joblib
    oracle.py            -- flat oracle AUC ceiling computation
    calibration.py       -- calibration evaluation (Platt/sigmoid)
    evaluation.py        -- reliability curves, per-code breakdown
    progress.py          -- progress bar for long searches
    models/
      baseline.py        -- rule-based lookup baseline
      gbm.py             -- XGBoost hyperparameter search and training
      neural_net.py      -- PyTorch MLP hyperparameter search and training
      sequence.py        -- PyTorch LSTM sequence model
      subscription_winner.joblib          -- deployed model bundle (generated)
      subscription_winner_metrics.json    -- human-readable audit copy (generated)
  modules/
    dummy/
      module.py          -- stub for orchestrator testing
    subscription/
      module.py          -- subscription payment-retry recovery
    checkout_abandonment/
      module.py          -- checkout session recovery

tests/
  core/                  -- orchestrator and event-sourcing tests
  data/                  -- generator, splitting, and retry-sequence tests
  integration/           -- full orchestrator + module end-to-end tests
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
- Whether to engineer the `customer_recent_failure_pressure` signal as a flat feature for GBM/MLP in `compare.py` to directly compare flat vs sequence architectures with identical feature parity.
- Whether to merge `compare_sequence.py` into `compare.py` as a unified 4-candidate comparison script — currently kept separate because the sequence model needs chained retry data rather than the flat per-row dataset the other three candidates share.

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
