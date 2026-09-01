"""
Smoke tests for the B2B ML comparison pipeline.
Does NOT run the full training loop (that takes seconds per model and
belongs in a manual benchmark, not in the test suite). Tests that every
component the comparison script depends on is importable, correctly shaped,
and self-consistent — the same pattern as test_baseline.py / test_oracle.py.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from backend.data.b2b_generator import generate_b2b_dataset, true_payment_probability
from backend.data.splitting import entity_level_split
from backend.ml.compare_b2b import (
    B2B_FEATURE_NAMES,
    _build_baseline,
    _build_gbm,
    _build_mlp,
    _chased_only,
    _flat_matrix,
    _oracle_auc,
)


def _small_chased():
    """A small dataset (500 customers) used in tests that fit/predict models."""
    records = generate_b2b_dataset(n_customers=500, seed=42)
    return _chased_only(records)


# --- Feature construction ---

def test_feature_vector_length_matches_feature_names():
    records = generate_b2b_dataset(n_customers=50, seed=1)
    chased = _chased_only(records)
    X, y = _flat_matrix(chased)
    assert X.shape[1] == len(B2B_FEATURE_NAMES), (
        f"Feature matrix has {X.shape[1]} columns but B2B_FEATURE_NAMES has {len(B2B_FEATURE_NAMES)}"
    )


def test_flat_matrix_labels_are_binary():
    chased = _small_chased()
    X, y = _flat_matrix(chased)
    assert set(y).issubset({0.0, 1.0})


def test_chased_only_excludes_dnd_opted_out_disputed():
    records = generate_b2b_dataset(n_customers=2000, seed=2)
    chased = _chased_only(records)
    assert all(
        not r.on_dnd_registry and not r.has_opted_out and not r.is_disputed
        for r in chased
    )
    # All remaining labels are sampled by the generator, not forced False
    assert len(chased) > 0


# --- Oracle ---

def test_oracle_auc_is_above_random():
    records = generate_b2b_dataset(n_customers=2000, seed=3)
    chased = _chased_only(records)
    _, _, test_records = entity_level_split(chased)
    oracle = _oracle_auc(test_records)
    assert oracle > 0.55, f"Oracle AUC unexpectedly low: {oracle:.4f}"


def test_oracle_uses_exact_generating_function():
    """
    true_payment_probability() is the exact function b2b_generator.py
    samples against. Passing it the same inputs the generator used means
    oracle AUC is a true Bayes ceiling, not an approximation.
    """
    records = generate_b2b_dataset(n_customers=200, seed=7)
    chased = _chased_only(records)
    probs = np.array([
        true_payment_probability(
            r.days_overdue, r.is_msme_registered, r.customer_recent_payment_pressure
        )
        for r in chased
    ])
    y = np.array([1.0 if r.recovered else 0.0 for r in chased])
    auc = roc_auc_score(y, probs)
    assert 0.5 < auc < 1.0


# --- Model smoke tests (tiny dataset, just confirm they fit and predict) ---

def test_baseline_fits_and_predicts():
    chased = _small_chased()
    train, val, test = entity_level_split(chased)
    X_tr, y_tr = _flat_matrix(train)
    X_te, y_te = _flat_matrix(test)
    model = _build_baseline()
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    assert probs.shape == (len(y_te),)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_gbm_fits_and_predicts():
    chased = _small_chased()
    train, val, test = entity_level_split(chased)
    X_tr, y_tr = _flat_matrix(train)
    X_te, y_te = _flat_matrix(test)
    model = _build_gbm()
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    assert probs.shape == (len(y_te),)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_mlp_fits_and_predicts():
    chased = _small_chased()
    train, val, test = entity_level_split(chased)
    X_tr, y_tr = _flat_matrix(train)
    X_te, y_te = _flat_matrix(test)
    model = _build_mlp()
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    assert probs.shape == (len(y_te),)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_all_models_beat_random_on_small_dataset():
    """
    A sanity check: every model should score above 0.50 AUC even on a 500-
    customer subset. If any fails this it's a sign of a feature bug, not
    just noise — the aging-bucket signal alone is strong enough to be found.
    """
    chased = _small_chased()
    train, _, test = entity_level_split(chased)
    X_tr, y_tr = _flat_matrix(train)
    X_te, y_te = _flat_matrix(test)

    for name, model in [("Baseline", _build_baseline()), ("GBM", _build_gbm()), ("MLP", _build_mlp())]:
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, probs)
        assert auc > 0.50, f"{name} failed random baseline: AUC={auc:.4f}"
