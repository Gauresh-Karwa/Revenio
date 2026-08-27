from __future__ import annotations

import numpy as np
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import roc_auc_score

from backend.ml.features import build_group_kfold
from backend.ml.progress import ProgressBar

PARAM_DISTRIBUTIONS = {
    "max_depth": randint(2, 9),           # 2..8
    "learning_rate": loguniform(0.01, 0.3),
    "n_estimators": randint(50, 400),
    "subsample": uniform(0.6, 0.4),        # 0.6..1.0
    "colsample_bytree": uniform(0.6, 0.4), # 0.6..1.0
    "min_child_weight": randint(1, 11),    # 1..10
    "reg_lambda": loguniform(0.1, 10.0),
}


def _sample_params(rng: np.random.Generator) -> dict:
    return {
        "max_depth": int(PARAM_DISTRIBUTIONS["max_depth"].rvs(random_state=rng)),
        "learning_rate": float(PARAM_DISTRIBUTIONS["learning_rate"].rvs(random_state=rng)),
        "n_estimators": int(PARAM_DISTRIBUTIONS["n_estimators"].rvs(random_state=rng)),
        "subsample": float(PARAM_DISTRIBUTIONS["subsample"].rvs(random_state=rng)),
        "colsample_bytree": float(PARAM_DISTRIBUTIONS["colsample_bytree"].rvs(random_state=rng)),
        "min_child_weight": int(PARAM_DISTRIBUTIONS["min_child_weight"].rvs(random_state=rng)),
        "reg_lambda": float(PARAM_DISTRIBUTIONS["reg_lambda"].rvs(random_state=rng)),
    }


def _build_model(params: dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        reg_lambda=params["reg_lambda"],
        eval_metric="logloss",
        verbosity=0,
    )


def tune_gbm(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    n_splits: int = 5, n_iter: int = 25, seed: int = 42, show_progress: bool = True,
) -> tuple[dict, float]:
    """
    Random search: n_iter combinations sampled from real ranges, each
    evaluated with n_splits-fold entity-aware CV. Returns (best_params, best_mean_auc).
    """
    rng = np.random.default_rng(seed)
    gkf = build_group_kfold(n_splits=n_splits)
    total_fits = n_iter * n_splits
    bar = ProgressBar(total_fits, label="GBM random search") if show_progress else None

    best_params = None
    best_score = -1.0

    for _ in range(n_iter):
        params = _sample_params(rng)
        fold_scores = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            model = _build_model(params)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict_proba(X[val_idx])[:, 1]
            fold_scores.append(roc_auc_score(y[val_idx], preds))
            if bar:
                bar.update(1, suffix=f"depth={params['max_depth']} lr={params['learning_rate']:.3f}")

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score


def train_final_gbm(X: np.ndarray, y: np.ndarray, params: dict) -> xgb.XGBClassifier:
    model = _build_model(params)
    model.fit(X, y)
    return model