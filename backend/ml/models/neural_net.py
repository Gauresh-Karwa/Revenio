from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import roc_auc_score

from backend.ml.features import build_group_kfold
from backend.ml.progress import ProgressBar

PARAM_DISTRIBUTIONS = {
    "n_layers": randint(1, 4),             # 1..3 hidden layers
    "width_multiplier": randint(2, 9),     # hidden width = n_features * this (2x..8x)
    "lr": loguniform(1e-4, 1e-1),
    "dropout": uniform(0.0, 0.4),          # 0..0.4
    "weight_decay": loguniform(1e-6, 1e-2),
}
EPOCHS_PER_FOLD = 40


def _sample_params(rng: np.random.Generator) -> dict:
    return {
        "n_layers": int(PARAM_DISTRIBUTIONS["n_layers"].rvs(random_state=rng)),
        "width_multiplier": int(PARAM_DISTRIBUTIONS["width_multiplier"].rvs(random_state=rng)),
        "lr": float(PARAM_DISTRIBUTIONS["lr"].rvs(random_state=rng)),
        "dropout": float(PARAM_DISTRIBUTIONS["dropout"].rvs(random_state=rng)),
        "weight_decay": float(PARAM_DISTRIBUTIONS["weight_decay"].rvs(random_state=rng)),
    }


class ConfigurableMLP(nn.Module):
    def __init__(self, n_features: int, n_layers: int, width_multiplier: int, dropout: float) -> None:
        super().__init__()
        hidden_size = max(4, n_features * width_multiplier)
        layers: list[nn.Module] = []
        in_size = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            in_size = hidden_size
            hidden_size = max(4, hidden_size // 2)  # taper width with depth
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_one_model(X_train, y_train, n_features, params, epochs, bar=None, bar_suffix=""):
    model = ConfigurableMLP(n_features, params["n_layers"], params["width_multiplier"], params["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()
        if bar:
            bar.update(1, suffix=bar_suffix)

    return model


def _predict_proba(model: ConfigurableMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def tune_nn(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    n_splits: int = 5, n_iter: int = 15, seed: int = 42, show_progress: bool = True,
) -> tuple[dict, float]:
    rng = np.random.default_rng(seed)
    gkf = build_group_kfold(n_splits=n_splits)
    n_features = X.shape[1]

    total_steps = n_iter * n_splits * EPOCHS_PER_FOLD
    bar = ProgressBar(total_steps, label="NN random search  ") if show_progress else None

    best_params = None
    best_score = -1.0

    for _ in range(n_iter):
        params = _sample_params(rng)
        fold_scores = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            model = _train_one_model(
                X[train_idx], y[train_idx], n_features, params, EPOCHS_PER_FOLD,
                bar=bar, bar_suffix=f"L={params['n_layers']} w={params['width_multiplier']}x",
            )
            preds = _predict_proba(model, X[val_idx])
            fold_scores.append(roc_auc_score(y[val_idx], preds))

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score


def train_final_nn(X: np.ndarray, y: np.ndarray, params: dict, epochs: int = 150) -> ConfigurableMLP:
    return _train_one_model(X, y, X.shape[1], params, epochs)