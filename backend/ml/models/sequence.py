"""
LSTM sequence model — the fourth diagnosis-layer comparison point required
by architecture doc 6.3. Mirrors models/neural_net.py's structure (random
search + entity-aware CV, final full-data training) so this candidate is
evaluated with the exact same held-out discipline as the other three.

HONEST SCOPE NOTE (read backend/data/subscription_generator.py's retry-
sequence section and backend/ml/sequence_features.py's docstring first):
the current generator's true_recovery_probability depends on attempt
history only through a COUNT (attempt_number), which flat models already
receive. This model's ceiling on the current generator is the same oracle
AUC as the flat comparison — it is not expected to beat GBM/NN here, and
that is a legitimate, reportable finding per architecture doc 6's own
"test, don't assume" discipline, not evidence something is broken.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import loguniform, randint
from sklearn.metrics import roc_auc_score

from backend.ml.sequence_features import N_STEP_FEATURES

PARAM_DISTRIBUTIONS = {
    "hidden_size": randint(4, 33),          # 4..32
    "lr": loguniform(1e-4, 1e-1),
    "weight_decay": loguniform(1e-6, 1e-2),
}
EPOCHS_PER_FOLD = 40


def _sample_params(rng: np.random.Generator) -> dict:
    return {
        "hidden_size": int(PARAM_DISTRIBUTIONS["hidden_size"].rvs(random_state=rng)),
        "lr": float(PARAM_DISTRIBUTIONS["lr"].rvs(random_state=rng)),
        "weight_decay": float(PARAM_DISTRIBUTIONS["weight_decay"].rvs(random_state=rng)),
    }


class RetryLSTM(nn.Module):
    """
    A genuinely small sequence model, per architecture doc 6.3's own framing
    ("A SMALL sequence model") — one LSTM layer over the padded per-attempt
    step features, final hidden state fed to a single linear head.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=N_STEP_FEATURES, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        final_hidden = h_n[-1]  # (batch, hidden_size) — last layer's final hidden state
        return self.head(final_hidden).squeeze(-1)  # logits


def _train_one_model(padded, lengths, y, hidden_size, lr, weight_decay, epochs) -> RetryLSTM:
    model = RetryLSTM(hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    padded_t = torch.tensor(padded, dtype=torch.float32)
    lengths_t = torch.tensor(lengths, dtype=torch.int64)
    y_t = torch.tensor(y, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(padded_t, lengths_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()

    return model


def predict_proba(model: RetryLSTM, padded: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        padded_t = torch.tensor(padded, dtype=torch.float32)
        lengths_t = torch.tensor(lengths, dtype=torch.int64)
        logits = model(padded_t, lengths_t)
        return torch.sigmoid(logits).numpy()


def tune_lstm(
    padded: np.ndarray, lengths: np.ndarray, y: np.ndarray, groups: np.ndarray,
    n_splits: int = 5, n_iter: int = 15, seed: int = 42,
) -> tuple[dict, float]:
    """Random search with entity-aware GroupKFold CV — same protocol as tune_gbm/tune_nn."""
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(seed)
    gkf = GroupKFold(n_splits=n_splits)

    best_params = None
    best_score = -1.0

    for _ in range(n_iter):
        params = _sample_params(rng)
        fold_scores = []
        for train_idx, val_idx in gkf.split(padded, y, groups=groups):
            model = _train_one_model(
                padded[train_idx], lengths[train_idx], y[train_idx],
                params["hidden_size"], params["lr"], params["weight_decay"], EPOCHS_PER_FOLD,
            )
            preds = predict_proba(model, padded[val_idx], lengths[val_idx])
            if len(set(y[val_idx].tolist())) > 1:
                fold_scores.append(roc_auc_score(y[val_idx], preds))
        if not fold_scores:
            continue
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score


def train_final_lstm(padded: np.ndarray, lengths: np.ndarray, y: np.ndarray, params: dict, epochs: int = 150) -> RetryLSTM:
    return _train_one_model(padded, lengths, y, params["hidden_size"], params["lr"], params["weight_decay"], epochs)
