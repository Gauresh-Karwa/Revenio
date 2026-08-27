from __future__ import annotations

import numpy as np
import pandas as pd


class RuleBasedBaseline:
    def __init__(self) -> None:
        self._code_rates: dict[str, float] = {}
        self._global_rate: float = 0.5

    def fit(self, df_train: pd.DataFrame) -> "RuleBasedBaseline":
        self._global_rate = df_train["recovered"].mean()
        self._code_rates = df_train.groupby("decline_code")["recovered"].mean().to_dict()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(
            [self._code_rates.get(code, self._global_rate) for code in df["decline_code"]]
        )