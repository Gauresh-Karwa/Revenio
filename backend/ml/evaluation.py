from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score


def print_reliability_table(y_true: np.ndarray, probs: np.ndarray, label: str, n_bins: int = 8) -> None:
    try:
        observed, predicted = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")
    except ValueError:
        print(f"  ({label}: not enough distinct probability values for a {n_bins}-bin reliability table)")
        return

    print(f"  Reliability curve — {label}:")
    print(f"  {'predicted':>10s}  {'observed':>10s}  {'gap':>8s}")
    for p, o in zip(predicted, observed):
        gap = o - p
        flag = "  <-- gap > 0.10" if abs(gap) > 0.10 else ""
        print(f"  {p:>10.3f}  {o:>10.3f}  {gap:>+8.3f}{flag}")


def print_per_code_breakdown(df_test: pd.DataFrame, y_true: np.ndarray, probs: np.ndarray, label: str) -> None:
    print(f"  Per-decline-code breakdown — {label}:")
    print(f"  {'code':>6s}  {'n':>5s}  {'auc':>6s}  {'pred_rate':>10s}  {'obs_rate':>9s}")

    codes = df_test["decline_code"].to_numpy()
    for code in sorted(set(codes)):
        mask = codes == code
        n = int(mask.sum())
        if n < 5:
            print(f"  {code:>6s}  {n:>5d}     n/a         n/a       n/a  (too few test cases)")
            continue
        y_code = y_true[mask]
        p_code = probs[mask]
        auc_str = f"{roc_auc_score(y_code, p_code):.3f}" if len(set(y_code.tolist())) > 1 else "n/a"
        print(f"  {code:>6s}  {n:>5d}  {auc_str:>6s}  {p_code.mean():>10.3f}  {y_code.mean():>9.3f}")