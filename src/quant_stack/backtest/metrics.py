from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(equity: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Standard performance statistics from an equity curve (risk-free rate assumed 0)."""
    equity = equity.dropna()
    if len(equity) < 3:
        raise ValueError("need at least 3 equity points")
    rets = equity.pct_change().dropna()
    years = len(rets) / periods_per_year
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = (
        float(rets.mean() / rets.std(ddof=1) * np.sqrt(periods_per_year)) if rets.std() > 0 else 0.0
    )
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1 else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }
