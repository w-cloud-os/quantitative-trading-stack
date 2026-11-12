"""Optional hyper-parameter search with walk-forward cross-validation (no look-ahead)."""

from __future__ import annotations

import pandas as pd
from skfolio import RiskMeasure
from skfolio.model_selection import WalkForward
from sklearn.model_selection import GridSearchCV

from quant_stack.config import OptimizationConfig
from quant_stack.optimization.portfolio import PortfolioOptimizer


def tune_hyperparameters(
    returns: pd.DataFrame,
    cfg: OptimizationConfig,
    sectors: dict[str, str] | None = None,
    l2_grid: tuple[float, ...] = (0.001, 0.01, 0.1),
    risk_grid: tuple[str, ...] = ("VARIANCE", "CVAR"),
    train_size: int = 252,
    test_size: int = 63,
) -> dict:
    """Returns the best {l2_coef, risk_measure}. Needs more than ``train_size`` rows."""
    if len(returns) <= train_size + test_size:
        raise ValueError(f"need more than {train_size + test_size} rows, got {len(returns)}")
    base = PortfolioOptimizer(cfg, sectors).build_model(list(returns.columns))
    search = GridSearchCV(
        estimator=base,
        param_grid={
            "l2_coef": list(l2_grid),
            "risk_measure": [getattr(RiskMeasure, r) for r in risk_grid],
        },
        cv=WalkForward(train_size=train_size, test_size=test_size),
        n_jobs=1,
    )
    search.fit(returns)
    best = search.best_params_
    return {"l2_coef": best["l2_coef"], "risk_measure": best["risk_measure"].name}
