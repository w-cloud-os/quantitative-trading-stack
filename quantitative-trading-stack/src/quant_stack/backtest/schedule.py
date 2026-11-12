"""Walk-forward target-weight schedule. Each decision only sees data up to its own date."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_stack.config import StackConfig
from quant_stack.data.pipeline import compute_returns
from quant_stack.forecasting.baseline import BaselineForecaster, Forecaster
from quant_stack.optimization.blend import benchmark_weights, blend_weights
from quant_stack.optimization.portfolio import PortfolioOptimizer


@dataclass(frozen=True)
class Decision:
    date: pd.Timestamp  # decision made using data up to and including this date's close
    weights: pd.Series  # final (blended) target weights
    optimizer_weights: pd.Series
    confidence: float
    method: str


def build_schedule(
    prices: pd.DataFrame, cfg: StackConfig, forecaster: Forecaster | None = None
) -> list[Decision]:
    returns = compute_returns(prices)
    tw = cfg.optimization.train_window
    step = cfg.execution.rebalance_interval_bars
    if len(returns) < tw + 1:
        raise ValueError(f"need at least {tw + 1} return rows, got {len(returns)}")
    if cfg.forecast.enabled and forecaster is None:
        forecaster = BaselineForecaster(cfg.forecast.window, cfg.forecast.full_confidence_tstat)
    optimizer = PortfolioOptimizer(cfg.optimization, cfg.data.sectors)
    bench = benchmark_weights(returns.columns)
    decisions: list[Decision] = []
    previous: pd.Series | None = None
    for i in range(tw - 1, len(returns), step):
        window = returns.iloc[i - tw + 1 : i + 1]
        confidence = forecaster.forecast(window).confidence if forecaster else 1.0
        opt = optimizer.optimize(window, previous)
        final = blend_weights(opt, bench, confidence)
        decisions.append(Decision(returns.index[i], final, opt, confidence, optimizer.last_method))
        previous = final
    return decisions


def benchmark_schedule(decisions: list[Decision]) -> list[Decision]:
    """Same dates, equal weights: the honest yardstick for the strategy."""
    out = []
    for d in decisions:
        b = benchmark_weights(d.weights.index)
        out.append(Decision(d.date, b, b, 0.0, "benchmark"))
    return out
