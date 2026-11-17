"""One entry point shared by the CLI and the dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_stack.backtest.metrics import compute_metrics
from quant_stack.backtest.schedule import Decision, benchmark_schedule, build_schedule
from quant_stack.backtest.simulate import SimResult, simulate
from quant_stack.config import StackConfig
from quant_stack.data.pipeline import clean_prices
from quant_stack.data.providers import PriceProvider


@dataclass
class BacktestReport:
    prices: pd.DataFrame
    schedule: list[Decision]
    strategy: SimResult
    benchmark: SimResult
    strategy_metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    provider: str
    evaluation_start: pd.Timestamp  # first trade; the idle warm-up before it is excluded


def run_reference_backtest(
    cfg: StackConfig, provider: PriceProvider, end: str | pd.Timestamp | None = None
) -> BacktestReport:
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    start_ts = pd.bdate_range(end=end_ts, periods=cfg.data.lookback_days)[0]
    prices = clean_prices(
        provider.fetch(cfg.data.tickers, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
    )
    schedule = build_schedule(prices, cfg)
    strat = simulate(prices, schedule, cfg.execution)
    bench = simulate(prices, benchmark_schedule(schedule), cfg.execution)
    # Both curves sit in cash while the optimizer warms up; measuring that idle time would dilute
    # CAGR/vol/Sharpe. Evaluate from the first trade (same start for strategy and benchmark).
    start = strat.fills[0]["date"] if strat.fills else prices.index[0]
    return BacktestReport(
        prices=prices,
        schedule=schedule,
        strategy=strat,
        benchmark=bench,
        strategy_metrics=compute_metrics(strat.equity.loc[start:]),
        benchmark_metrics=compute_metrics(bench.equity.loc[start:]),
        provider=getattr(provider, "name", type(provider).__name__),
        evaluation_start=start,
    )
