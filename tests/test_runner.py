import pandas as pd

from quant_stack.backtest.metrics import compute_metrics
from quant_stack.backtest.runner import run_reference_backtest
from quant_stack.data.providers import SyntheticProvider


def test_metrics_exclude_the_idle_warmup(cfg):
    rep = run_reference_backtest(cfg, SyntheticProvider(42), "2024-06-28")
    first_trade = rep.strategy.fills[0]["date"]
    assert rep.evaluation_start == first_trade
    # The warm-up (no trades) is a large share of the run, and it is excluded from metrics.
    assert (rep.strategy.equity.loc[:first_trade].iloc[:-1] == cfg.execution.initial_capital).all()
    expected = compute_metrics(rep.strategy.equity.loc[first_trade:])
    assert rep.strategy_metrics == expected
    full = compute_metrics(rep.strategy.equity)
    assert abs(expected["ann_vol"] - full["ann_vol"]) > 1e-4  # idle days would have diluted it


def test_strategy_and_benchmark_share_the_same_evaluation_window(cfg):
    rep = run_reference_backtest(cfg, SyntheticProvider(1), "2024-06-28")
    assert isinstance(rep.evaluation_start, pd.Timestamp)
    assert rep.benchmark.fills[0]["date"] == rep.evaluation_start
