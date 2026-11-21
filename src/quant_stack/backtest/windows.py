"""Evaluate real-price backtests across historical windows.

The fraction in cash counts evaluation bars strictly after each halt and strictly before its next
re-arm (or through the end if there is no later re-arm), divided by all evaluation bars. Overlap
selection is greedy in chronological order from the earliest window, so it is deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_stack.backtest.runner import BacktestReport, run_reference_backtest
from quant_stack.backtest.simulate import SimResult
from quant_stack.config import StackConfig
from quant_stack.data.providers import PriceProvider


@dataclass(frozen=True)
class WindowsSummary:
    count: int
    strategy_wins: int
    mean_difference: float
    median_difference: float
    strategy_halts: int
    benchmark_halts: int


class InMemoryProvider:
    """Slice one downloaded price history inclusively, without further provider requests."""

    def __init__(self, prices: pd.DataFrame, name: str) -> None:
        self.prices = prices.sort_index()
        self.name = name

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        ordered_tickers = sorted(tickers)
        missing = set(ordered_tickers) - set(self.prices.columns)
        if missing:
            raise ValueError(f"Downloaded history is missing tickers: {sorted(missing)}")
        return self.prices.loc[start:end, ordered_tickers]


class NoSuccessfulWindowsError(ValueError):
    def __init__(self, skipped: list[tuple[str, str]], message: str) -> None:
        super().__init__(message)
        self.skipped = skipped


def _halt_date(report_result: SimResult) -> str:
    halted_on = report_result.halted_on
    return halted_on.date().isoformat() if halted_on is not None else ""


def _cash_fraction(
    equity: pd.Series,
    evaluation_start: pd.Timestamp,
    halts: list[pd.Timestamp],
    rearms: list[pd.Timestamp],
) -> float:
    """Return the fraction of evaluation bars after halts and before their next re-arm."""
    evaluation_bars = equity.loc[evaluation_start:].index
    if not halts or len(evaluation_bars) == 0:
        return 0.0
    sorted_rearms = sorted(rearms)
    in_cash = pd.Series(False, index=evaluation_bars)
    for halt in sorted(halts):
        next_rearm = next((rearm for rearm in sorted_rearms if rearm > halt), None)
        mask = evaluation_bars > halt
        if next_rearm is not None:
            mask &= evaluation_bars < next_rearm
        in_cash |= mask
    return float(in_cash.sum() / len(evaluation_bars))


def _window_row(report: BacktestReport, window_end: pd.Timestamp) -> dict[str, object]:
    if not report.strategy.fills:
        raise ValueError("no first trade")
    strategy_halt_count = len(report.strategy.halts)
    benchmark_halt_count = len(report.benchmark.halts)
    strategy_return = report.strategy_metrics["total_return"]
    benchmark_return = report.benchmark_metrics["total_return"]
    return {
        "window_end": window_end.date().isoformat(),
        "evaluation_start": report.evaluation_start.date().isoformat(),
        "strategy_total_return": strategy_return,
        "strategy_cagr": report.strategy_metrics["cagr"],
        "strategy_sharpe": report.strategy_metrics["sharpe"],
        "strategy_max_drawdown": report.strategy_metrics["max_drawdown"],
        "strategy_kill_switch_date": _halt_date(report.strategy),
        "strategy_halt_count": strategy_halt_count,
        "strategy_fraction_in_cash_after_halt": _cash_fraction(
            report.strategy.equity,
            report.evaluation_start,
            report.strategy.halts,
            report.strategy.rearms,
        ),
        "strategy_fees_paid": report.strategy.fees,
        "benchmark_total_return": benchmark_return,
        "benchmark_cagr": report.benchmark_metrics["cagr"],
        "benchmark_sharpe": report.benchmark_metrics["sharpe"],
        "benchmark_max_drawdown": report.benchmark_metrics["max_drawdown"],
        "benchmark_kill_switch_date": _halt_date(report.benchmark),
        "benchmark_halt_count": benchmark_halt_count,
        "benchmark_fraction_in_cash_after_halt": _cash_fraction(
            report.benchmark.equity,
            report.evaluation_start,
            report.benchmark.halts,
            report.benchmark.rearms,
        ),
        "benchmark_fees_paid": report.benchmark.fees,
        "return_difference": strategy_return - benchmark_return,
    }


def _mark_independent(results: pd.DataFrame) -> pd.DataFrame:
    ordered = results.sort_values("window_end", kind="stable").reset_index(drop=True)
    independent: list[bool] = []
    last_selected_end: pd.Timestamp | None = None
    for row in ordered.itertuples(index=False):
        evaluation_start = pd.Timestamp(row.evaluation_start)
        window_end = pd.Timestamp(row.window_end)
        selected = last_selected_end is None or evaluation_start > last_selected_end
        independent.append(selected)
        if selected:
            last_selected_end = window_end
    ordered["independent"] = independent
    return ordered


def _summarize(results: pd.DataFrame) -> WindowsSummary:
    selected = results.loc[results["independent"]]
    return WindowsSummary(
        count=len(selected),
        strategy_wins=int((selected["return_difference"] > 0).sum()),
        mean_difference=float(selected["return_difference"].mean()),
        median_difference=float(selected["return_difference"].median()),
        strategy_halts=int(selected["strategy_halt_count"].sum()),
        benchmark_halts=int(selected["benchmark_halt_count"].sum()),
    )


def run_window_evaluation(
    cfg: StackConfig,
    provider: PriceProvider,
    start: str,
    end: str,
    step_months: int = 6,
    max_drawdown: float | None = None,
    results_dir: str | Path = "results",
    output_name: str | None = None,
    rearm_after_bars: int | None = None,
) -> tuple[pd.DataFrame, WindowsSummary, Path, list[tuple[str, str]]]:
    """Run eligible windows from one full-range download and write the per-window CSV."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts >= end_ts:
        raise ValueError("start must be earlier than end")
    if step_months < 1:
        raise ValueError("step_months must be at least 1")
    if rearm_after_bars is not None and (
        isinstance(rearm_after_bars, bool)
        or not isinstance(rearm_after_bars, int)
        or rearm_after_bars < 1
    ):
        raise ValueError("rearm_after_bars must be at least 1")
    preset_name = output_name or cfg.preset_name
    if not re.fullmatch(r"[A-Za-z0-9_-]+", preset_name):
        raise ValueError("preset_name must contain only letters, numbers, underscores, and hyphens")

    run_cfg = cfg
    if max_drawdown is not None or rearm_after_bars is not None:
        run_cfg = cfg.model_copy(deep=True)
    if max_drawdown is not None:
        if not 0 < max_drawdown < 1:
            raise ValueError("max_drawdown must be strictly between 0 and 1")
        run_cfg.execution.risk_limits.max_drawdown_pct = max_drawdown
    if rearm_after_bars is not None:
        run_cfg.execution.risk_limits.rearm_after_bars = rearm_after_bars

    download_end = (end_ts + pd.Timedelta(days=1)).date().isoformat()
    prices = provider.fetch(sorted(run_cfg.data.tickers), start_ts.date().isoformat(), download_end)
    in_memory = InMemoryProvider(prices, getattr(provider, "name", type(provider).__name__))

    window_ends: list[pd.Timestamp] = []
    step = 0
    while True:
        window_end = end_ts - pd.DateOffset(months=step * step_months)
        first_lookback_day = pd.bdate_range(end=window_end, periods=run_cfg.data.lookback_days)[0]
        if first_lookback_day < start_ts:
            break
        window_ends.append(window_end)
        step += 1
    if not window_ends:
        raise NoSuccessfulWindowsError([], "no window has a full lookback after --start")

    rows: list[dict[str, object]] = []
    skipped: list[tuple[str, str]] = []
    for window_end in window_ends:
        window_label = window_end.date().isoformat()
        try:
            report = run_reference_backtest(run_cfg, in_memory, window_end)
            rows.append(_window_row(report, window_end))
        except Exception as exc:
            skipped.append((window_label, f"{type(exc).__name__}: {exc}"))
    if not rows:
        raise NoSuccessfulWindowsError(skipped, "no windows succeeded")

    results = _mark_independent(pd.DataFrame(rows))
    summary = _summarize(results)
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_dd{max_drawdown * 100:g}" if max_drawdown is not None else ""
    if rearm_after_bars is not None:
        suffix += f"_rearm{rearm_after_bars}"
    csv_path = output_dir / f"windows_{preset_name}_{in_memory.name}{suffix}.csv"
    results.to_csv(csv_path, index=False, float_format="%.12g", lineterminator="\n")
    return results, summary, csv_path, skipped
