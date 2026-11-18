"""Seed sweeps using the same reference backtest as the CLI and dashboard."""

from __future__ import annotations

import math
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from quant_stack.backtest.runner import run_reference_backtest
from quant_stack.config import StackConfig
from quant_stack.data.providers import SyntheticProvider

DEFAULT_END_DATE = "2026-11-17"


@dataclass(frozen=True)
class SweepSummary:
    mean_difference: float
    median_difference: float
    std_difference: float
    strategy_win_rate: float
    confidence_interval: tuple[float, float] | None
    kill_switch_trip_rate: float


def _run_seed(config_data: dict, seed: int, end_date: str) -> dict:
    cfg = StackConfig.model_validate(config_data)
    report = run_reference_backtest(cfg, SyntheticProvider(seed), end_date)
    strategy_halt = report.strategy.halted_on
    benchmark_halt = report.benchmark.halted_on
    strategy_return = report.strategy_metrics["total_return"]
    benchmark_return = report.benchmark_metrics["total_return"]
    return {
        "seed": seed,
        "strategy_total_return": strategy_return,
        "benchmark_total_return": benchmark_return,
        "return_difference": strategy_return - benchmark_return,
        "strategy_sharpe": report.strategy_metrics["sharpe"],
        "benchmark_sharpe": report.benchmark_metrics["sharpe"],
        "strategy_max_drawdown": report.strategy_metrics["max_drawdown"],
        "benchmark_max_drawdown": report.benchmark_metrics["max_drawdown"],
        "strategy_kill_switch_tripped": strategy_halt is not None,
        "strategy_kill_switch_date": strategy_halt.date().isoformat()
        if strategy_halt is not None
        else "",
        "benchmark_kill_switch_tripped": benchmark_halt is not None,
        "benchmark_kill_switch_date": benchmark_halt.date().isoformat()
        if benchmark_halt is not None
        else "",
        "strategy_fees_paid": report.strategy.fees,
        "benchmark_fees_paid": report.benchmark.fees,
    }


def summarize_sweep(results: pd.DataFrame) -> SweepSummary:
    differences = results["return_difference"].to_numpy(dtype=float)
    count = len(differences)
    mean = float(np.mean(differences))
    interval = None
    if count > 1:
        margin = float(t.ppf(0.975, count - 1) * np.std(differences, ddof=1) / math.sqrt(count))
        interval = (mean - margin, mean + margin)
    return SweepSummary(
        mean_difference=mean,
        median_difference=float(np.median(differences)),
        std_difference=float(np.std(differences, ddof=1)) if count > 1 else float("nan"),
        strategy_win_rate=float(np.mean(differences > 0)),
        confidence_interval=interval,
        kill_switch_trip_rate=float(results["strategy_kill_switch_tripped"].mean()),
    )


def _write_histogram(results: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Seed-sweep plots require matplotlib; install it with `pip install -e '.[dev]'`."
        ) from exc

    differences = results["return_difference"]
    fig, ax = plt.subplots()
    ax.hist(
        differences, bins=min(30, max(5, math.ceil(math.sqrt(len(differences))))), edgecolor="black"
    )
    median = float(differences.median())
    ax.axvline(median, color="crimson", linestyle="--", label=f"Median: {median:.2%}")
    ax.set(title="Strategy minus benchmark return", xlabel="Return difference", ylabel="Seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_seed_sweep(
    cfg: StackConfig,
    seed_count: int,
    end_date: str = DEFAULT_END_DATE,
    workers: int = 1,
    results_dir: str | Path = "results",
    output_name: str | None = None,
) -> tuple[pd.DataFrame, SweepSummary, Path, Path]:
    """Run seeds from zero through ``seed_count - 1`` and write stable CSV/PNG outputs."""
    if seed_count < 1:
        raise ValueError("seed_count must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    preset_name = output_name or cfg.preset_name
    if not re.fullmatch(r"[A-Za-z0-9_-]+", preset_name):
        raise ValueError("preset_name must contain only letters, numbers, underscores, and hyphens")

    config_data = cfg.model_dump(mode="python")
    seeds = range(seed_count)
    if workers == 1:
        rows = [_run_seed(config_data, seed, end_date) for seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_run_seed, repeat(config_data), seeds, repeat(end_date)))

    results = pd.DataFrame(rows).sort_values("seed", kind="stable").reset_index(drop=True)
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"seed_sweep_{preset_name}.csv"
    png_path = output_dir / f"seed_sweep_{preset_name}.png"
    results.to_csv(csv_path, index=False, float_format="%.12g", lineterminator="\n")
    _write_histogram(results, png_path)
    return results, summarize_sweep(results), csv_path, png_path
