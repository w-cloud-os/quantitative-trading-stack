from pathlib import Path

import pandas as pd
import pytest

from quant_stack.backtest.sweep import SweepSummary, run_seed_sweep, summarize_sweep
from quant_stack.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _short_config(cfg):
    short = cfg.model_copy(deep=True)
    short.data.lookback_days = 90
    short.optimization.train_window = 60
    return short


def test_seed_sweep_writes_sorted_reproducible_results(cfg, tmp_path: Path):
    short = _short_config(cfg)
    results, summary, csv_path, png_path = run_seed_sweep(
        short, 3, "2024-06-28", results_dir=tmp_path
    )
    first_csv = csv_path.read_bytes()

    repeated, repeated_summary, _, _ = run_seed_sweep(short, 3, "2024-06-28", results_dir=tmp_path)

    assert results["seed"].tolist() == [0, 1, 2]
    assert repeated.equals(results)
    assert csv_path.read_bytes() == first_csv
    assert png_path.is_file()
    assert set(
        (
            "strategy_total_return",
            "benchmark_total_return",
            "return_difference",
            "strategy_sharpe",
            "benchmark_sharpe",
            "strategy_max_drawdown",
            "benchmark_max_drawdown",
            "strategy_kill_switch_tripped",
            "strategy_kill_switch_date",
            "strategy_fees_paid",
        )
    ).issubset(results.columns)
    assert summary == repeated_summary
    assert summary.confidence_interval is not None
    assert 0.0 <= summary.strategy_win_rate <= 1.0
    assert 0.0 <= summary.kill_switch_trip_rate <= 1.0


def test_seed_sweep_uses_independent_processes_when_requested(cfg, tmp_path: Path):
    results, _, _, _ = run_seed_sweep(
        _short_config(cfg), 3, "2024-06-28", workers=2, results_dir=tmp_path
    )

    assert results["seed"].tolist() == [0, 1, 2]
    assert pd.api.types.is_numeric_dtype(results["return_difference"])


def test_summarize_sweep_breaks_down_kill_switch_outcomes():
    results = pd.DataFrame(
        {
            "return_difference": [0.1, 0.3, 0.2, 0.4, 0.5],
            "strategy_kill_switch_tripped": [False, False, True, True, True],
            "benchmark_kill_switch_tripped": [False, False, False, False, True],
            "strategy_fees_paid": [2.0, 4.0, 6.0, 8.0, 10.0],
            "benchmark_fees_paid": [1.0, 1.0, 2.0, 2.0, 4.0],
        }
    )

    summary = summarize_sweep(results)

    assert summary.neither_tripped.count == 2
    assert summary.neither_tripped.mean == pytest.approx(0.2)
    assert summary.neither_tripped.median == pytest.approx(0.2)
    assert summary.neither_tripped.std == pytest.approx(0.1414213562373095)
    assert summary.only_strategy_tripped.count == 2
    assert summary.only_strategy_tripped.mean == pytest.approx(0.3)
    assert summary.only_strategy_tripped.median == pytest.approx(0.3)
    assert summary.only_strategy_tripped.std == pytest.approx(0.1414213562373095)
    assert summary.only_benchmark_tripped.count == 0
    assert summary.only_benchmark_tripped.mean is None
    assert summary.only_benchmark_tripped.median is None
    assert summary.only_benchmark_tripped.std is None
    assert summary.both_tripped.count == 1
    assert summary.both_tripped.mean == pytest.approx(0.5)
    assert summary.both_tripped.median == pytest.approx(0.5)
    assert summary.both_tripped.std is None
    assert summary.mean_extra_fees == pytest.approx(4.0)


def test_sweep_cli_uses_fixed_default_end_date_and_preset_name(monkeypatch, tmp_path, capsys):
    call = {}

    def fake_sweep(cfg, seed_count, end_date, workers, output_name):
        call.update(
            seed_count=seed_count, end_date=end_date, workers=workers, output_name=output_name
        )
        summary = SweepSummary(0.0, 0.0, 0.0, 0.0, (0.0, 0.0), 0.0)
        return (
            pd.DataFrame({"seed": [0]}),
            summary,
            tmp_path / "seed_sweep_paper_trading.csv",
            tmp_path / "seed_sweep_paper_trading.png",
        )

    monkeypatch.setenv("QUANT_STACK_CONFIG_DIR", str(ROOT / "configs"))
    monkeypatch.setattr("quant_stack.cli.run_seed_sweep", fake_sweep)

    assert main(["sweep", "--preset", "paper_trading", "--seeds", "3"]) == 0

    assert call == {
        "seed_count": 3,
        "end_date": "2026-11-17",
        "workers": 1,
        "output_name": "paper_trading",
    }
    output = capsys.readouterr().out
    assert "vectorized reference simulator" in output
    assert "Neither tripped (n=0)" in output
    assert "Only strategy tripped (n=0)" in output
    assert "Only benchmark tripped (n=0)" in output
    assert "Both tripped (n=0)" in output
