from pathlib import Path

import pandas as pd

from quant_stack.backtest.sweep import SweepSummary, run_seed_sweep
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
    assert "vectorized reference simulator" in capsys.readouterr().out
