from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from quant_stack.backtest.runner import BacktestReport, run_reference_backtest
from quant_stack.backtest.simulate import SimResult
from quant_stack.backtest.windows import (
    InMemoryProvider,
    _cash_fraction,
    _mark_independent,
    _summarize,
    _window_row,
    run_window_evaluation,
)
from quant_stack.cli import main
from quant_stack.data.providers import SyntheticProvider

START = "2015-01-01"
END = "2023-01-01"
WINDOW_COLUMNS = {
    "window_end",
    "evaluation_start",
    "strategy_total_return",
    "strategy_cagr",
    "strategy_sharpe",
    "strategy_max_drawdown",
    "strategy_kill_switch_date",
    "strategy_halt_count",
    "strategy_fraction_in_cash_after_halt",
    "strategy_fees_paid",
    "benchmark_total_return",
    "benchmark_cagr",
    "benchmark_sharpe",
    "benchmark_max_drawdown",
    "benchmark_kill_switch_date",
    "benchmark_halt_count",
    "benchmark_fraction_in_cash_after_halt",
    "benchmark_fees_paid",
    "return_difference",
    "independent",
}


def _short_config(cfg):
    short = cfg.model_copy(deep=True)
    short.data.lookback_days = 90
    short.optimization.train_window = 60
    return short


def _history(cfg):
    return SyntheticProvider(31).fetch(
        cfg.data.tickers, START, (pd.Timestamp(END) + pd.Timedelta(days=1)).date().isoformat()
    )


def test_mark_independent_rejects_overlaps_and_touching_windows():
    rows = pd.DataFrame(
        [
            {"window_end": "2023-01-10", "evaluation_start": "2023-01-01"},
            {"window_end": "2023-01-20", "evaluation_start": "2023-01-05"},
            {"window_end": "2023-01-30", "evaluation_start": "2023-01-10"},
            {"window_end": "2023-02-10", "evaluation_start": "2023-01-11"},
            {"window_end": "2023-02-20", "evaluation_start": "2023-02-01"},
        ]
    )

    marked = _mark_independent(rows)

    assert marked["independent"].tolist() == [True, False, False, True, False]


def test_cash_fraction_counts_bars_strictly_after_halt():
    equity = pd.Series(
        range(10), index=pd.date_range("2023-01-02", periods=10, freq="D"), dtype=float
    )
    evaluation_start = equity.index[0]

    assert _cash_fraction(equity, evaluation_start, [equity.index[3]], []) == pytest.approx(0.6)
    assert _cash_fraction(equity, evaluation_start, [], []) == 0.0
    assert _cash_fraction(equity, evaluation_start, [equity.index[-1]], []) == 0.0


def test_cash_fraction_counts_bars_until_matching_rearm():
    equity = pd.Series(
        range(20), index=pd.date_range("2023-01-02", periods=20, freq="D"), dtype=float
    )
    halts = [equity.index[5], equity.index[12]]
    rearms = [equity.index[8]]

    assert _cash_fraction(equity, equity.index[0], halts, rearms) == pytest.approx(9 / 20)
    assert _cash_fraction(equity, equity.index[6], halts, rearms) == pytest.approx(9 / 14)


def test_cash_fraction_counts_bars_strictly_between_halt_and_last_bar_rearm():
    equity = pd.Series(
        range(20), index=pd.date_range("2023-01-02", periods=20, freq="D"), dtype=float
    )

    assert _cash_fraction(
        equity, equity.index[0], [equity.index[5]], [equity.index[-1]]
    ) == pytest.approx(13 / 20)


def test_window_evaluation_marks_non_overlapping_windows_and_matches_direct_run(
    cfg, tmp_path: Path
):
    short = _short_config(cfg)
    original_rearm = short.execution.risk_limits.rearm_after_bars
    history = _history(short)
    provider = InMemoryProvider(history, "synthetic")

    results, summary, csv_path, skipped = run_window_evaluation(
        short, provider, START, END, step_months=24, results_dir=tmp_path
    )

    assert len(results) == 4
    assert not skipped
    assert WINDOW_COLUMNS == set(results.columns)
    assert results["independent"].all()
    assert summary.count == 4
    independent = results.loc[results["independent"]].reset_index(drop=True)
    for i in range(1, len(independent)):
        assert pd.Timestamp(independent.loc[i, "evaluation_start"]) > pd.Timestamp(
            independent.loc[i - 1, "window_end"]
        )
    assert csv_path.is_file()
    assert csv_path.name == "windows_paper_trading_synthetic.csv"
    assert short.execution.risk_limits.rearm_after_bars == original_rearm
    selected = results.iloc[-1]
    direct = run_reference_backtest(short, provider, selected["window_end"])
    assert selected["evaluation_start"] == direct.evaluation_start.date().isoformat()
    assert selected["strategy_total_return"] == pytest.approx(
        direct.strategy_metrics["total_return"]
    )
    assert selected["benchmark_total_return"] == pytest.approx(
        direct.benchmark_metrics["total_return"]
    )
    assert selected["strategy_fees_paid"] == pytest.approx(direct.strategy.fees)
    assert selected["benchmark_fees_paid"] == pytest.approx(direct.benchmark.fees)


@pytest.mark.parametrize(
    ("overrides", "expected_name"),
    [
        ({}, "windows_paper_trading_synthetic.csv"),
        ({"max_drawdown": 0.30}, "windows_paper_trading_synthetic_dd30.csv"),
        ({"rearm_after_bars": 63}, "windows_paper_trading_synthetic_rearm63.csv"),
        (
            {"max_drawdown": 0.30, "rearm_after_bars": 63},
            "windows_paper_trading_synthetic_dd30_rearm63.csv",
        ),
    ],
)
def test_window_evaluation_filename_includes_overrides(
    cfg, tmp_path: Path, overrides: dict[str, float | int], expected_name: str
):
    short = _short_config(cfg)
    history = SyntheticProvider(31).fetch(short.data.tickers, "2022-01-01", "2022-09-01")

    _, _, csv_path, skipped = run_window_evaluation(
        short,
        InMemoryProvider(history, "synthetic"),
        "2022-01-03",
        "2022-08-31",
        step_months=24,
        results_dir=tmp_path,
        **overrides,
    )

    assert not skipped
    assert csv_path.name == expected_name
    assert csv_path.is_file()


def test_window_evaluation_halts_and_summarizes_only_independent_windows(cfg, tmp_path: Path):
    short = _short_config(cfg)
    short.data.lookback_days = 180
    start, end = "2021-08-02", "2022-12-30"
    history = SyntheticProvider(31).fetch(
        short.data.tickers,
        start,
        (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
    )

    results, summary, _, skipped = run_window_evaluation(
        short,
        InMemoryProvider(history, "synthetic"),
        start,
        end,
        step_months=4,
        max_drawdown=0.02,
        results_dir=tmp_path,
    )

    assert not skipped
    assert len(results) == 3
    assert results["strategy_kill_switch_date"].ne("").any()
    fractions = results["strategy_fraction_in_cash_after_halt"]
    assert fractions.between(0, 1, inclusive="neither").any()
    independent = results.loc[results["independent"]]
    assert summary.strategy_halts == int(independent["strategy_halt_count"].sum())
    assert summary.strategy_halts <= results["strategy_halt_count"].sum()


def test_window_evaluation_rearm_override_is_copied_and_counts_all_halts(
    cfg, tmp_path: Path, monkeypatch
):
    short = _short_config(cfg)
    short_original_rearm = short.execution.risk_limits.rearm_after_bars
    short_original_drawdown = short.execution.risk_limits.max_drawdown_pct
    provider = InMemoryProvider(_history(short), "synthetic")
    import quant_stack.backtest.windows as windows

    seen_configs = []
    original_run = run_reference_backtest

    def capture_run(run_cfg, in_memory, end):
        seen_configs.append(run_cfg)
        return original_run(run_cfg, in_memory, end)

    monkeypatch.setattr(windows, "run_reference_backtest", capture_run)
    results, summary, _, skipped = run_window_evaluation(
        short,
        provider,
        START,
        END,
        step_months=24,
        max_drawdown=0.25,
        rearm_after_bars=3,
        results_dir=tmp_path,
    )

    assert not skipped
    assert seen_configs
    assert all(config.execution.risk_limits.rearm_after_bars == 3 for config in seen_configs)
    assert all(config.execution.risk_limits.max_drawdown_pct == 0.25 for config in seen_configs)
    assert short.execution.risk_limits.rearm_after_bars == short_original_rearm
    assert short.execution.risk_limits.max_drawdown_pct == short_original_drawdown
    assert "strategy_halt_count" in results
    assert "benchmark_halt_count" in results
    independent = results.loc[results["independent"]]
    assert summary.strategy_halts == int(independent["strategy_halt_count"].sum())
    assert summary.benchmark_halts == int(independent["benchmark_halt_count"].sum())


def test_window_row_and_summary_count_multiple_halts():
    idx = pd.bdate_range("2023-01-01", periods=20)
    equity = pd.Series(100.0, index=idx)
    strategy = SimResult(
        equity=equity,
        fills=[{"date": idx[0]}],
        halted_on=idx[5],
        halts=[idx[5], idx[12]],
        rearms=[idx[8]],
    )
    benchmark = SimResult(
        equity=equity,
        fills=[{"date": idx[0]}],
        halted_on=idx[7],
        halts=[idx[7], idx[14]],
        rearms=[idx[10]],
    )
    report = BacktestReport(
        prices=equity.to_frame("A"),
        schedule=[],
        strategy=strategy,
        benchmark=benchmark,
        strategy_metrics={
            "total_return": 0.1,
            "cagr": 0.1,
            "sharpe": 1.0,
            "max_drawdown": 0.1,
        },
        benchmark_metrics={
            "total_return": 0.05,
            "cagr": 0.05,
            "sharpe": 0.5,
            "max_drawdown": 0.2,
        },
        provider="synthetic",
        evaluation_start=idx[0],
    )

    row = _window_row(report, idx[-1])
    assert row["strategy_halt_count"] == 2
    assert row["benchmark_halt_count"] == 2
    results = pd.DataFrame([{"independent": True, **row}])
    summary = _summarize(results)
    assert summary.strategy_halts == 2
    assert summary.benchmark_halts == 2


@pytest.mark.parametrize("rearm_after_bars", [0, -1, 1.5, True])
def test_window_evaluation_rejects_invalid_rearm(cfg, tmp_path: Path, rearm_after_bars):
    with pytest.raises(ValueError, match="rearm_after_bars must be at least 1"):
        run_window_evaluation(
            cfg,
            InMemoryProvider(_history(cfg), "synthetic"),
            START,
            END,
            rearm_after_bars=rearm_after_bars,
            results_dir=tmp_path,
        )


def test_window_evaluation_skips_a_failed_window(cfg, tmp_path: Path, monkeypatch):
    import quant_stack.backtest.windows as windows

    short = _short_config(cfg)
    provider = InMemoryProvider(_history(short), "synthetic")
    original_run = run_reference_backtest

    def fail_one_window(config, in_memory, end):
        if pd.Timestamp(end).date().isoformat() == END:
            raise ValueError("optimizer failed")
        return original_run(config, in_memory, end)

    monkeypatch.setattr(windows, "run_reference_backtest", fail_one_window)

    results, _, _, skipped = run_window_evaluation(
        short, provider, START, END, step_months=24, results_dir=tmp_path
    )

    assert len(results) == 3
    assert skipped == [(END, "ValueError: optimizer failed")]


def test_windows_cli_csv_and_drawdown_override_do_not_mutate_preset(
    cfg, tmp_path: Path, monkeypatch, capsys
):
    short = _short_config(cfg)
    original_drawdown = short.execution.risk_limits.max_drawdown_pct
    original_rearm = short.execution.risk_limits.rearm_after_bars
    history = _history(short)
    csv_path = tmp_path / "history.csv"
    history.to_csv(csv_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "paper_trading.yaml"
    config_path.write_text(yaml.safe_dump(short.model_dump(mode="json")), encoding="utf-8")
    monkeypatch.setenv("QUANT_STACK_CONFIG_DIR", str(config_dir))
    monkeypatch.chdir(tmp_path)
    import quant_stack.cli as cli

    original_load = cli.load_config
    loaded_configs = []

    def capture_config(path):
        loaded = original_load(path)
        loaded_configs.append(loaded)
        return loaded

    monkeypatch.setattr(cli, "load_config", capture_config)

    assert (
        main(
            [
                "windows",
                "--preset",
                "paper_trading",
                "--provider",
                "csv",
                "--csv-path",
                str(csv_path),
                "--start",
                START,
                "--end",
                END,
                "--step-months",
                "24",
                "--max-drawdown",
                "0.30",
                "--rearm-after-bars",
                "3",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "historical data (csv), not synthetic" in output
    assert "effective max_drawdown_pct: 30.00%" in output
    assert "Effective rearm_after_bars: 3" in output
    assert "Each window is a fresh run" in output
    table_header = next(line for line in output.splitlines() if "window_end" in line)
    assert table_header.split() == [
        "window_end",
        "evaluation_start",
        "strategy_total_return",
        "benchmark_total_return",
        "return_difference",
        "strategy_kill_switch_date",
        "strategy_halt_count",
        "benchmark_kill_switch_date",
        "benchmark_halt_count",
        "strategy_fraction_in_cash_after_halt",
        "independent",
    ]
    assert "windows_paper_trading_csv_dd30_rearm3.csv" in output
    assert (tmp_path / "results" / "windows_paper_trading_csv_dd30_rearm3.csv").is_file()
    assert short.execution.risk_limits.max_drawdown_pct == original_drawdown
    assert short.execution.risk_limits.rearm_after_bars == original_rearm
    assert loaded_configs[0].execution.risk_limits.max_drawdown_pct == original_drawdown
    assert loaded_configs[0].execution.risk_limits.rearm_after_bars == original_rearm


@pytest.mark.parametrize(
    "extra_args",
    [
        [
            "--provider",
            "csv",
            "--csv-path",
            "history.csv",
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-01",
        ],
        [
            "--provider",
            "csv",
            "--csv-path",
            "history.csv",
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-02",
            "--max-drawdown",
            "0",
        ],
        [
            "--provider",
            "csv",
            "--csv-path",
            "history.csv",
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-02",
            "--max-drawdown",
            "1.5",
        ],
        [
            "--provider",
            "csv",
            "--csv-path",
            "history.csv",
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-02",
            "--rearm-after-bars",
            "0",
        ],
        ["--provider", "csv", "--start", "2023-01-01", "--end", "2023-01-02"],
        [
            "--provider",
            "csv",
            "--csv-path",
            "history.csv",
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-02",
            "--preset",
            "missing_windows_test_preset",
        ],
    ],
)
def test_windows_cli_rejects_invalid_arguments_without_running_backtests(
    extra_args, monkeypatch, tmp_path
):
    monkeypatch.setenv("QUANT_STACK_CONFIG_DIR", str(tmp_path / "empty-configs"))
    with pytest.raises(SystemExit):
        main(["windows", "--preset", "paper_trading", *extra_args])


def test_windows_cli_reports_missing_yfinance_dependency(cfg, tmp_path: Path, monkeypatch, capsys):
    short = _short_config(cfg)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "paper_trading.yaml").write_text(
        yaml.safe_dump(short.model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setenv("QUANT_STACK_CONFIG_DIR", str(config_dir))

    def missing_yfinance(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'yfinance'", name="yfinance")

    import quant_stack.cli as cli

    monkeypatch.setattr(cli, "run_window_evaluation", missing_yfinance)
    result = main(
        [
            "windows",
            "--preset",
            "paper_trading",
            "--provider",
            "yfinance",
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-02",
        ]
    )

    assert result == 1
    assert capsys.readouterr().err.strip() == (
        'Error: yfinance is not installed; run pip install -e ".[yfinance]"'
    )
