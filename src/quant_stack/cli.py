"""Command line: ``quant-stack backtest --config configs/paper_trading.yaml``."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from quant_stack.backtest.runner import run_reference_backtest
from quant_stack.backtest.sweep import DEFAULT_END_DATE, run_seed_sweep
from quant_stack.config import load_config
from quant_stack.data.providers import get_provider


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _fmt(metrics: dict[str, float]) -> str:
    return (
        f"return {metrics['total_return']:>7.2%} | CAGR {metrics['cagr']:>7.2%} | "
        f"vol {metrics['ann_vol']:>6.2%} | Sharpe {metrics['sharpe']:>5.2f} | "
        f"maxDD {metrics['max_drawdown']:>7.2%}"
    )


def dashboard_command(host: str = "127.0.0.1", port: int = 8501) -> list[str]:
    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    return [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.address", host, "--server.port", str(port), "--browser.gatherUsageStats", "false",
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quant-stack", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bt = sub.add_parser("backtest", help="walk-forward backtest (paper only)")
    bt.add_argument("--config", required=True)
    bt.add_argument("--provider", default="synthetic", choices=["synthetic", "yfinance", "csv"])
    bt.add_argument("--csv-path", help="wide CSV file when --provider csv")
    bt.add_argument("--engine", default="vectorized", choices=["vectorized", "nautilus", "both"])
    bt.add_argument("--seed", type=int, default=42, help="synthetic provider seed")
    bt.add_argument("--end", help="end date YYYY-MM-DD (default: today)")
    sweep = sub.add_parser("sweep", help="compare strategy and equal-weight returns across seeds")
    sweep.add_argument("--preset", required=True, help="preset name from the configs directory")
    sweep.add_argument("--seeds", required=True, type=_positive_int, help="number of seeds to run")
    sweep.add_argument("--workers", type=_positive_int, default=1, help="parallel worker processes")
    sweep.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help=f"end date YYYY-MM-DD (default: {DEFAULT_END_DATE})",
    )
    dash = sub.add_parser("dashboard", help="launch the Streamlit research dashboard")
    dash.add_argument("--host", default="127.0.0.1")
    dash.add_argument("--port", type=int, default=8501)
    args = parser.parse_args(argv)

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper())
    if args.command == "dashboard":
        return subprocess.call(dashboard_command(args.host, args.port))
    if args.command == "sweep":
        if Path(args.preset).name != args.preset:
            parser.error("--preset must be a preset name, not a path")
        config_dir = Path(os.environ.get("QUANT_STACK_CONFIG_DIR", "configs"))
        config_path = next(
            (
                path
                for path in (config_dir / f"{args.preset}.yaml", config_dir / f"{args.preset}.yml")
                if path.is_file()
            ),
            None,
        )
        if config_path is None:
            parser.error(f"preset '{args.preset}' not found in {config_dir}")
        cfg = load_config(config_path)
        print("Backtest path: vectorized reference simulator (shared CLI/dashboard path).")
        try:
            results, summary, csv_path, png_path = run_seed_sweep(
                cfg, args.seeds, args.end_date, args.workers, output_name=args.preset
            )
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Results: {csv_path} | {png_path}")
        print(f"Return difference (strategy - benchmark), {len(results)} seeds:")
        print(f"  Mean: {summary.mean_difference:.2%}")
        print(f"  Median: {summary.median_difference:.2%}")
        print(f"  Std: {summary.std_difference:.2%}")
        print(f"  Strategy beats benchmark: {summary.strategy_win_rate:.1%}")
        if summary.confidence_interval is None:
            print(
                "  95% confidence interval for mean difference: unavailable (need at least 2 seeds)"
            )
        else:
            lower, upper = summary.confidence_interval
            print(f"  95% confidence interval for mean difference: [{lower:.2%}, {upper:.2%}]")
        print(f"  Kill-switch trip rate: {summary.kill_switch_trip_rate:.1%}")
        print("  Return difference by kill-switch outcome:")
        for label, outcome in (
            ("Neither tripped", summary.neither_tripped),
            ("Only strategy tripped", summary.only_strategy_tripped),
            ("Only benchmark tripped", summary.only_benchmark_tripped),
            ("Both tripped", summary.both_tripped),
        ):
            if outcome.count == 0:
                print(f"    {label} (n=0)")
            else:
                line = (
                    f"    {label} (n={outcome.count}): mean {outcome.mean:.2%}, "
                    f"median {outcome.median:.2%}"
                )
                if outcome.std is not None:
                    line += f", std {outcome.std:.2%}"
                print(line)
        print(f"  Mean extra fees paid by strategy over benchmark: {summary.mean_extra_fees:.2f}")
        return 0
    cfg = load_config(args.config)
    kwargs: dict = {"synthetic": {"seed": args.seed}, "csv": {"path": args.csv_path}}.get(
        args.provider, {}
    )
    provider = get_provider(args.provider, **kwargs)
    report = run_reference_backtest(cfg, provider, args.end)

    print(f"Preset: {cfg.preset_name} | environment: {cfg.environment} | data: {report.provider}")
    if report.provider == "synthetic":
        print("NOTE: synthetic (random) prices: results say nothing about real markets.")
    print(
        f"Period: {report.prices.index[0].date()} -> {report.prices.index[-1].date()} "
        f"({len(report.prices)} bars, {len(report.schedule)} rebalance decisions)"
    )
    print(f"Metrics measured from first trade: {report.evaluation_start.date()} (warm-up excluded)")
    print(f"Strategy : {_fmt(report.strategy_metrics)}")
    print(f"Benchmark: {_fmt(report.benchmark_metrics)}  (equal-weight, same rules and costs)")
    s = report.strategy
    print(f"Fills: {len(s.fills)} | fees: ${s.fees:,.2f} | traded: ${s.traded_notional:,.0f}")
    halt = s.halted_on
    status = f"TRIPPED on {halt.date()}" if halt is not None else "not tripped"
    print(f"Kill-switch: {status}")

    if args.engine in ("nautilus", "both"):
        from quant_stack.trading.engine import run_nautilus_backtest

        nt = run_nautilus_backtest(report.prices, report.schedule, cfg.execution)
        diff = float((nt.equity - s.equity).abs().max())
        print(
            f"Nautilus: final equity ${nt.equity.iloc[-1]:,.2f} | fills {len(nt.fills)} | "
            f"max |diff| vs reference simulator ${diff:,.2f}"
        )
        if diff > 1.0:
            print("WARNING: engines disagree by more than $1; investigate before trusting results.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
