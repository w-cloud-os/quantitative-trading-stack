# Quant Stack

An educational **paper-trading research stack**: walk-forward portfolio optimization
([skfolio](https://github.com/skfolio/skfolio)) executed through an event-driven simulator
([NautilusTrader](https://github.com/nautechsystems/nautilus_trader)), with a confidence-weighted
forecast layer and hard risk limits.

> **Not financial advice. Paper/backtest only.** There is no live-trading mode (the config schema
> rejects `environment: live`). Nothing in this repository is evidence of a profitable strategy:
> the default demo runs on *synthetic random prices*, where no edge exists by construction.

## Architecture: four desks, one pipeline

```
 prices ─► 1. FORECAST ─► 2. PORTFOLIO ─► blend(confidence) ─► 3. EXECUTION ─► 4. OVERSIGHT
           forecasting/    optimization/   optimization/        trading/        risk.py +
           baseline.py     portfolio.py    blend.py             strategy.py     trading/strategy.py
```

| Desk | What it does | Where |
|---|---|---|
| Forecast | Expected return **with a standard error**; reduces to one confidence number in [0, 1]. Baseline = trailing mean and t-stat (swap in any model via the `Forecaster` protocol). | `forecasting/baseline.py` |
| Portfolio | Max-ratio mean-risk optimization (CVaR / CDaR / variance), denoised covariance, shrunk mean, sector limits, per-asset turnover cap, fallback chain. | `optimization/portfolio.py` |
| Blend | `w = benchmark + confidence * (optimized - benchmark)`. Confident: tilt hard. Guessing: snap back to equal weight. | `optimization/blend.py` |
| Execution | Rebalances to target weights as whole-share market orders; orders are sliced to respect `max_order_value`. | `trading/strategy.py` |
| Oversight | Drawdown kill-switch: cancel orders, flatten everything, halt (latched). Pure logic lives in `risk.py` and is unit-tested separately. | `risk.py` |

No LLM is anywhere in the order path.

## Quick start

Requires **Python 3.12+** (NautilusTrader does not support 3.11).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest                                                  # 52 tests, includes the engine cross-check
quant-stack backtest --config configs/paper_trading.yaml --engine both
quant-stack dashboard                                   # http://127.0.0.1:8501
```

Example output (synthetic data, so the numbers are meaningless as evidence):

```
Preset: paper_trading | environment: paper | data: synthetic
NOTE: synthetic (random) prices: results say nothing about real markets.
Period: 2021-08-06 -> 2024-06-28 (756 bars, 26 rebalance decisions)
Metrics measured from first trade: 2022-07-27 (warm-up excluded)
Strategy : return  19.33% | CAGR   9.28% | vol 10.69% | Sharpe  0.88 | maxDD -10.37%
Benchmark: return  17.92% | CAGR   8.62% | vol 10.36% | Sharpe  0.85 | maxDD -10.13%  (equal-weight, same rules and costs)
Fills: 144 | fees: $2,110.92 | traded: $4,221,888
Kill-switch: TRIPPED on 2024-04-09
Nautilus: final equity $1,193,340.32 | fills 144 | max |diff| vs reference simulator $0.03
```

Real prices: `pip install -e ".[yfinance]"` then `--provider yfinance`. Yahoo's terms restrict
commercial use, so treat it as a research source only.

Docker (not exercised in CI): `docker compose up --build` serves the dashboard on port 8501;
`docker compose --profile tools run --rm backtest` runs the CLI backtest.

## How the engines are verified

`simulate()` (a fast reference simulator) and the NautilusTrader strategy implement the same rules:
decide at close *t*, trade at close *t + `execution_lag_bars`*, whole shares, fees on notional,
order slicing, drawdown kill-switch. `tests/test_nautilus_parity.py` asserts that they produce the
same equity curve (within $1 on $1,000,000), the same number of fills, the same final positions and
the same kill-switch date. If you change one engine, the test tells you when they diverge.

## Project layout

```
configs/                    YAML presets (validated by quant_stack/config.py)
src/quant_stack/
  config.py                 typed schema; unknown keys and live mode are rejected
  risk.py                   DrawdownMonitor, order slicing (pure Python)
  data/                     providers (synthetic, csv, yfinance), cache, cleaning
  forecasting/baseline.py   Forecast + Forecaster protocol
  optimization/             portfolio.py, blend.py, tuning.py (walk-forward grid search)
  backtest/                 schedule.py (walk-forward), simulate.py, metrics.py, runner.py
  trading/                  strategy.py + engine.py (NautilusTrader)
  dashboard/app.py          Streamlit UI (shows only real simulation output)
  cli.py                    quant-stack backtest | dashboard
tests/                      unit, property, parity and smoke tests
```

## Configuration notes

* `max_weight_change` maps to skfolio's `max_turnover`, which is a **per-asset** cap on
  `|new weight - previous weight|`, not total portfolio turnover. It only has effect when previous
  weights are supplied; the walk-forward loop does this for you.
* Sector groups are passed to skfolio as `{asset: [sector]}`. The reverse mapping is silently
  ignored by the library, and a test guards against that regression.
* `cash_buffer` must exceed twice `fee_rate` (validated) so a rebalance can never overspend.

## Known limitations (read before trusting any number)

* **Close-to-close bars.** Bars are built from adjusted closes (O=H=L=C) and fills happen at the
  execution bar's close. There is no intraday path, spread or market impact, only a proportional fee.
* **Survivorship and selection bias.** Presets use today's well-known tickers.
* **Baseline forecaster is simple.** It estimates confidence from trailing t-statistics; it is a
  placeholder for a real model, not a source of alpha.
* **Warm-up is excluded from metrics.** The optimizer needs `train_window` days of history, so the
  first months hold only cash; CAGR/vol/Sharpe are measured from the first trade (same start for the
  benchmark).
* **Kill-switch latches** for the rest of a run (by design); restart to re-arm.
* **Data licensing.** yfinance is research-only. A commercial product needs licensed data.
* **License note.** This project is BSD-3-Clause; NautilusTrader is LGPL-3.0, so review its terms
  before redistributing a bundled or modified build.

## Roadmap ideas

1. Plug a real forecaster (e.g. a time-series foundation model) into the `Forecaster` protocol and
   evaluate it out-of-sample against the equal-weight benchmark.
2. Walk-forward report: per-period returns, turnover, cost drag, benchmark-relative statistics.
3. Paper-trading adapter for a broker sandbox behind the same strategy class.
