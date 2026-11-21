# Quant Stack

An educational **paper-trading research stack**: walk-forward portfolio optimization
([skfolio](https://github.com/skfolio/skfolio)) executed through an event-driven simulator
([NautilusTrader](https://github.com/nautechsystems/nautilus_trader)), with a confidence-weighted
forecast layer and hard risk limits.

> **Not financial advice. Paper/backtest only.** There is no live-trading mode (the config schema
> rejects `environment: live`). Nothing in this repository is evidence of a profitable strategy:
> the default demo runs on *synthetic random prices*, where returns are independent over time and only small random per-asset drifts differ, so a forecaster has almost nothing to find.

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
| Oversight | Drawdown kill-switch: cancel orders, flatten everything, halt (latched by default; optional cool-down re-arm). Pure logic lives in `risk.py` and is unit-tested separately. | `risk.py` |

No LLM is anywhere in the order path.

## Quick start

Requires **Python 3.12+** (NautilusTrader does not support 3.11).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest                                                  # 93 tests, includes the engine cross-check
quant-stack backtest --config configs/paper_trading.yaml --engine both --end 2024-06-28
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

## Screenshots

All screenshots use **synthetic random prices** (seed 42, `paper_trading` preset, 10% kill-switch), so the results say nothing about real markets. The benchmark is an equal-weight portfolio run through the same simulator as the strategy (same fees, order slicing and kill-switch), rebalanced on the same dates.

![Setup](docs/screenshots/setup.png)
![Strategy vs equal-weight benchmark](docs/screenshots/performance.png)
![Metrics](docs/screenshots/metrics.png)
![Kill-switch trip](docs/screenshots/kill-switch.png)
![Drawdown](docs/screenshots/drawdown.png)
![Allocation over time](docs/screenshots/allocation.png)

## What this does and doesn't do

**Does:**
- Optimizes portfolio weights (skfolio)
- Simulates execution (NautilusTrader)
- Applies fees and a drawdown kill-switch
- Compares against an equal-weight benchmark

**Doesn't:**
- Place live orders
- Use real market data by default
- Demonstrate a profitable strategy

## Seed sweep (synthetic data)

```bash
quant-stack sweep --preset aggressive_growth --seeds 200 --workers 2
quant-stack sweep --preset paper_trading --seeds 200 --workers 2
```

The sweep runs the same reference backtest on 200 different random price histories (seeds 0 to 199, fixed end date, so the same command always gives the same CSV) and compares the strategy with the equal-weight benchmark. It writes a CSV and a histogram to `results/`, and the printed summary splits the seeds by whether the kill-switch tripped for the strategy, the benchmark, both or neither. The plot needs matplotlib, which is part of the `dev` extra.

| | `aggressive_growth` (15% kill-switch) | `paper_trading` (10% kill-switch) |
|---|---|---|
| Mean difference (strategy minus benchmark) | -0.23% (95% CI -0.65% to +0.20%) | -0.43% (95% CI -1.38% to +0.53%) |
| Median difference | -0.03% | 0.00% |
| Strategy beats benchmark | 46.0% of seeds | 48.5% of seeds |
| Strategy hit the kill-switch | 13.5% of seeds | 74.0% of seeds |
| Neither tripped | 173 seeds, mean +0.06%, std 1.44 pts | 51 seeds, mean -0.42%, std 3.38 pts |
| Both tripped | 24 seeds, mean -0.27%, std 0.73 pts | 146 seeds, mean +0.08%, std 2.94 pts |
| Only the strategy tripped | 3 seeds, mean -16.38% | 2 seeds, mean -54.22% |
| Only the benchmark tripped | 0 seeds | 1 seed, +32.35% |
| Extra fees paid by the strategy per run | about $271 | about $651 |

- In both presets the confidence interval for the mean difference includes zero: there is no detectable difference between the optimizer's weights and equal weights on random prices.
- The kill-switch shapes the tails. At the 10% limit the strategy trips in 74% of random histories, and once it trips it stays in cash for the rest of the run.
- The large gaps come from kill-switch timing. In the 3 seeds per preset where only one side tripped, the side that did not trip stayed 0.02 to 1.9 points below its limit, so a tiny difference at the threshold decided whether a portfolio sat in cash for the rest of the run. In `paper_trading`, three more large gaps (10 to 24 points) come from seeds where both tripped but 169 to 308 days apart. The gaps go both ways, and there are too few of them to say either side is favoured. Setting the one-sided seeds aside, the mean difference is within 0.1 points of zero in both presets.

![Seed sweep histogram, aggressive_growth](results/seed_sweep_aggressive_growth.png)
![Seed sweep histogram, paper_trading](results/seed_sweep_paper_trading.png)

*Return difference per seed (strategy minus benchmark), as a fraction: -0.1 means the strategy finished 10 points behind. The distant bars are mostly kill-switch timing: seeds where only one side tripped, or where both tripped months apart. In `aggressive_growth` the one far bar is such a seed. In `paper_trading`, 6 of the 7 seeds more than 10 points from zero are of this kind, and the seventh is a high-return path where neither tripped.*

These are results on synthetic random prices. They say nothing about real markets.

## Real-data windows (historical prices)

```bash
quant-stack windows --preset paper_trading --provider yfinance --start 2010-07-01 --end 2026-11-18 --step-months 6
quant-stack windows --preset paper_trading --provider yfinance --start 2010-07-01 --end 2026-11-18 --step-months 6 --max-drawdown 0.30
```

The strategy and the equal-weight benchmark are run over rolling windows of real Yahoo Finance history (the `paper_trading` tickers; one download, a fresh run per window, 5 bps fees). Windows overlap, so they are not independent: only the greedy set of non-overlapping windows (7 of 27) is summarized, and no p-values are reported. Run on 2026-11-19 with data through 2026-11-18; Yahoo's adjusted history can be revised, so later runs may differ slightly.

| | 10% kill-switch (preset) | 30% kill-switch (`--max-drawdown 0.30`) |
|---|---|---|
| Windows where the strategy hit the kill-switch | 27 of 27 | 8 of 27 |
| Strategy beat benchmark (7 non-overlapping windows) | 5 of 7 | 4 of 7 |
| Mean difference (strategy minus benchmark) | -0.19% | +1.18% |
| Median difference | +0.13% | +0.23% |
| Range across the 7 windows | -5.96% to +2.36% | -5.59% to +8.00% |

- No consistent edge. The mean difference changes sign with the kill-switch setting, the medians are close to zero, and single windows differ by several points in both directions. With 7 windows this does not support a claim that the optimizer helps.
- The 10% latched kill-switch dominates that run. It fired in all 27 windows, for the strategy and the benchmark, and the strategy spent a median of about 80% of its evaluation period in cash, so most of what it measures is when each portfolio went to cash. The 30% run is the better test of the optimizer itself.
- Absolute returns are inflated by survivorship bias: the tickers are today's well-known large caps, several of them among the biggest winners of 2010 to 2026. Compare the strategy with the benchmark, not either with the market.

This is a historical backtest with close-to-close fills and a proportional fee. It is not evidence of a profitable strategy.

## Kill-switch re-arm (optional)

By default the drawdown kill-switch latches: after a trip the portfolio stays in cash for the rest of the run. Setting `rearm_after_bars: N` under `execution.risk_limits` in a preset (default `null`, meaning latched) re-arms it N bars after a trip: the high-water mark resets to the current equity and trading resumes at the next scheduled rebalance. The rule is implemented identically in the reference simulator and the NautilusTrader strategy, and the parity test covers it. For the real-data windows command use `--rearm-after-bars N`.

Real-data windows with the `paper_trading` tickers and its 10% limit (27 windows, 7 of them non-overlapping; gap = strategy minus benchmark):

| | Latched | Re-arm after 21 bars | After 63 bars | After 126 bars |
|---|---|---|---|---|
| Strategy beat benchmark (7 non-overlapping windows) | 5 | 6 | 5 | 4 |
| Mean gap | -0.19% | +5.45% | +4.99% | -0.30% |
| Median gap | +0.13% | +5.22% | +1.26% | +0.22% |
| Strategy halts in those 7 windows (total) | 7 | 22 | 19 | 14 |
| Median share of the evaluated period in cash (all 27 windows) | 80% | 13% | 37% | 52% |

What the numbers do and do not show:

- Re-arm trades protection for participation. Time in cash falls, but halts rise from one per window to two or three, and to seven in the window ending 2023-11 (the 2022 bear market) at 21 bars: repeated stop-outs.
- No cool-down is best. In the window ending 2021-11 (the COVID crash and rebound) the gap was -5.96 points latched, +11.63 at 21 bars, +20.51 at 63 bars and -11.17 at 126 bars: a 32-point swing from the cool-down length alone, which is re-entry timing, not optimizer skill.
- The +5% means are carried by a few windows. At 21 bars three windows (ending 2013-11, 2019-11, 2021-11) supply most of the total and the other four average about +1.4 points; at 63 bars two windows account for more than the whole total.
- Placebo check: the same 21-bar windows on 10 random synthetic histories gave mean gaps between -2.29% and +4.10% (average about -0.2%, standard deviation about 2). The real +5.45% is above all ten, but real stocks trip the switch nearly three times as often and these six tickers are survivors, so it is not a perfect control.
- A second universe, the `conservative_all_weather` ETFs with a 5% limit (4 non-overlapping windows out of 32), shows no such gap: mean gaps of -1.62% latched and -0.27%, -1.36% and +0.61% at 21, 63 and 126 bars.

Read together: there is no demonstrated edge. The re-arm results are hypotheses that a handful of windows, a survivor-biased universe and several tried settings cannot confirm. Reproduce with the commands in the previous section plus `--rearm-after-bars N` (Yahoo's adjusted history can be revised, so digits may shift slightly).

## How the engines are verified

`simulate()` (a fast reference simulator) and the NautilusTrader strategy implement the same rules:
decide at close *t*, trade at close *t + `execution_lag_bars`*, whole shares, fees on notional,
order slicing, drawdown kill-switch. `tests/test_nautilus_parity.py` asserts that they produce the
same equity curve (within $1 on $1,000,000), the same number of fills, the same final positions and
the same kill-switch date. If you change one engine, the test tells you when they diverge.

## Project layout

```
.github/workflows/ci.yml    ruff + full test suite on pushes to main and on pull requests
configs/                    YAML presets (validated by quant_stack/config.py)
docs/screenshots/           dashboard screenshots used in this README
results/                    seed-sweep output (CSVs are git-ignored, PNGs are kept)
src/quant_stack/
  config.py                 typed schema; unknown keys and live mode are rejected
  risk.py                   DrawdownMonitor, order slicing (pure Python)
  data/                     providers (synthetic, csv, yfinance), cache, cleaning
  forecasting/baseline.py   Forecast + Forecaster protocol
  optimization/             portfolio.py, blend.py, tuning.py (walk-forward grid search)
  backtest/                 schedule.py (walk-forward), simulate.py, metrics.py, runner.py,
                            sweep.py (seed sweeps), windows.py (real-data windows)
  trading/                  strategy.py + engine.py (NautilusTrader)
  dashboard/app.py          Streamlit UI (shows only real simulation output)
  cli.py                    quant-stack backtest | dashboard | sweep | windows
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
* **Kill-switch latches by default** for the rest of a run; the opt-in `rearm_after_bars` setting re-arms it after a cool-down. In the seed sweep
  this makes a single trip decisive for a seed's result.
* **Data licensing.** yfinance is research-only. A commercial product needs licensed data.
* **License note.** This project is BSD-3-Clause; NautilusTrader is LGPL-3.0, so review its terms
  before redistributing a bundled or modified build.

## Roadmap ideas

1. Plug a real forecaster (e.g. a time-series foundation model) into the `Forecaster` protocol and
   evaluate it out-of-sample against the equal-weight benchmark.
2. Walk-forward report: per-period returns, turnover, cost drag, benchmark-relative statistics.
3. Paper-trading adapter for a broker sandbox behind the same strategy class.
