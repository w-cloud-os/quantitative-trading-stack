# Migrating from the first version

| Old file | New location | What changed |
|---|---|---|
| `portfolio.py` | `src/quant_stack/optimization/portfolio.py` | Driven by config; correct group format; per-asset turnover cap wired via `previous_weights`; fallback chain |
| `pipeline.py` | `src/quant_stack/data/pipeline.py` | No `bfill` (look-ahead), no zero-filled returns; data fetching moved to `providers.py` |
| `cache.py` | `src/quant_stack/data/cache.py` | Cache key hashes every fetch parameter |
| `connectors.py` | removed for now | Databento/Alpaca adapters were untested; re-add behind the `PriceProvider` protocol |
| `strategy.py` | `src/quant_stack/trading/strategy.py` | Real Nautilus API; equity includes unrealized P&L; slicing; kill-switch flattens correctly |
| `engine.py` | `src/quant_stack/trading/engine.py` | Actually loads instruments and data, runs, and returns results |
| `app.py` | `src/quant_stack/dashboard/app.py` | Fake logs/status removed; shows real simulation output |
| `test_strategy.py` | `tests/` (7 files) | Tests exercise real code; no network needed |
| `aggresive_tech_momentum.yml` etc. | `configs/*.yaml` | New schema; unknown keys rejected; crypto preset dropped (no crypto data path) |
| `Dockerfile`, `docker-compose.*` | same names | Python 3.12, runs the dashboard, named cache volume, non-root user |

Delete from Git: all `*.pyc` files, the old flat files above, and the `.pyc` copies of `cache`/`portfolio`.
