"""Price providers. Every provider returns a wide DataFrame of adjusted closes
(DatetimeIndex, one column per ticker). Only ``SyntheticProvider`` and ``CSVProvider`` are
exercised by the automated tests; ``YFinanceProvider`` needs internet access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from quant_stack.data.cache import LocalDataCache


class PriceProvider(Protocol):
    name: str

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame: ...


class SyntheticProvider:
    """Seeded random-walk prices with a shared market factor. FAKE DATA: for demos and tests."""

    name = "synthetic"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        tickers = sorted(tickers)
        dates = pd.bdate_range(start, end)
        if len(dates) < 5:
            raise ValueError("date range too short")
        rng = np.random.default_rng(self.seed)
        n, k = len(dates), len(tickers)
        market = rng.normal(0.0003, 0.008, size=(n, 1))
        idio_vol = rng.uniform(0.006, 0.014, size=k)
        beta = rng.uniform(0.6, 1.3, size=k)
        drift = rng.uniform(0.0, 0.0004, size=k)
        rets = drift + beta * market + rng.normal(0.0, 1.0, size=(n, k)) * idio_vol
        prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
        return pd.DataFrame(prices, index=dates, columns=tickers)


class CSVProvider:
    """Reads a wide CSV (first column = date, other columns = tickers)."""

    name = "csv"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        df = pd.read_csv(self.path, index_col=0, parse_dates=True).sort_index()
        missing = set(tickers) - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing tickers: {sorted(missing)}")
        return df.loc[start:end, sorted(tickers)]


class YFinanceProvider:
    """Yahoo Finance via yfinance. Yahoo's terms limit this to personal/research use: do not
    build a paid product on it. Swap in a licensed provider before commercializing."""

    name = "yfinance"

    def __init__(self, cache: LocalDataCache | None = None) -> None:
        self.cache = cache or LocalDataCache()

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        key = LocalDataCache.make_key(
            provider=self.name, tickers=sorted(tickers), start=start, end=end
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        import yfinance as yf  # lazy: keeps the package importable without network deps

        raw = yf.download(sorted(tickers), start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError("yfinance returned no data (check tickers/dates/network)")
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(sorted(tickers)[0])
        close.index = pd.to_datetime(close.index).tz_localize(None)
        self.cache.set(key, close)
        return close


def get_provider(name: str, **kwargs: object) -> PriceProvider:
    providers = {"synthetic": SyntheticProvider, "yfinance": YFinanceProvider, "csv": CSVProvider}
    if name not in providers:
        raise ValueError(f"unknown provider '{name}'. Choose from {sorted(providers)}")
    return providers[name](**kwargs)  # type: ignore[arg-type]
