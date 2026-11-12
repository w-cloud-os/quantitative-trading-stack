"""Price cleaning and return computation with no look-ahead (no back-fill, no zero-fill)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skfolio.preprocessing import prices_to_returns


def clean_prices(prices: pd.DataFrame, max_ffill: int = 5) -> pd.DataFrame:
    """Sort, de-duplicate, forward-fill short gaps, drop columns with no data, and drop leading
    rows until every asset has a price. Never back-fills (that would leak future prices)."""
    if prices.empty:
        raise ValueError("no price data")
    out = prices.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(axis=1, how="all").ffill(limit=max_ffill).dropna(how="any")
    if out.empty:
        raise ValueError("no overlapping history across all tickers after cleaning")
    if (out <= 0).any().any():
        raise ValueError("non-positive prices found")
    return out


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices_to_returns(prices).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if returns.empty:
        raise ValueError("no returns could be computed")
    return returns
