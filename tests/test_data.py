import numpy as np
import pandas as pd
import pytest

from quant_stack.data.cache import LocalDataCache
from quant_stack.data.pipeline import clean_prices, compute_returns
from quant_stack.data.providers import CSVProvider, SyntheticProvider, get_provider


def test_synthetic_is_deterministic_and_seed_sensitive():
    a = SyntheticProvider(1).fetch(["A", "B"], "2022-01-03", "2022-06-30")
    b = SyntheticProvider(1).fetch(["B", "A"], "2022-01-03", "2022-06-30")
    c = SyntheticProvider(2).fetch(["A", "B"], "2022-01-03", "2022-06-30")
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    assert (a > 0).all().all()


def test_clean_prices_never_backfills():
    idx = pd.bdate_range("2023-01-02", periods=6)
    df = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5, 6.0], "B": [np.nan, np.nan, 10, 11, 12, 13.0]}, index=idx
    )
    out = clean_prices(df)
    assert out.index[0] == idx[2]  # rows before B existed are dropped, not filled
    assert out.loc[idx[2], "B"] == 10


def test_clean_prices_forward_fills_short_gaps_only():
    idx = pd.bdate_range("2023-01-02", periods=10)
    a = pd.Series(np.arange(1, 11, dtype=float), index=idx)
    b = a.copy()
    b.iloc[3:5] = np.nan  # 2-day gap: filled
    out = clean_prices(pd.DataFrame({"A": a, "B": b}), max_ffill=5)
    assert out.loc[idx[4], "B"] == b.iloc[2]
    gap = b.copy()
    gap.iloc[3:9] = np.nan  # 6-day gap exceeds the limit: rows are dropped, not invented
    out2 = clean_prices(pd.DataFrame({"A": a, "B": gap}), max_ffill=2)
    assert len(out2) < 10


def test_clean_prices_rejects_bad_input():
    with pytest.raises(ValueError):
        clean_prices(pd.DataFrame())
    idx = pd.bdate_range("2023-01-02", periods=3)
    with pytest.raises(ValueError):
        clean_prices(pd.DataFrame({"A": [1.0, -2.0, 3.0], "B": [1.0, 2.0, 3.0]}, index=idx))


def test_compute_returns_has_no_missing_values(prices):
    r = compute_returns(prices)
    assert not r.isna().any().any()
    assert len(r) == len(prices) - 1
    assert np.isclose(r.iloc[0, 0], prices.iloc[1, 0] / prices.iloc[0, 0] - 1)


def test_cache_key_depends_on_every_parameter(tmp_path):
    k = LocalDataCache.make_key
    base = k(provider="x", tickers=["A"], start="2020", end="2021")
    assert base == k(end="2021", start="2020", tickers=["A"], provider="x")  # order-independent
    assert base != k(provider="y", tickers=["A"], start="2020", end="2021")
    assert base != k(provider="x", tickers=["A", "B"], start="2020", end="2021")
    assert base != k(provider="x", tickers=["A"], start="2020", end="2022")


def test_cache_round_trip(tmp_path):
    cache = LocalDataCache(tmp_path)
    df = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.bdate_range("2023-01-02", periods=2))
    assert cache.get("k") is None
    cache.set("k", df)
    pd.testing.assert_frame_equal(cache.get("k"), df, check_freq=False)


def test_csv_provider_and_registry(tmp_path):
    p = tmp_path / "px.csv"
    SyntheticProvider(3).fetch(["A", "B", "C"], "2022-01-03", "2022-03-31").to_csv(p)
    out = CSVProvider(p).fetch(["B", "A"], "2022-02-01", "2022-03-01")
    assert list(out.columns) == ["A", "B"]
    with pytest.raises(ValueError):
        CSVProvider(p).fetch(["ZZZ"], "2022-01-03", "2022-03-31")
    with pytest.raises(ValueError):
        get_provider("nope")
