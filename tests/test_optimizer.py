import numpy as np
import pandas as pd
import pytest

from quant_stack.backtest.schedule import build_schedule
from quant_stack.config import OptimizationConfig
from quant_stack.data.pipeline import compute_returns
from quant_stack.forecasting.baseline import BaselineForecaster
from quant_stack.optimization.blend import benchmark_weights, blend_weights
from quant_stack.optimization.portfolio import PortfolioOptimizer
from quant_stack.optimization.tuning import tune_hyperparameters

SECTORS = {
    "AAPL": "Tech",
    "MSFT": "Tech",
    "GOOGL": "Tech",
    "AMZN": "Consumer",
    "TSLA": "Consumer",
    "JPM": "Financials",
}


def _returns(n=400, k=6, seed=0, mu=0.0006):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(mu, 0.012, size=(n, k)),
        index=pd.bdate_range("2022-01-03", periods=n),
        columns=list(SECTORS)[:k],
    )


def test_weights_valid_and_bounded():
    cfg = OptimizationConfig(max_asset_weight=0.35)
    w = PortfolioOptimizer(cfg).optimize(_returns())
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-9).all() and (w <= 0.35 + 1e-6).all()


def test_sector_constraints_are_actually_enforced():
    cfg = OptimizationConfig(
        max_asset_weight=0.6, sector_max={"Tech": 0.30}, sector_min={"Financials": 0.20}
    )
    for seed in range(5):
        w = PortfolioOptimizer(cfg, SECTORS).optimize(_returns(seed=seed))
        assert w[["AAPL", "MSFT", "GOOGL"]].sum() <= 0.30 + 1e-6
        assert w["JPM"] >= 0.20 - 1e-6


def test_per_asset_change_cap_is_respected():
    cfg = OptimizationConfig(max_asset_weight=0.6, max_weight_change=0.05)
    prev = benchmark_weights(list(SECTORS))
    w = PortfolioOptimizer(cfg).optimize(_returns(), previous_weights=prev)
    assert (w - prev).abs().max() <= 0.05 + 1e-6


def test_all_negative_returns_fall_back_instead_of_crashing():
    opt = PortfolioOptimizer(OptimizationConfig())
    w = opt.optimize(_returns(mu=-0.004))
    assert opt.last_method in {"min_risk", "min_risk_no_turnover_cap"}
    assert np.isclose(w.sum(), 1.0)


def test_blend_extremes_and_sum():
    assets = ["A", "B", "C", "D"]
    bench = benchmark_weights(assets)
    opt = pd.Series([0.7, 0.1, 0.1, 0.1], index=assets)
    pd.testing.assert_series_equal(blend_weights(opt, bench, 0.0), bench, check_names=False)
    pd.testing.assert_series_equal(blend_weights(opt, bench, 1.0), opt, check_names=False)
    mid = blend_weights(opt, bench, 0.5)
    assert np.isclose(mid.sum(), 1.0) and np.isclose(mid["A"], 0.475)
    with pytest.raises(ValueError):
        blend_weights(opt, bench, 1.5)


def test_forecaster_noise_vs_signal():
    f = BaselineForecaster()
    noise = f.forecast(_returns(mu=0.0)).confidence
    strong = _returns(seed=3)
    strong = strong.assign(**{c: strong[c] + 0.004 for c in strong.columns})
    assert f.forecast(strong).confidence > noise
    assert f.forecast(strong).confidence == pytest.approx(1.0)
    assert 0.0 <= noise <= 1.0


def test_schedule_has_no_lookahead(cfg, prices):
    """Changing prices AFTER a decision date must not change that decision."""
    base = build_schedule(prices, cfg)
    cut = base[10].date
    tampered = prices.copy()
    after = tampered.index > cut
    tampered.loc[after] = tampered.loc[after] * 3.0
    alt = build_schedule(tampered, cfg)
    for a, b in zip(base[:11], alt[:11], strict=True):
        assert a.date == b.date
        pd.testing.assert_series_equal(a.weights, b.weights, atol=1e-9)


def test_schedule_weights_are_valid(cfg, prices):
    sched = build_schedule(prices, cfg)
    assert len(sched) >= 20
    for d in sched:
        assert np.isclose(d.weights.sum(), 1.0, atol=1e-6)
        assert (d.weights >= -1e-9).all()
        assert 0.0 <= d.confidence <= 1.0
    assert sched[0].date == compute_returns(prices).index[cfg.optimization.train_window - 1]


def test_tuning_returns_a_choice():
    r = _returns(n=700, k=4, seed=1)
    best = tune_hyperparameters(
        r,
        OptimizationConfig(max_asset_weight=0.6),
        l2_grid=(0.001, 0.1),
        risk_grid=("VARIANCE", "CVAR"),
    )
    assert set(best) == {"l2_coef", "risk_measure"}
    with pytest.raises(ValueError):
        tune_hyperparameters(r.iloc[:100], OptimizationConfig(max_asset_weight=0.6))
