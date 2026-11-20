import numpy as np
import pandas as pd

from quant_stack.backtest.metrics import compute_metrics
from quant_stack.backtest.schedule import Decision, benchmark_schedule, build_schedule
from quant_stack.backtest.simulate import simulate


def _flat_prices(n=40, tickers=("A", "B")):
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({t: np.full(n, 100.0) for t in tickers}, index=idx)


def _decision(prices, i, weights):
    w = pd.Series(weights)
    return Decision(prices.index[i], w, w, 1.0, "test")


def test_flat_prices_only_costs_fees(cfg):
    px = _flat_prices()
    ex = cfg.execution
    res = simulate(px, [_decision(px, 5, {"A": 0.5, "B": 0.5})], ex)
    assert res.fees > 0
    assert np.isclose(res.equity.iloc[-1], ex.initial_capital - res.fees, atol=0.01)
    assert res.equity.iloc[0] == ex.initial_capital


def test_trade_happens_after_the_lag_not_before(cfg):
    px = _flat_prices()
    res = simulate(px, [_decision(px, 5, {"A": 0.5, "B": 0.5})], cfg.execution)
    first = min(f["date"] for f in res.fills)
    assert first == px.index[5 + cfg.execution.execution_lag_bars]


def test_whole_shares_and_orders_respect_max_order_value(cfg, prices):
    sched = build_schedule(prices, cfg)
    res = simulate(prices, sched, cfg.execution)
    limit = cfg.execution.risk_limits.max_order_value
    assert all(isinstance(f["qty"], int) and f["qty"] > 0 for f in res.fills)
    assert all(f["qty"] * f["price"] <= limit + 1e-6 for f in res.fills)


def test_never_spends_more_than_equity(cfg, prices):
    res = simulate(prices, build_schedule(prices, cfg), cfg.execution)
    assert (res.equity > 0).all()
    gross = sum(res.final_shares[t] * float(prices[t].round(2).iloc[-1]) for t in prices.columns)
    assert gross <= res.equity.iloc[-1] * 1.0001


def test_kill_switch_flattens_and_stays_flat(cfg):
    idx = pd.bdate_range("2023-01-02", periods=60)
    px = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
    px.loc[idx[20] :, ["A", "B"]] = 70.0  # a 30% crash on both assets
    ex = cfg.execution.model_copy(deep=True)
    ex.risk_limits.max_drawdown_pct = 0.10
    res = simulate(
        px, [_decision(px, 5, {"A": 0.5, "B": 0.5}), _decision(px, 30, {"A": 0.5, "B": 0.5})], ex
    )
    assert res.halted_on == idx[20]
    assert all(v == 0 for v in res.final_shares.values())
    assert all(f["date"] <= idx[20] for f in res.fills)  # no trades after the halt


def test_kill_switch_rearms_then_can_trip_again(cfg):
    idx = pd.bdate_range("2023-01-02", periods=15)
    px = pd.DataFrame({"A": 100.0}, index=idx)
    px.loc[idx[5] : idx[9], "A"] = 80.0
    px.loc[idx[12] :, "A"] = 50.0
    ex = cfg.execution.model_copy(deep=True)
    ex.initial_capital = 10_000.0
    ex.execution_lag_bars = 0
    ex.cash_buffer = 0.0
    ex.risk_limits.max_drawdown_pct = 0.10
    ex.risk_limits.rearm_after_bars = 3
    schedule = [
        _decision(px, 0, {"A": 1.0}),
        _decision(px, 10, {"A": 1.0}),
    ]

    res = simulate(px, schedule, ex)

    assert res.halts == [idx[5], idx[12]]
    assert res.halted_on == idx[5]
    assert res.rearms == [idx[8]]
    assert any(fill["date"] == idx[10] for fill in res.fills)
    assert all(fill["date"] <= idx[5] or fill["date"] >= idx[10] for fill in res.fills)
    assert all(value == 0 for value in res.final_shares.values())


def test_scheduled_rebalance_executes_on_rearm_bar(cfg):
    idx = pd.bdate_range("2023-01-02", periods=12)
    px = pd.DataFrame({"A": 100.0}, index=idx)
    px.loc[idx[3] : idx[5], "A"] = 80.0
    ex = cfg.execution.model_copy(deep=True)
    ex.initial_capital = 10_000.0
    ex.execution_lag_bars = 0
    ex.cash_buffer = 0.0
    ex.risk_limits.max_drawdown_pct = 0.10
    ex.risk_limits.rearm_after_bars = 3
    schedule = [
        _decision(px, 0, {"A": 1.0}),
        _decision(px, 5, {"A": 1.0}),
        _decision(px, 6, {"A": 1.0}),
    ]

    res = simulate(px, schedule, ex)

    assert res.halts == [idx[3]]
    assert res.rearms == [idx[6]]
    assert any(fill["date"] == idx[6] and fill["side"] == "BUY" for fill in res.fills)
    assert all(fill["date"] not in (idx[4], idx[5]) for fill in res.fills)


def test_kill_switch_without_rearm_remains_latched(cfg):
    idx = pd.bdate_range("2023-01-02", periods=20)
    px = pd.DataFrame({"A": 100.0}, index=idx)
    px.loc[idx[5] :, "A"] = 80.0
    ex = cfg.execution.model_copy(deep=True)
    ex.initial_capital = 10_000.0
    ex.execution_lag_bars = 0
    ex.cash_buffer = 0.0
    ex.risk_limits.max_drawdown_pct = 0.10
    ex.risk_limits.rearm_after_bars = None
    schedule = [
        _decision(px, 0, {"A": 1.0}),
        _decision(px, 10, {"A": 1.0}),
    ]

    res = simulate(px, schedule, ex)

    assert res.halts == [idx[5]]
    assert res.halted_on == idx[5]
    assert res.rearms == []
    assert all(fill["date"] <= idx[5] for fill in res.fills)
    assert all(value == 0 for value in res.final_shares.values())


def test_benchmark_schedule_is_equal_weight(cfg, prices):
    sched = build_schedule(prices, cfg)
    bench = benchmark_schedule(sched)
    assert all(np.allclose(b.weights, 1 / len(b.weights)) for b in bench)
    assert [b.date for b in bench] == [d.date for d in sched]


def test_metrics_known_values():
    eq = pd.Series([100.0, 110.0, 99.0, 120.0], index=pd.bdate_range("2023-01-02", periods=4))
    m = compute_metrics(eq)
    assert np.isclose(m["total_return"], 0.2)
    assert np.isclose(m["max_drawdown"], 99 / 110 - 1)
