"""Cross-check: the NautilusTrader event-driven engine must reproduce the reference simulator."""

import pandas as pd
import pytest

from quant_stack.backtest.schedule import build_schedule
from quant_stack.backtest.simulate import simulate
from quant_stack.trading.engine import run_nautilus_backtest

pytestmark = pytest.mark.nautilus


@pytest.fixture(scope="module")
def schedule(cfg, prices):
    return build_schedule(prices, cfg)


def _compare(prices, schedule, ex):
    sim = simulate(prices, schedule, ex)
    nt = run_nautilus_backtest(prices, schedule, ex)
    assert len(nt.equity) == len(sim.equity)
    assert (nt.equity - sim.equity).abs().max() < 1.0  # dollars, on ~$1,000,000
    assert len(nt.fills) == len(sim.fills)
    assert nt.final_shares == sim.final_shares
    assert nt.halted_on == sim.halted_on
    return sim, nt


def test_engines_agree_without_kill_switch(cfg, prices, schedule):
    ex = cfg.execution.model_copy(deep=True)
    ex.risk_limits.max_drawdown_pct = 0.60
    sim, nt = _compare(prices, schedule, ex)
    assert sim.halted_on is None and len(sim.fills) > 50


def test_engines_agree_when_kill_switch_trips(cfg, prices, schedule):
    ex = cfg.execution.model_copy(deep=True)
    ex.risk_limits.max_drawdown_pct = 0.05
    sim, nt = _compare(prices, schedule, ex)
    assert isinstance(sim.halted_on, pd.Timestamp)
    assert all(v == 0 for v in nt.final_shares.values())


def test_engine_can_run_twice_in_one_process(cfg, prices, schedule):
    ex = cfg.execution
    a = run_nautilus_backtest(prices, schedule, ex)
    b = run_nautilus_backtest(prices, schedule, ex)
    assert a.equity.equals(b.equity)  # deterministic
