"""Fast reference simulator. It follows exactly the same rules as the Nautilus strategy
(whole shares, same sizing, fees on notional, order slicing, drawdown kill-switch), so the two
engines can be cross-checked against each other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from quant_stack.backtest.schedule import Decision
from quant_stack.config import ExecutionConfig
from quant_stack.risk import DrawdownMonitor, slice_quantity


@dataclass
class SimResult:
    equity: pd.Series  # equity at each bar's close, BEFORE that bar's own trades
    fills: list[dict] = field(default_factory=list)
    fees: float = 0.0
    traded_notional: float = 0.0
    halted_on: pd.Timestamp | None = None
    final_shares: dict[str, int] = field(default_factory=dict)


def prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Round to cents (the instrument's price precision) so every engine sees identical prices."""
    return prices.round(2)


def target_shares(
    weights: pd.Series, equity: float, prices_row: pd.Series, cfg: ExecutionConfig
) -> dict[str, int]:
    gross = min(1.0, cfg.risk_limits.max_position_leverage) * (1.0 - cfg.cash_buffer)
    return {
        t: max(0, math.floor(float(weights[t]) * gross * equity / float(prices_row[t])))
        for t in prices_row.index
    }


def simulate(prices: pd.DataFrame, schedule: list[Decision], cfg: ExecutionConfig) -> SimResult:
    prices = prepare_prices(prices)
    lag = cfg.execution_lag_bars
    exec_at: dict[int, pd.Series] = {}
    for d in schedule:
        k = prices.index.get_loc(d.date) + lag
        if k < len(prices):
            exec_at[k] = d.weights
    tickers = list(prices.columns)
    shares = {t: 0 for t in tickers}
    cash = cfg.initial_capital
    monitor = DrawdownMonitor(cfg.risk_limits.max_drawdown_pct)
    res = SimResult(equity=pd.Series(dtype=float))
    equity_log: list[float] = []

    def trade(date: pd.Timestamp, ticker: str, delta: int, px: float) -> None:
        nonlocal cash
        if delta == 0:
            return
        side = "BUY" if delta > 0 else "SELL"
        for child in slice_quantity(abs(delta), px, cfg.risk_limits.max_order_value):
            notional = child * px
            fee = round(notional * cfg.fee_rate, 2)
            cash += (-notional if side == "BUY" else notional) - fee
            res.fees += fee
            res.traded_notional += notional
            res.fills.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "side": side,
                    "qty": child,
                    "price": px,
                    "fee": fee,
                }
            )
        shares[ticker] += delta

    for k, date in enumerate(prices.index):
        row = prices.iloc[k]
        equity = cash + sum(shares[t] * float(row[t]) for t in tickers)
        equity_log.append(equity)
        if monitor.tripped:
            continue
        if monitor.update(equity):
            res.halted_on = date
            for t in tickers:
                trade(date, t, -shares[t], float(row[t]))
            continue
        if k in exec_at:
            targets = target_shares(exec_at[k], equity, row, cfg)
            for t in tickers:  # sells first, then buys
                if targets[t] < shares[t]:
                    trade(date, t, targets[t] - shares[t], float(row[t]))
            for t in tickers:
                if targets[t] > shares[t]:
                    trade(date, t, targets[t] - shares[t], float(row[t]))
    res.equity = pd.Series(equity_log, index=prices.index, name="equity")
    res.final_shares = dict(shares)
    return res
