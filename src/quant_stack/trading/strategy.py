"""Execution + oversight desks as a NautilusTrader strategy.

* Rebalances to target weights on the execution bars listed in ``schedule``.
* Enforces ``max_order_value`` by slicing orders, and never exceeds ``max_position_leverage``.
* Drawdown kill-switch: cancels orders, flattens every position and stops trading (latched).
"""

from __future__ import annotations

import math

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.trading.strategy import Strategy

from quant_stack.risk import DrawdownMonitor, slice_quantity


class RebalanceConfig(StrategyConfig, frozen=True):
    instrument_ids: list[str]
    # execution bar ts_event (ns since epoch) -> {ticker: target weight}
    schedule: dict[int, dict[str, float]]
    max_drawdown_pct: float
    max_order_value: float
    max_position_leverage: float = 1.0
    cash_buffer: float = 0.01


class PortfolioRebalanceStrategy(Strategy):
    def __init__(self, config: RebalanceConfig) -> None:
        super().__init__(config=config)
        self.instruments: dict[InstrumentId, object] = {}
        self.monitor = DrawdownMonitor(config.max_drawdown_pct)
        self.equity_log: list[tuple[int, float]] = []
        self.halted_ts: int | None = None
        self.final_shares: dict[str, int] = {}
        self._last_close: dict[InstrumentId, float] = {}
        self._bar_ts: dict[InstrumentId, int] = {}
        self._venue: Venue | None = None

    def on_start(self) -> None:
        for raw in self.config.instrument_ids:
            iid = InstrumentId.from_str(raw)
            instrument = self.cache.instrument(iid)
            if instrument is None:
                self.log.error(f"Instrument {iid} not found in cache; stopping.")
                self.stop()
                return
            self.instruments[iid] = instrument
            self._venue = iid.venue
            self.subscribe_bars(BarType.from_str(f"{iid}-1-DAY-LAST-EXTERNAL"))
        self.log.info(f"Initialized with {len(self.instruments)} instruments.")

    def on_bar(self, bar: Bar) -> None:
        iid = bar.bar_type.instrument_id
        self._last_close[iid] = float(bar.close)
        self._bar_ts[iid] = bar.ts_event
        # Act once per day, after the LAST instrument's bar for this timestamp has arrived.
        if any(self._bar_ts.get(i) != bar.ts_event for i in self.instruments):
            return
        equity = self._equity()
        self.equity_log.append((bar.ts_event, equity))
        if self.monitor.tripped:
            return
        if self.monitor.update(equity):
            self.log.error(
                f"KILL-SWITCH: drawdown {self.monitor.drawdown:.2%} >= "
                f"{self.config.max_drawdown_pct:.2%}. Flattening and halting."
            )
            self._liquidate()
            self.halted_ts = bar.ts_event
            return
        weights = self.config.schedule.get(bar.ts_event)
        if weights:
            self._rebalance(weights, equity)

    def _equity(self) -> float:
        equity = self.portfolio.equity(self._venue)  # {Currency: Money}, net of commissions
        money = equity.get(USD)
        return float(money.as_double()) if money is not None else 0.0

    def _submit(self, iid: InstrumentId, side: OrderSide, quantity: int) -> None:
        instrument = self.instruments[iid]
        for child in slice_quantity(quantity, self._last_close[iid], self.config.max_order_value):
            order = self.order_factory.market(
                instrument_id=iid, order_side=side, quantity=instrument.make_qty(child)
            )
            self.submit_order(order)

    def _rebalance(self, weights: dict[str, float], equity: float) -> None:
        gross = min(1.0, self.config.max_position_leverage) * (1.0 - self.config.cash_buffer)
        targets: dict[InstrumentId, int] = {}
        for iid in self.instruments:
            w = weights.get(iid.symbol.value, 0.0)
            targets[iid] = max(0, math.floor(w * gross * equity / self._last_close[iid]))
        current = {iid: int(self.portfolio.net_position(iid)) for iid in self.instruments}
        for iid, target in targets.items():  # sells first so cash is freed before buys
            if target < current[iid]:
                self._submit(iid, OrderSide.SELL, current[iid] - target)
        for iid, target in targets.items():
            if target > current[iid]:
                self._submit(iid, OrderSide.BUY, target - current[iid])
        self.log.info(f"Rebalanced at equity {equity:,.2f}")

    def _liquidate(self) -> None:
        for iid in self.instruments:
            self.cancel_all_orders(iid)
            self.close_all_positions(iid)

    def on_stop(self) -> None:
        for iid in self.instruments:
            self.cancel_all_orders(iid)
        self.final_shares = {
            iid.symbol.value: int(self.portfolio.net_position(iid)) for iid in self.instruments
        }
