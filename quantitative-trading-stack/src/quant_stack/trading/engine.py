"""Runs the schedule through NautilusTrader's event-driven BacktestEngine (simulated venue)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Money, Price, Quantity

from quant_stack.backtest.schedule import Decision
from quant_stack.backtest.simulate import prepare_prices
from quant_stack.config import ExecutionConfig
from quant_stack.trading.strategy import PortfolioRebalanceStrategy, RebalanceConfig

_BAR_VOLUME = 10_000_000  # synthetic liquidity; the sim venue does not model volume limits here


@dataclass
class NautilusResult:
    equity: pd.Series
    fills: pd.DataFrame
    halted_on: pd.Timestamp | None
    final_shares: dict[str, int]


def _make_instrument(ticker: str, venue: Venue, fee_rate: float) -> Equity:
    return Equity(
        instrument_id=InstrumentId(Symbol(ticker), venue),
        raw_symbol=Symbol(ticker),
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        maker_fee=Decimal(str(fee_rate)),
        taker_fee=Decimal(str(fee_rate)),
        ts_event=0,
        ts_init=0,
    )


def run_nautilus_backtest(
    prices: pd.DataFrame, schedule: list[Decision], cfg: ExecutionConfig, log_level: str = "ERROR"
) -> NautilusResult:
    prices = prepare_prices(prices)
    venue = Venue(cfg.venue)
    # Daily bars stamped 21:00 UTC (after the US close). O=H=L=C=adjusted close: this is a
    # close-to-close rebalancing study, not an intraday simulation.
    ts_ns = [(pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=21)).value for d in prices.index]

    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level=log_level)))
    try:
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            starting_balances=[Money(cfg.initial_capital, USD)],
            base_currency=USD,
            default_leverage=Decimal(1),
        )
        instrument_ids: list[str] = []
        for ticker in prices.columns:
            instrument = _make_instrument(ticker, venue, cfg.fee_rate)
            engine.add_instrument(instrument)
            bar_type = BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")
            bars = []
            for ts, px in zip(ts_ns, prices[ticker].to_numpy(dtype=float), strict=True):
                p = Price(float(px), 2)
                bars.append(Bar(bar_type, p, p, p, p, Quantity.from_int(_BAR_VOLUME), ts, ts))
            engine.add_data(bars)
            instrument_ids.append(str(instrument.id))

        exec_schedule: dict[int, dict[str, float]] = {}
        for d in schedule:
            k = prices.index.get_loc(d.date) + cfg.execution_lag_bars
            if k < len(prices):
                exec_schedule[ts_ns[k]] = {t: float(w) for t, w in d.weights.items()}

        strategy = PortfolioRebalanceStrategy(
            RebalanceConfig(
                instrument_ids=instrument_ids,
                schedule=exec_schedule,
                max_drawdown_pct=cfg.risk_limits.max_drawdown_pct,
                max_order_value=cfg.risk_limits.max_order_value,
                max_position_leverage=cfg.risk_limits.max_position_leverage,
                cash_buffer=cfg.cash_buffer,
            )
        )
        engine.add_strategy(strategy)
        engine.run()

        date_by_ts = dict(zip(ts_ns, prices.index, strict=True))
        equity = pd.Series(
            {date_by_ts[ts]: eq for ts, eq in strategy.equity_log}, name="equity", dtype=float
        )
        fills = engine.trader.generate_fills_report()
        halted = date_by_ts.get(strategy.halted_ts) if strategy.halted_ts is not None else None
        return NautilusResult(equity, fills, halted, dict(strategy.final_shares))
    finally:
        engine.dispose()
