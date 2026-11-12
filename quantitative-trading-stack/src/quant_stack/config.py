"""Typed, validated configuration. One schema is shared by every module.

``extra="forbid"`` means a typo in a YAML file fails loudly instead of being silently ignored.
``environment`` only accepts "paper" or "backtest": there is deliberately no "live" option.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    tickers: list[str] = Field(min_length=2)
    lookback_days: int = Field(default=756, ge=60, description="Trading days of history.")
    frequency: Literal["1d"] = "1d"
    sectors: dict[str, str] = Field(default_factory=dict, description="ticker -> sector name")

    @model_validator(mode="after")
    def _check(self) -> DataConfig:
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("tickers must be unique")
        unknown = set(self.sectors) - set(self.tickers)
        if unknown:
            raise ValueError(f"sectors reference tickers not in the universe: {sorted(unknown)}")
        return self


class OptimizationConfig(_Strict):
    risk_measure: Literal["VARIANCE", "CVAR", "CDAR"] = "CVAR"
    max_asset_weight: float = Field(default=0.35, gt=0, le=1)
    min_asset_weight: float = Field(default=0.0, ge=0, lt=1)
    l2_coef: float = Field(default=0.01, ge=0)
    # skfolio's max_turnover is a *per-asset* cap on |w_new - w_previous|, not total turnover.
    max_weight_change: float | None = Field(default=0.25, gt=0, le=1)
    denoise_covariance: bool = True
    shrink_mu: bool = True
    sector_max: dict[str, float] = Field(default_factory=dict)
    sector_min: dict[str, float] = Field(default_factory=dict)
    train_window: int = Field(default=252, ge=60, description="Trading days used per optimization.")


class ForecastConfig(_Strict):
    enabled: bool = True
    window: int = Field(default=63, ge=20)
    full_confidence_tstat: float = Field(default=2.0, gt=0.8)


class RiskLimits(_Strict):
    max_drawdown_pct: float = Field(default=0.10, gt=0, lt=1)
    max_order_value: float = Field(default=250_000.0, gt=0)
    max_position_leverage: float = Field(default=1.0, gt=0, le=1.0)  # long-only, no leverage


class ExecutionConfig(_Strict):
    venue: str = "SIM"
    base_currency: Literal["USD"] = "USD"
    initial_capital: float = Field(default=1_000_000.0, gt=0)
    rebalance_interval_bars: int = Field(default=20, ge=1)
    execution_lag_bars: int = Field(default=1, ge=0, description="Bars between decision and trade.")
    fee_rate: float = Field(default=0.0005, ge=0, lt=0.05, description="Fraction of notional.")
    cash_buffer: float = Field(default=0.01, ge=0, lt=0.5)
    risk_limits: RiskLimits = Field(default_factory=RiskLimits)


class StackConfig(_Strict):
    version: str = "1.0"
    preset_name: str = "unnamed"
    environment: Literal["paper", "backtest"] = "paper"
    data: DataConfig
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @model_validator(mode="after")
    def _check(self) -> StackConfig:
        n = len(self.data.tickers)
        o = self.optimization
        if o.min_asset_weight > o.max_asset_weight:
            raise ValueError("min_asset_weight cannot exceed max_asset_weight")
        if o.max_asset_weight * n < 1.0 - 1e-9:
            raise ValueError(
                f"max_asset_weight {o.max_asset_weight} x {n} assets cannot sum to 100%"
            )
        if o.min_asset_weight * n > 1.0 + 1e-9:
            raise ValueError("min_asset_weight x number of assets exceeds 100%")
        if not (o.min_asset_weight - 1e-9 <= 1.0 / n <= o.max_asset_weight + 1e-9):
            raise ValueError("equal-weight benchmark must satisfy min/max asset weight bounds")
        sectors = set(self.data.sectors.values())
        for name in (*o.sector_max, *o.sector_min):
            if name not in sectors:
                raise ValueError(f"sector constraint '{name}' has no assets in data.sectors")
        if self.execution.fee_rate * 2 >= self.execution.cash_buffer:
            raise ValueError("cash_buffer must exceed 2x fee_rate so rebalances never overspend")
        return self


def load_config(path: str | Path) -> StackConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return StackConfig.model_validate(raw)
