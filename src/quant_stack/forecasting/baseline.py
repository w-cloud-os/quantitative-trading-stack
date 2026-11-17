"""Forecast desk: expected returns *with uncertainty attached*.

The baseline here is intentionally simple (trailing mean + standard error) so the interface is
what matters. Any model (Chronos, TimesFM, Kronos, your own) can replace it by implementing
``Forecaster.forecast`` and returning a ``Forecast``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

# E|Z| for Z ~ N(0,1): the average |t-stat| you would see from pure noise.
NOISE_FLOOR = math.sqrt(2.0 / math.pi)


@dataclass(frozen=True)
class Forecast:
    mu: pd.Series  # expected daily return per asset
    stderr: pd.Series  # standard error of that estimate
    confidence: float  # 0 = pure noise (hold the benchmark), 1 = strong signal (trust optimizer)


class Forecaster(Protocol):
    def forecast(self, returns: pd.DataFrame) -> Forecast: ...


class BaselineForecaster:
    def __init__(self, window: int = 63, full_confidence_tstat: float = 2.0) -> None:
        if full_confidence_tstat <= NOISE_FLOOR:
            raise ValueError("full_confidence_tstat must exceed the noise floor (~0.8)")
        self.window = window
        self.full_confidence_tstat = full_confidence_tstat

    def forecast(self, returns: pd.DataFrame) -> Forecast:
        tail = returns.iloc[-self.window :]
        n = len(tail)
        if n < 20:
            raise ValueError("need at least 20 observations to forecast")
        mu = tail.mean()
        stderr = tail.std(ddof=1) / math.sqrt(n)
        tstat = (mu / stderr.replace(0.0, np.nan)).abs().fillna(0.0)
        strength = (float(tstat.mean()) - NOISE_FLOOR) / (self.full_confidence_tstat - NOISE_FLOOR)
        return Forecast(mu=mu, stderr=stderr, confidence=float(np.clip(strength, 0.0, 1.0)))
