"""Confidence-weighted blend: confident forecast -> tilt to the optimizer, guessing -> benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd


def benchmark_weights(assets: list[str] | pd.Index) -> pd.Series:
    n = len(assets)
    return pd.Series(np.full(n, 1.0 / n), index=list(assets), name="benchmark")


def blend_weights(optimized: pd.Series, benchmark: pd.Series, confidence: float) -> pd.Series:
    """w = benchmark + confidence * (optimized - benchmark). Sums to 1 if both inputs do."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    bench = benchmark.reindex(optimized.index)
    if bench.isna().any():
        raise ValueError("benchmark and optimized weights must cover the same assets")
    return bench + confidence * (optimized - bench)
