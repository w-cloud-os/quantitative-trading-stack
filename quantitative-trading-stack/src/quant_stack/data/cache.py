"""Parquet cache keyed by a hash of *all* fetch parameters (provider, tickers, dates, ...)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


class LocalDataCache:
    def __init__(self, cache_dir: str | Path = ".data_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(**params: object) -> str:
        blob = json.dumps(params, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:24]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def get(self, key: str) -> pd.DataFrame | None:
        path = self._path(key)
        return pd.read_parquet(path) if path.exists() else None

    def set(self, key: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._path(key))
