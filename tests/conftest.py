from pathlib import Path

import pytest

from quant_stack.config import load_config
from quant_stack.data.pipeline import clean_prices
from quant_stack.data.providers import SyntheticProvider

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "paper_trading.yaml")


@pytest.fixture(scope="session")
def prices(cfg):
    return clean_prices(SyntheticProvider(7).fetch(cfg.data.tickers, "2021-01-01", "2024-06-28"))
