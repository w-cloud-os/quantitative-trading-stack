from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from quant_stack.config import StackConfig, load_config

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_preset_is_valid(path):
    assert load_config(path).environment in ("paper", "backtest")


def _raw():
    return yaml.safe_load(CONFIGS[0].read_text())


def test_unknown_key_is_rejected():
    raw = _raw()
    raw["optimization"]["turnover_max"] = 0.5  # the old, wrong key name
    with pytest.raises(ValidationError):
        StackConfig.model_validate(raw)


def test_live_environment_does_not_exist():
    raw = _raw()
    raw["environment"] = "live"
    with pytest.raises(ValidationError):
        StackConfig.model_validate(raw)


def test_impossible_weight_bounds_rejected():
    raw = _raw()
    raw["optimization"]["max_asset_weight"] = 0.10  # 6 x 10% < 100%
    with pytest.raises(ValidationError):
        StackConfig.model_validate(raw)


def test_sector_constraint_needs_matching_sector():
    raw = _raw()
    raw["optimization"]["sector_max"] = {"Energy": 0.2}
    with pytest.raises(ValidationError):
        StackConfig.model_validate(raw)


def test_cash_buffer_must_cover_fees():
    raw = _raw()
    raw["execution"]["fee_rate"] = 0.01
    raw["execution"]["cash_buffer"] = 0.01
    with pytest.raises(ValidationError):
        StackConfig.model_validate(raw)
