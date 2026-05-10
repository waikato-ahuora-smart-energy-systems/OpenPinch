"""Additional coverage tests for config and enum helpers."""

import pytest

from OpenPinch.lib.config import Configuration
from OpenPinch.lib.enums import TT, ZT


def test_configuration_parses_refrigerant_list_option():
    cfg = Configuration(options={"REFRIGERANTS": "water;ammonia,co2"})
    assert cfg.REFRIGERANTS == ["water", "ammonia", "co2"]


def test_configuration_rejects_unknown_option_keys():
    with pytest.raises(ValueError, match="Unknown configuration option"):
        Configuration(options={"NOT_A_REAL_OPTION": 1})


def test_configuration_rejects_legacy_turbine_gateway():
    with pytest.raises(
        ValueError,
        match="Legacy workbook option gateways are no longer supported",
    ):
        Configuration(options={"turbine": [{"key": "PROP_TOP_0", "value": 450.0}]})


def test_configuration_rejects_renamed_condensate_flag():
    with pytest.raises(
        ValueError,
        match="HP_CONDESATE -> IS_HIGH_P_COND_FLASH",
    ):
        Configuration(options={"HP_CONDESATE": True})


def test_zone_and_target_enum_str_methods():
    assert str(ZT.S) == ZT.S.value
    assert str(TT.DI) == TT.DI.value
