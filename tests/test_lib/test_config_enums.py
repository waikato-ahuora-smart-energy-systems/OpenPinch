"""Additional coverage tests for config and enum helpers."""

from OpenPinch.lib.config import Configuration
from OpenPinch.lib.enums import TT, ZT


def test_configuration_parses_refrigerant_list_option():
    cfg = Configuration(options={"REFRIGERANTS": "water;ammonia,co2"})
    assert cfg.REFRIGERANTS == ["water", "ammonia", "co2"]


def test_zone_and_target_enum_str_methods():
    assert str(ZT.S) == ZT.S.value
    assert str(TT.DI) == TT.DI.value
