"""Additional coverage tests for config and enum helpers."""

import pytest

from OpenPinch.lib.config import Configuration
from OpenPinch.lib.config_metadata import configuration_option_status
from OpenPinch.lib.enums import TT, ZT


def test_configuration_parses_refrigerant_list_option():
    cfg = Configuration(options={"REFRIGERANTS": "water;ammonia,co2"})
    assert cfg.REFRIGERANTS == ["water", "ammonia", "co2"]


def test_configuration_rejects_unknown_option_keys():
    with pytest.raises(ValueError, match="Unknown configuration option"):
        Configuration(options={"NOT_A_REAL_OPTION": 1})


def test_configuration_rejects_removed_legacy_turbine_gateway():
    with pytest.raises(ValueError, match="Unknown configuration option"):
        Configuration(options={"turbine": [{"key": "PROP_TOP_0", "value": 450.0}]})


def test_configuration_rejects_removed_condensate_alias():
    with pytest.raises(ValueError, match="Unknown configuration option"):
        Configuration(options={"HP_CONDESATE": True})


def test_configuration_option_status_classifies_runtime_roles():
    assert configuration_option_status("DT_CONT").runtime_status == "supported"
    assert (
        configuration_option_status("DO_EXERGY_TARGETING").runtime_status
        == "experimental"
    )
    assert configuration_option_status("HP_CONDESATE").runtime_status == "dead"
    assert configuration_option_status("PROP_TOP_0").runtime_status == "dead"
    assert configuration_option_status("NOT_A_REAL_OPTION").runtime_status == "dead"


def test_zone_and_target_enum_str_methods():
    assert str(ZT.S) == ZT.S.value
    assert str(TT.DI) == TT.DI.value
