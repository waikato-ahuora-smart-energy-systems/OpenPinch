"""Additional coverage tests for config and enum helpers."""

import pytest

from OpenPinch.lib.config import Configuration
from OpenPinch.lib.config_metadata import configuration_option_status
from OpenPinch.lib.enums import TT, ZT, HPRcycle


def test_configuration_parses_refrigerant_list_option():
    cfg = Configuration(options={"REFRIGERANTS": "water;ammonia,co2"})
    assert cfg.REFRIGERANTS == ["water", "ammonia", "co2"]


def test_configuration_accepts_input_and_output_unit_maps():
    cfg = Configuration(
        options={
            "INPUT_UNITS": {"temperature": "K"},
            "OUTPUT_UNITS": {"heat_flow": "MW"},
        }
    )

    assert cfg.INPUT_UNITS == {"temperature": "K"}
    assert cfg.OUTPUT_UNITS == {"heat_flow": "MW"}


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


def test_configuration_accepts_current_hpr_names_and_rejects_invalid_names():
    assert [cycle.name for cycle in HPRcycle] == [
        "CascadeCarnot",
        "ParallelCarnot",
        "Brayton",
        "CascadeVapourComp",
        "ParallelVapourComp",
        "VapourCompMVR",
    ]
    assert [cycle.value for cycle in HPRcycle] == [
        "Cascade Carnot cycles",
        "Parallel Carnot cycles",
        "Brayton cycle",
        "Cascade vapour compression cycles",
        "Parallel vapour compression cycles",
        "Vapour compression with MVR cascade",
    ]
    assert (
        Configuration(options={"HPR_TYPE": HPRcycle.CascadeCarnot.value}).HPR_TYPE
        == HPRcycle.CascadeCarnot.value
    )
    assert (
        Configuration(options={"HPR_TYPE": HPRcycle.ParallelCarnot.value}).HPR_TYPE
        == HPRcycle.ParallelCarnot.value
    )
    assert (
        Configuration(options={"HPR_TYPE": HPRcycle.ParallelVapourComp.value}).HPR_TYPE
        == HPRcycle.ParallelVapourComp.value
    )

    with pytest.raises(ValueError, match="Invalid value"):
        Configuration(options={"HPR_TYPE": "Not a real HPR cycle"})
