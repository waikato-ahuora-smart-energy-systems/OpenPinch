"""Additional coverage tests for config and enum helpers."""

import json

import pytest

import OpenPinch.domain.enums as enums
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.configuration_fields import configuration_option_status
from OpenPinch.domain.enums import HeatPumpAndRefrigerationCycle, TargetType, ZoneType
from OpenPinch.presentation.configuration import configuration_options
from tests.support.paths import FIXTURES_ROOT

FIXTURE_PATH = FIXTURES_ROOT / "config_cases.json"


def test_retired_enum_identity_aliases_are_absent():
    for name in (
        "ZT",
        "TT",
        "ST",
        "SID",
        "PT",
        "GT",
        "ResultsType",
        "HPRcycle",
        "HENDesignMethod",
        "SynthesisMethod",
        "SynthesisDesignMethod",
    ):
        assert not hasattr(enums, name)


def _config_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_configuration_parses_refrigerant_list_option():
    cfg = Configuration(options={"HPR_REFRIGERANTS": ["water", "ammonia", "co2"]})
    assert cfg.hpr.refrigerants == ["water", "ammonia", "co2"]
    assert not hasattr(cfg, "HPR_REFRIGERANTS")


def test_configuration_exposes_normalised_hpr_backend_options():
    cfg = Configuration(
        options={
            "HPR_REFRIGERANTS": ["water", "ammonia", "co2"],
            "HPR_MVR_FLUIDS": ["Water", "R134A"],
            "HPR_HE_ETA_II_CARNOT": 0.4,
            "HPR_INTEGRATED_EXPANDER_ENABLED": True,
        }
    )

    assert cfg.hpr.normalised_refrigerants == ["WATER", "AMMONIA", "CO2"]
    assert cfg.hpr.normalised_mvr_fluids == ["Water", "R134A"]
    assert cfg.hpr.effective_eta_ii_he_carnot == pytest.approx(0.4)
    assert not hasattr(cfg, "hpr_refrigerants")
    assert not hasattr(cfg, "hpr_mvr_fluids")
    assert not hasattr(cfg, "effective_eta_ii_he_carnot")


def test_configuration_disables_effective_hpr_heat_engine_efficiency_by_default():
    cfg = Configuration(
        options={
            "HPR_HE_ETA_II_CARNOT": 0.4,
            "HPR_INTEGRATED_EXPANDER_ENABLED": False,
        }
    )

    assert cfg.hpr.effective_eta_ii_he_carnot == pytest.approx(0.0)


def test_configuration_accepts_input_and_output_unit_maps():
    cfg = Configuration(options=_config_fixture()["period_config"])

    assert cfg.input_unit_overrides["temperature"] == "K"
    assert cfg.output_unit_overrides["heat_flow"] == "MW"


def test_configuration_constructor_helpers_and_catalog_use_static_fixture():
    fixture = _config_fixture()["period_config"]

    with pytest.raises(TypeError, match="provided as a dict"):
        Configuration(options=[("THERMAL_DT_CONT", 5.0)])

    cfg = Configuration.from_options(
        fixture,
        top_zone_name="Custom Site",
        top_zone_identifier=ZoneType.S.value,
    )

    assert cfg.problem.top_zone_name == "Custom Site"
    assert cfg.problem.top_zone_identifier == ZoneType.S.value
    assert "THERMAL_DT_CONT" in Configuration._known_option_keys()
    assert any(field.name == "THERMAL_DT_CONT" for field in configuration_options())


def test_configuration_for_period_resolves_ids_indexes_and_default_weights():
    cfg = Configuration(options=_config_fixture()["period_config"])

    default_context = cfg.for_period()
    peak_context = cfg.for_period(period_id="peak")
    index_context = cfg.for_period(period_idx=1)

    assert default_context.period_id == "base"
    assert default_context.period_idx == 0
    assert default_context.weight == pytest.approx(0.25)
    assert peak_context.period_id == "peak"
    assert peak_context.period_idx == 1
    assert peak_context.weight == pytest.approx(0.75)
    assert index_context.period_id == "peak"
    assert index_context.period_idx == 1
    assert index_context.weight == pytest.approx(0.75)

    missing_weight_cfg = Configuration(
        options=_config_fixture()["missing_weight_config"]
    )
    assert missing_weight_cfg.for_period(period_idx=1).weight == pytest.approx(1.0)

    with pytest.raises(ValueError, match="Unknown period_id"):
        cfg.for_period(period_id="missing")
    with pytest.raises(ValueError, match="Unknown period index"):
        cfg.for_period(period_idx=10)


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
    assert configuration_option_status("THERMAL_DT_CONT").runtime_status == "supported"
    assert (
        configuration_option_status("TARGETING_EXERGY_ENABLED").runtime_status == "dead"
    )
    assert configuration_option_status("HP_CONDESATE").runtime_status == "dead"
    assert configuration_option_status("PROP_TOP_0").runtime_status == "dead"
    assert configuration_option_status("NOT_A_REAL_OPTION").runtime_status == "dead"


def test_zone_and_target_enum_str_methods():
    assert str(ZoneType.S) == ZoneType.S.value
    assert str(TargetType.DI) == TargetType.DI.value


def test_hpr_cycle_enum_remains_internal_and_configuration_rejects_selectors():
    assert [cycle.name for cycle in HeatPumpAndRefrigerationCycle] == [
        "CascadeCarnot",
        "ParallelCarnot",
        "Brayton",
        "CascadeVapourComp",
        "ParallelVapourComp",
        "VapourCompMVR",
    ]
    assert [cycle.value for cycle in HeatPumpAndRefrigerationCycle] == [
        "Cascade Carnot cycles",
        "Parallel Carnot cycles",
        "Brayton cycle",
        "Cascade vapour compression cycles",
        "Parallel vapour compression cycles",
        "Vapour compression with MVR cascade",
    ]
    assert Configuration().hpr.type == HeatPumpAndRefrigerationCycle.CascadeCarnot.value
    with pytest.raises(ValueError, match="Unknown configuration option"):
        Configuration(
            options={"HPR_TYPE": HeatPumpAndRefrigerationCycle.ParallelCarnot.value}
        )
    with pytest.raises(ValueError, match="Unknown configuration option"):
        Configuration(options={"POWER_TURB_MODEL": "Sun-Smith"})


@pytest.mark.parametrize(
    "legacy_key",
    [
        "REFRIGERANTS",
        "MVR_FLUIDS",
        "ETA_II_HE_CARNOT",
        "ALLOW_INTEGRATED_EXPANDER",
        "INPUT_UNITS",
        "OUTPUT_UNITS",
        "DT_CONT",
        "DO_EXERGY_TARGETING",
    ],
)
def test_configuration_rejects_retired_flat_option_names(legacy_key):
    with pytest.raises(ValueError, match="Unknown configuration option"):
        Configuration(options={legacy_key: None})


def test_configuration_builds_two_layer_runtime_config_from_flat_options():
    cfg = Configuration(
        options={
            "PROBLEM_TOP_ZONE_NAME": "Plant",
            "PROBLEM_TOP_ZONE_IDENTIFIER": ZoneType.S.value,
            "COSTING_ANNUAL_OP_TIME": 7000.0,
            "ENV_TEMPERATURE": 20.0,
            "POWER_TURB_P_IN": 110.0,
            "COSTING_HX_UNIT_COST": 5000.0,
            "HENS_SOLVER_EVM": "ipopt-pyomo",
            "THERMAL_DT_CONT": 8.0,
            "THERMAL_DT_PHASE_CHANGE": 0.02,
        }
    )

    assert cfg.problem.top_zone_name == "Plant"
    assert cfg.problem.top_zone_identifier == ZoneType.S.value
    assert cfg.costing.annual_op_time == pytest.approx(7000.0)
    assert cfg.environment.temperature == pytest.approx(20.0)
    assert cfg.power.turb_p_in == pytest.approx(110.0)
    assert cfg.costing.hx_unit_cost == pytest.approx(5000.0)
    assert cfg.hens.solver_evm == "ipopt-pyomo"
    assert not hasattr(cfg, "targeting")
    assert cfg.thermal.dt_cont == pytest.approx(8.0)
    assert cfg.thermal.dt_phase_change == pytest.approx(0.02)
    assert not hasattr(cfg, "COSTING_ANNUAL_OP_TIME")
    assert not hasattr(cfg, "POWER_TURB_P_IN")
    assert not hasattr(cfg, "PROBLEM_TOP_ZONE_NAME")
    assert not hasattr(cfg, "PROBLEM_TOP_ZONE_IDENTIFIER")
    assert not hasattr(cfg, "THERMAL_DT_CONT")
    assert not hasattr(cfg, "THERMAL_DT_PHASE_CHANGE")
