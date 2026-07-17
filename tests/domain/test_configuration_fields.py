"""Focused tests for declarative configuration metadata validators."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

import OpenPinch.domain.configuration_fields as meta
from OpenPinch.domain.configuration_fields import ConfigurationFieldSpec
from OpenPinch.domain.enums import HeatPumpAndRefrigerationCycle
from tests.support.paths import FIXTURES_ROOT

FIXTURE_PATH = FIXTURES_ROOT / "config_metadata_cases.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_configuration_metadata_group_status_and_support_level_edges():
    assert meta.configuration_group("NOT_A_REAL_OPTION") == "problem"
    assert meta.configuration_field_support_level("THERMAL_DT_CONT") == "stable"
    assert meta.configuration_option_status("HPR_TYPE").runtime_status == "dead"


def test_validate_configuration_options_rejects_non_dict_and_unknown_keys():
    with pytest.raises(ValueError, match="must be provided as a dict"):
        meta.validate_configuration_options([("THERMAL_DT_CONT", 5.0)])

    with pytest.raises(ValueError, match="Unknown configuration option"):
        meta.validate_configuration_options({"NOT_A_REAL_OPTION": 1})


def test_validate_configuration_option_value_accepts_static_valid_cases():
    valid_values = _fixture()["valid_option_values"]

    assert (
        meta.validate_configuration_option_value(
            "INPUT_UNIT_TEMPERATURE", valid_values["INPUT_UNIT_TEMPERATURE"]
        )
        == "K"
    )
    assert (
        meta.validate_configuration_option_value(
            "HENS_DT_CONT_MULTIPLIERS",
            valid_values["HENS_DT_CONT_MULTIPLIERS"],
        )
        is None
    )
    assert meta.validate_configuration_option_value(
        "HENS_APPROACH_TEMPERATURES",
        valid_values["HENS_APPROACH_TEMPERATURES"],
    ) == [5.0, 10.5]
    assert meta.validate_configuration_option_value(
        "HENS_DERIVATIVE_THRESHOLDS",
        valid_values["HENS_DERIVATIVE_THRESHOLDS"],
    ) == [0.25]
    assert meta.validate_configuration_option_value(
        "HENS_STAGE_SELECTION",
        valid_values["HENS_STAGE_SELECTION"],
    ) == [1, 2, 3]
    assert meta.validate_configuration_option_value(
        "HENS_OUTPUT_FORMATS",
        valid_values["HENS_OUTPUT_FORMATS"],
    ) == ["json", "xlsx"]
    assert (
        meta.validate_configuration_option_value(
            "HENS_STAGE_PACKING",
            valid_values["HENS_STAGE_PACKING"],
        )
        == "all"
    )
    assert meta.validate_configuration_option_value(
        "HENS_SOLVE_TOLERANCE",
        valid_values["HENS_SOLVE_TOLERANCE"],
    ) == pytest.approx(0.001)
    assert (
        meta.validate_configuration_option_value(
            "HENS_MAX_PARALLEL",
            valid_values["HENS_MAX_PARALLEL"],
        )
        == 2
    )
    assert (
        meta.validate_configuration_option_value(
            "HENS_EVM_N_AD_BRANCHES",
            valid_values["HENS_EVM_N_AD_BRANCHES"],
        )
        is None
    )
    assert (
        meta.validate_configuration_option_value(
            "HENS_EVM_N_RM_BRANCHES",
            valid_values["HENS_EVM_N_RM_BRANCHES"],
        )
        == 3
    )
    assert (
        meta.validate_configuration_option_value(
            "HENS_RUN_ID",
            valid_values["HENS_RUN_ID"],
        )
        == "Run_01.alpha"
    )
    assert (
        meta.validate_configuration_option_value(
            "HENS_SOLVER_PDM",
            valid_values["HENS_SOLVER_PDM"],
        )
        == "couenne"
    )
    assert meta.validate_configuration_option_value(
        "HENS_SOLVER_OPTIONS_PDM",
        valid_values["HENS_SOLVER_OPTIONS_PDM"],
    ) == {"max_iter": 100, "tol": 1e-6}
    assert (
        meta.validate_configuration_option_value(
            "HENS_OUTPUT_FOLDER",
            valid_values["HENS_OUTPUT_FOLDER"],
        )
        == "artifacts/hens"
    )
    assert (
        meta.validate_configuration_option_value(
            "HPR_LOAD_MODE",
            valid_values["HPR_LOAD_MODE"],
        )
        == "duty"
    )
    assert meta.validate_configuration_option_value(
        "HPR_LOAD_PERIOD_VALUES",
        valid_values["HPR_LOAD_PERIOD_VALUES"],
    ) == {"peak": 10.0}
    assert (
        meta.validate_configuration_option_value(
            "HENS_PDM_STAGE_PAIR_LIMIT",
            valid_values["HENS_PDM_STAGE_PAIR_LIMIT"],
        )
        == 3
    )
    assert (
        meta.validate_configuration_option_value(
            "HENS_TDM_PARENT_LIMIT",
            valid_values["HENS_TDM_PARENT_LIMIT"],
        )
        is None
    )
    assert (
        meta.validate_configuration_option_value(
            "REPORTING_DECIMAL_PLACES",
            valid_values["REPORTING_DECIMAL_PLACES"],
        )
        == 3
    )
    assert meta.validate_configuration_option_value(
        "THERMAL_DT_CONT",
        valid_values["THERMAL_DT_CONT"],
    ) == pytest.approx(7.5)
    assert (
        meta.validate_configuration_option_value(
            "HENS_LOG_LEVEL",
            valid_values["HENS_LOG_LEVEL"],
        )
        == "DEBUG"
    )
    assert meta.validate_configuration_option_value(
        "HENS_SOLVER_OPTIONS_EVM",
        valid_values["HENS_SOLVER_OPTIONS_EVM"],
    ) == {"bound_push": 1e-4}


def test_public_configuration_rejects_internal_method_selectors():
    with pytest.raises(ValueError, match="Unknown configuration option"):
        meta.validate_configuration_option_value(
            "HPR_TYPE", HeatPumpAndRefrigerationCycle.CascadeCarnot
        )
    with pytest.raises(ValueError, match="Unknown configuration option"):
        meta.validate_configuration_options({"POWER_TURB_MODEL": "Medina-Flores"})


def test_unit_option_maps_use_flat_configuration_defaults():
    options = {name: spec.default for name, spec in meta.CONFIG_FIELD_SPECS.items()}

    assert meta.input_unit_options_to_map(options)["temperature"] == "degC"
    assert meta.input_unit_options_to_map(options)["utility_price"] == "$/MWh"
    assert meta.output_unit_options_to_map(options)["heat_flow"] == "kW"
    assert meta.output_unit_options_to_map(options)["dimensionless"] == "dimensionless"


@pytest.mark.parametrize(
    ("name", "fixture_key", "message"),
    [
        ("INPUT_UNIT_TEMPERATURE", "bad_unit", "compatible"),
        ("HENS_OUTPUT_FOLDER", "bad_output_folder", "must be a string"),
        ("HENS_STAGE_SELECTION", "duplicate_stage_selection", "must be unique"),
        ("HENS_STAGE_PACKING", "bad_stage_packing", "must be one of"),
        ("HENS_SOLVE_TOLERANCE", "bad_positive_float", "positive number"),
        ("HENS_MAX_PARALLEL", "bad_positive_int", "positive integer"),
        ("HENS_RUN_ID", "bad_run_id", "must start with an alphanumeric"),
        ("HENS_SOLVER_PDM", "empty_solver_name", "non-empty string"),
        ("HENS_SOLVER_OPTIONS_PDM", "bad_solver_options", "empty option names"),
        ("HPR_LOAD_PERIOD_VALUES", "bad_period_value", "greater than or equal"),
        ("HPR_MVR_ETA_COMP", "bad_fraction", "less than or equal"),
        ("REPORTING_DECIMAL_PLACES", "bad_int", "integer"),
        ("THERMAL_DT_CONT", "bad_float", "finite number"),
    ],
)
def test_validate_configuration_option_value_rejects_static_invalid_cases(
    name: str,
    fixture_key: str,
    message: str,
):
    value = _fixture()["invalid_values"][fixture_key]

    with pytest.raises(ValueError, match=message):
        meta.validate_configuration_option_value(name, value)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("HENS_APPROACH_TEMPERATURES", [math.inf], "finite number"),
        ("PROBLEM_PERIOD_IDS", [1], "values must be strings"),
        ("HENS_SOLVER_OPTIONS_TDM", [], "must be provided as a dict"),
        ("HPR_LOAD_PERIOD_VALUES", [], "must be provided as a dict"),
        ("PROBLEM_TOP_ZONE_NAME", 123, "must be a string"),
    ],
)
def test_validate_configuration_option_value_rejects_non_fixture_edges(
    name: str,
    value: object,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        meta.validate_configuration_option_value(name, value)


def test_validate_configuration_options_applies_hpr_load_cross_field_rules():
    assert meta.validate_configuration_options(
        {"HPR_LOAD_MODE": "duty", "HPR_LOAD_DUTY": 12.0}
    ) == {"HPR_LOAD_MODE": "duty", "HPR_LOAD_DUTY": 12.0}
    assert meta.validate_configuration_options(
        {
            "HPR_LOAD_MODE": "period_values",
            "HPR_LOAD_PERIOD_VALUES": {"peak": 5.0},
        }
    ) == {
        "HPR_LOAD_MODE": "period_values",
        "HPR_LOAD_PERIOD_VALUES": {"peak": 5.0},
    }

    meta._validate_hpr_load_options({"HPR_LOAD_MODE": None}, provided_keys=set())

    for case in _fixture()["hpr_load_errors"]:
        with pytest.raises(ValueError, match=case["message"]):
            meta.validate_configuration_options(case["options"])


def test_generic_annotation_coercion_fallbacks_cover_custom_metadata_specs():
    dict_spec = ConfigurationFieldSpec(
        annotation=dict[str, Any],
        default={},
        group="test",
        config_path=("test", "mapping"),
    )
    list_any_spec = ConfigurationFieldSpec(
        annotation=list[Any],
        default=[],
        group="test",
        config_path=("test", "items"),
    )
    list_int_spec = ConfigurationFieldSpec(
        annotation=list[int],
        default=[],
        group="test",
        config_path=("test", "stages"),
        numeric_min=1.0,
    )
    list_float_spec = ConfigurationFieldSpec(
        annotation=list[float],
        default=[],
        group="test",
        config_path=("test", "temperatures"),
    )
    fallback_spec = ConfigurationFieldSpec(
        annotation=object,
        default=None,
        group="test",
        config_path=("test", "raw"),
    )

    assert meta._coerce_annotation_value("CUSTOM_DICT", {"a": 1}, dict_spec) == {"a": 1}
    assert meta._coerce_annotation_value("CUSTOM_LIST", [object()], list_any_spec)
    assert meta._coerce_annotation_value("CUSTOM_INT_LIST", [1, 2], list_int_spec) == [
        1,
        2,
    ]
    assert meta._coerce_annotation_value(
        "CUSTOM_FLOAT_LIST",
        [1, 2.5],
        list_float_spec,
    ) == [1.0, 2.5]
    assert meta._coerce_annotation_value("CUSTOM_RAW", "raw", fallback_spec) == "raw"
    assert meta.validate_configuration_option_value("PROBLEM_PERIOD_IDS", ["a"]) == [
        "a"
    ]

    with pytest.raises(ValueError, match="must be provided as a dict"):
        meta._coerce_annotation_value("CUSTOM_DICT", [], dict_spec)
