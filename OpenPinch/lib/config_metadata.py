"""Shared declarative metadata for :mod:`OpenPinch` configuration fields."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import UnionType
from typing import Any, List, get_args, get_origin

from .enums import ZT, BB_Minimiser, HPRcycle, TurbineModel


@dataclass(frozen=True)
class ConfigurationFieldSpec:
    """Describe one editable configuration field and its runtime config path."""

    annotation: Any
    default: Any
    group: str
    config_path: tuple[str, str]
    enum_cls: type[Enum] | None = None
    numeric_min: float | None = None
    numeric_max: float | None = None
    runtime_status: str = "supported"


@dataclass(frozen=True)
class ConfigurationOptionStatus:
    """Classify one incoming configuration option name."""

    name: str
    runtime_status: str


HENS_METHOD_SEQUENCE_VALUES = frozenset(
    {
        "pinch_design_method",
        "thermal_derivative_method",
        "network_evolution_method",
    }
)
HENS_OUTPUT_FORMAT_VALUES = frozenset({"json", "csv", "xlsx"})
HENS_STAGE_PACKING_VALUES = frozenset({"auto", "none", "pdm", "tdm", "all"})
HPR_LOAD_MODE_VALUES = frozenset({"fraction", "duty", "period_values"})
_HENS_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

_INPUT_UNIT_TARGETS = {
    "INPUT_UNIT_TEMPERATURE": "degC",
    "INPUT_UNIT_PRESSURE": "kPa",
    "INPUT_UNIT_ENTHALPY": "kJ/kg",
    "INPUT_UNIT_HEAT_FLOW": "kW",
    "INPUT_UNIT_DELTA_T": "delta_degC",
    "INPUT_UNIT_HTC": "kW/m^2/delta_degC",
    "INPUT_UNIT_PRICE": "$/MWh",
}
_OUTPUT_UNIT_TARGETS = {
    "OUTPUT_UNIT_HEAT_FLOW": "kW",
    "OUTPUT_UNIT_TEMPERATURE": "degC",
    "OUTPUT_UNIT_PERCENT": "%",
    "OUTPUT_UNIT_UTILITY_COST": "$/h",
    "OUTPUT_UNIT_WORK": "kW",
    "OUTPUT_UNIT_AREA": "m^2",
    "OUTPUT_UNIT_CAPITAL_COST": "$",
    "OUTPUT_UNIT_ANNUAL_COST": "$/y",
    "OUTPUT_UNIT_EXERGY": "kW",
    "OUTPUT_UNIT_HPR_COP": "dimensionless",
}
_UNIT_TARGETS = _INPUT_UNIT_TARGETS | _OUTPUT_UNIT_TARGETS


def _spec(
    annotation: Any,
    default: Any,
    group: str,
    field: str,
    *,
    enum_cls: type[Enum] | None = None,
    numeric_min: float | None = None,
    numeric_max: float | None = None,
    runtime_status: str = "supported",
) -> ConfigurationFieldSpec:
    return ConfigurationFieldSpec(
        annotation=annotation,
        default=default,
        group=group,
        config_path=(group, field),
        enum_cls=enum_cls,
        numeric_min=numeric_min,
        numeric_max=numeric_max,
        runtime_status=runtime_status,
    )


# fmt: off
CONFIG_FIELD_SPECS: dict[str, ConfigurationFieldSpec] = {
    # Problem shape and state handling.
    "PROBLEM_TOP_ZONE_NAME": _spec(str, "Site", "problem", "top_zone_name"),
    "PROBLEM_TOP_ZONE_IDENTIFIER": _spec(str, ZT.S.value, "problem", "top_zone_identifier", enum_cls=ZT),
    "PROBLEM_PERIOD_IDS": _spec(List[str], ["0"], "problem", "period_ids"),
    "PROBLEM_PERIOD_WEIGHTS": _spec(List[float], [1.0], "problem", "period_weights", numeric_min=0.0),

    # Explicit input/output units.
    "INPUT_UNIT_TEMPERATURE": _spec(str, "degC", "input_units", "temperature"),
    "INPUT_UNIT_PRESSURE": _spec(str, "kPa", "input_units", "pressure"),
    "INPUT_UNIT_ENTHALPY": _spec(str, "kJ/kg", "input_units", "enthalpy"),
    "INPUT_UNIT_HEAT_FLOW": _spec(str, "kW", "input_units", "heat_flow"),
    "INPUT_UNIT_DELTA_T": _spec(str, "delta_degC", "input_units", "delta_t"),
    "INPUT_UNIT_HTC": _spec(str, "kW/m^2/delta_degC", "input_units", "htc"),
    "INPUT_UNIT_PRICE": _spec(str, "$/MWh", "input_units", "price"),
    "OUTPUT_UNIT_HEAT_FLOW": _spec(str, "kW", "output_units", "heat_flow"),
    "OUTPUT_UNIT_TEMPERATURE": _spec(str, "degC", "output_units", "temperature"),
    "OUTPUT_UNIT_PERCENT": _spec(str, "%", "output_units", "percent"),
    "OUTPUT_UNIT_UTILITY_COST": _spec(str, "$/h", "output_units", "utility_cost"),
    "OUTPUT_UNIT_WORK": _spec(str, "kW", "output_units", "work"),
    "OUTPUT_UNIT_AREA": _spec(str, "m^2", "output_units", "area"),
    "OUTPUT_UNIT_CAPITAL_COST": _spec(str, "$", "output_units", "capital_cost"),
    "OUTPUT_UNIT_ANNUAL_COST": _spec(str, "$/y", "output_units", "annual_cost"),
    "OUTPUT_UNIT_EXERGY": _spec(str, "kW", "output_units", "exergy"),
    "OUTPUT_UNIT_HPR_COP": _spec(str, "dimensionless", "output_units", "hpr_cop"),

    # General runtime controls.
    "REPORTING_DECIMAL_PLACES": _spec(int, 2, "reporting", "decimal_places", numeric_min=0.0),
    "ENV_TEMPERATURE": _spec(float, 15.0, "environment", "temperature"),
    "ENV_PRESSURE": _spec(float, 101.0, "environment", "pressure", numeric_min=0.0),
    "THERMAL_DT_CONT": _spec(float, 5.0, "thermal", "dt_cont", numeric_min=0.0),
    "THERMAL_DT_PHASE_CHANGE": _spec(float, 0.01, "thermal", "dt_phase_change", numeric_min=0.0),
    "THERMAL_HTC": _spec(float, 1.0, "thermal", "htc", numeric_min=0.0),

    # Targeting selectors.
    "TARGETING_DIRECT_SITE_ENABLED": _spec(bool, True, "targeting", "direct_site_enabled"),
    "TARGETING_DIRECT_OPERATION_ENABLED": _spec(bool, False, "targeting", "direct_operation_enabled"),
    "TARGETING_INDIRECT_PROCESS_ENABLED": _spec(bool, False, "targeting", "indirect_process_enabled"),
    "TARGETING_PROCESS_HP_ENABLED": _spec(bool, False, "targeting", "process_hp_enabled"),
    "TARGETING_PROCESS_RFRG_ENABLED": _spec(bool, False, "targeting", "process_rfrg_enabled"),
    "TARGETING_UTILITY_HP_ENABLED": _spec(bool, False, "targeting", "utility_hp_enabled"),
    "TARGETING_UTILITY_RFRG_ENABLED": _spec(bool, False, "targeting", "utility_rfrg_enabled"),
    "TARGETING_TURBINE_ENABLED": _spec(bool, False, "targeting", "turbine_enabled"),
    "TARGETING_EXERGY_ENABLED": _spec(bool, False, "targeting", "exergy_enabled", runtime_status="experimental"),
    "TARGETING_AREA_COST_ENABLED": _spec(bool, False, "targeting", "area_cost_enabled"),

    # Direct integration.
    "DIRECT_BALANCED_CC_ENABLED": _spec(bool, True, "direct", "balanced_cc_enabled"),
    "DIRECT_VERTICAL_GCC_ENABLED": _spec(bool, False, "direct", "vertical_gcc_enabled"),
    "DIRECT_ASSISTED_HT_ENABLED": _spec(bool, False, "direct", "assisted_ht_enabled"),
    "DIRECT_ASSISTED_HT_DT": _spec(float, 10.0, "direct", "assisted_ht_dt", numeric_min=0.0),

    # Costing and economics.
    "COSTING_UTILITY_PRICE": _spec(float, 100.0, "costing", "utility_price", numeric_min=0.0),
    "COSTING_ANNUAL_OP_TIME": _spec(float, 8300.0, "costing", "annual_op_time", numeric_min=0.0),
    "COSTING_HX_UNIT_COST": _spec(float, 0.0, "costing", "hx_unit_cost", numeric_min=0.0),
    "COSTING_HX_AREA_COEFF": _spec(float, 10000.0, "costing", "hx_area_coeff", numeric_min=0.0),
    "COSTING_HX_AREA_EXP": _spec(float, 0.6, "costing", "hx_area_exp", numeric_min=0.0),
    "COSTING_DISCOUNT_RATE": _spec(float, 0.07, "costing", "discount_rate", numeric_min=0.0),
    "COSTING_SERVICE_LIFE": _spec(float, 20.0, "costing", "service_life", numeric_min=0.0),
    "COSTING_HPR_ELE_PRICE": _spec(float, 200.0, "costing", "hpr_ele_price", numeric_min=0.0),
    "COSTING_HPR_PRICE_RATIO_HEAT_TO_ELE": _spec(float, 1.0, "costing", "hpr_price_ratio_heat_to_ele", numeric_min=0.0),
    "COSTING_HPR_PRICE_RATIO_COLD_TO_ELE": _spec(float, 1.0, "costing", "hpr_price_ratio_cold_to_ele", numeric_min=0.0),
    "COSTING_HPR_COMP_FIXED_COST": _spec(float, 0.0, "costing", "hpr_comp_fixed_cost", numeric_min=0.0),
    "COSTING_HPR_COMP_VARIABLE_COST": _spec(float, 10000.0, "costing", "hpr_comp_variable_cost", numeric_min=0.0),
    "COSTING_HPR_COMP_COST_EXP": _spec(float, 1.0, "costing", "hpr_comp_cost_exp", numeric_min=0.0),
    "COSTING_HPR_HX_DUTY_FIXED_COST": _spec(float, 0.0, "costing", "hpr_hx_duty_fixed_cost", numeric_min=0.0),
    "COSTING_HPR_HX_DUTY_VARIABLE_COST": _spec(float, 10000.0, "costing", "hpr_hx_duty_variable_cost", numeric_min=0.0),
    "COSTING_HPR_HX_DUTY_COST_EXP": _spec(float, 1.0, "costing", "hpr_hx_duty_cost_exp", numeric_min=0.0),

    # HEN synthesis.
    "HENS_APPROACH_TEMPERATURES": _spec(List[float], [14.0], "hens", "approach_temperatures"),
    "HENS_DT_CONT_MULTIPLIERS": _spec(List[float] | None, None, "hens", "dt_cont_multipliers"),
    "HENS_DERIVATIVE_THRESHOLDS": _spec(List[float], [0.5], "hens", "derivative_thresholds"),
    "HENS_SYNTHESIS_QUALITY_TIER": _spec(int, 1, "hens", "synthesis_quality_tier", numeric_min=0.0, numeric_max=5.0),
    "HENS_PDM_STAGE_PAIR_LIMIT": _spec(int | None, None, "hens", "pdm_stage_pair_limit", numeric_min=0.0, numeric_max=12.0),
    "HENS_TDM_PARENT_LIMIT": _spec(int | None, None, "hens", "tdm_parent_limit", numeric_min=1.0),
    "HENS_STAGE_PACKING": _spec(str, "auto", "hens", "stage_packing"),
    "HENS_STAGE_SELECTION": _spec(List[int], [1, 2, 3], "hens", "stage_selection"),
    "HENS_METHOD_SEQUENCE": _spec(List[str], ["pinch_design_method", "thermal_derivative_method", "network_evolution_method"], "hens", "method_sequence"),
    "HENS_SOLVER_PDM": _spec(str, "couenne", "hens", "solver_pdm"),
    "HENS_SOLVER_TDM": _spec(str, "couenne", "hens", "solver_tdm"),
    "HENS_SOLVER_EVM": _spec(str, "ipopt-pyomo", "hens", "solver_evm"),
    "HENS_SOLVER_OPTIONS_PDM": _spec(dict[str, Any], {}, "hens", "solver_options_pdm"),
    "HENS_SOLVER_OPTIONS_TDM": _spec(dict[str, Any], {}, "hens", "solver_options_tdm"),
    "HENS_SOLVER_OPTIONS_EVM": _spec(dict[str, Any], {}, "hens", "solver_options_evm"),
    "HENS_SOLVE_TOLERANCE": _spec(float, 1e-3, "hens", "solve_tolerance", numeric_min=0.0),
    "HENS_MAX_PARALLEL": _spec(int, 1, "hens", "max_parallel", numeric_min=1.0),
    "HENS_EVM_N_AD_BRANCHES": _spec(int | None, None, "hens", "evm_n_ad_branches", numeric_min=1.0),
    "HENS_EVM_N_RM_BRANCHES": _spec(int | None, None, "hens", "evm_n_rm_branches", numeric_min=1.0),
    "HENS_LOG_LEVEL": _spec(str, "INFO", "hens", "log_level"),
    "HENS_OUTPUT_FOLDER": _spec(str, "", "hens", "output_folder"),
    "HENS_OUTPUT_FORMATS": _spec(List[str], [], "hens", "output_formats"),
    "HENS_RUN_ID": _spec(str, "default", "hens", "run_id"),
    "HENS_BEST_SOLUTIONS_TO_SAVE": _spec(int, 1, "hens", "best_solutions_to_save", numeric_min=1.0),
    # Heat pump and refrigeration.
    "HPR_TYPE": _spec(str, HPRcycle.CascadeCarnot.value, "hpr", "type", enum_cls=HPRcycle),
    "HPR_LOAD_MODE": _spec(str, "fraction", "hpr", "load_mode"),
    "HPR_LOAD_FRACTION": _spec(float, 1.0, "hpr", "load_fraction", numeric_min=0.0),
    "HPR_LOAD_DUTY": _spec(float | None, None, "hpr", "load_duty", numeric_min=0.0),
    "HPR_LOAD_PERIOD_VALUES": _spec(dict[str, float], {}, "hpr", "load_period_values"),
    "HPR_REFRIGERANTS": _spec(List[str], ["water", "ammonia"], "hpr", "refrigerants"),
    "HPR_REFRIGERANT_SORT_ENABLED": _spec(bool, True, "hpr", "refrigerant_sort_enabled"),
    "HPR_MVR_FLUIDS": _spec(List[str], ["Water"], "hpr", "mvr_fluids"),
    "HPR_MVR_COUNT": _spec(int, 1, "hpr", "mvr_count", numeric_min=1.0),
    "HPR_MVR_ETA_COMP": _spec(float, 0.7, "hpr", "mvr_eta_comp", numeric_min=0.0, numeric_max=1.0),
    "HPR_MVR_ETA_MOTOR": _spec(float, 0.95, "hpr", "mvr_eta_motor", numeric_min=0.0, numeric_max=1.0),
    "HPR_N_COND": _spec(int, 3, "hpr", "n_cond", numeric_min=0.0),
    "HPR_N_EVAP": _spec(int, 2, "hpr", "n_evap", numeric_min=0.0),
    "HPR_ETA_COMP": _spec(float, 0.7, "hpr", "eta_comp", numeric_min=0.0, numeric_max=1.0),
    "HPR_ETA_EXP": _spec(float, 0.7, "hpr", "eta_exp", numeric_min=0.0, numeric_max=1.0),
    "HPR_ETA_II_CARNOT": _spec(float, 0.5, "hpr", "eta_ii_carnot", numeric_min=0.0, numeric_max=1.0),
    "HPR_HE_ETA_II_CARNOT": _spec(float, 0.5, "hpr", "he_eta_ii_carnot", numeric_min=0.0, numeric_max=1.0),
    "HPR_INTEGRATED_EXPANDER_ENABLED": _spec(bool, False, "hpr", "integrated_expander_enabled"),
    "HPR_DT_CONT": _spec(float, 0.0, "hpr", "dt_cont", numeric_min=0.0),
    "HPR_DT_IHX": _spec(float, 0.0, "hpr", "dt_ihx", numeric_min=0.0),
    "HPR_DT_CASCADE_HX": _spec(float, 0.0, "hpr", "dt_cascade_hx", numeric_min=0.0),
    "HPR_DT_ENV_CONT": _spec(float, 10.0, "hpr", "dt_env_cont", numeric_min=0.0),
    "HPR_MAX_MULTISTART": _spec(int, 10, "hpr", "max_multistart", numeric_min=0.0),
    "HPR_ETA_PENALTY": _spec(float, 0.001, "hpr", "eta_penalty", numeric_min=0.0),
    "HPR_RHO_PENALTY": _spec(float, 10.0, "hpr", "rho_penalty", numeric_min=0.0),
    "HPR_BB_MINIMISER": _spec(str, BB_Minimiser.CMAES.value, "hpr", "bb_minimiser", enum_cls=BB_Minimiser),
    "HPR_INITIALISE_SIMULATED_CYCLE": _spec(bool, True, "hpr", "initialise_simulated_cycle"),

    # Direct process MVR and power cogeneration.
    "PROCESS_MVR_ETA_COMP": _spec(float, 0.7, "process_mvr", "eta_comp", numeric_min=0.0, numeric_max=1.0),
    "PROCESS_MVR_ETA_MOTOR": _spec(float, 0.95, "process_mvr", "eta_motor", numeric_min=0.0, numeric_max=1.0),
    "POWER_TURBINE_WORK_ENABLED": _spec(bool, False, "power", "turbine_work_enabled"),
    "POWER_TURB_T_IN": _spec(float, 450.0, "power", "turb_t_in"),
    "POWER_TURB_P_IN": _spec(float, 90.0, "power", "turb_p_in", numeric_min=0.0),
    "POWER_MIN_EFF": _spec(float, 0.1, "power", "min_eff", numeric_min=0.0),
    "POWER_LOAD_FRACTION": _spec(float, 1.0, "power", "load_fraction", numeric_min=0.0),
    "POWER_ETA_MECH": _spec(float, 1.0, "power", "eta_mech", numeric_min=0.0),
    "POWER_TURB_MODEL": _spec(str, TurbineModel.MEDINA_FLORES.value, "power", "turb_model", enum_cls=TurbineModel),
    "POWER_HIGH_P_COND_FLASH_ENABLED": _spec(bool, False, "power", "high_p_cond_flash_enabled"),
}
# fmt: on

CONFIG_ENUMS: dict[str, type[Enum]] = {
    name: spec.enum_cls
    for name, spec in CONFIG_FIELD_SPECS.items()
    if spec.enum_cls is not None
}

CONFIG_GROUPS: dict[str, set[str]] = {}
for _field_name, _field_spec in CONFIG_FIELD_SPECS.items():
    CONFIG_GROUPS.setdefault(_field_spec.group, set()).add(_field_name)

CONFIG_MINIMUMS: dict[str, float] = {
    name: spec.numeric_min
    for name, spec in CONFIG_FIELD_SPECS.items()
    if spec.numeric_min is not None
}


def configuration_group(name: str) -> str:
    """Return the workspace group name for a configuration field."""
    spec = CONFIG_FIELD_SPECS.get(name)
    if spec is None:
        return "problem"
    return spec.group


def configuration_option_status(name: str) -> ConfigurationOptionStatus:
    """Classify one configuration option key by runtime support status."""
    if name in CONFIG_FIELD_SPECS:
        spec = CONFIG_FIELD_SPECS[name]
        return ConfigurationOptionStatus(name=name, runtime_status=spec.runtime_status)
    return ConfigurationOptionStatus(name=name, runtime_status="dead")


def configuration_field_support_level(name: str) -> str:
    """Return the frontend support level for one editable config field."""
    group = configuration_group(name)
    return (
        "stable"
        if group
        in {
            "problem",
            "input_units",
            "output_units",
            "reporting",
            "thermal",
            "targeting",
        }
        else "advanced"
    )


def validate_configuration_options(options: dict) -> dict:
    """Validate user-provided configuration option keys and values."""
    if not isinstance(options, dict):
        raise ValueError("Configuration options must be provided as a dict.")

    statuses = {str(key): configuration_option_status(str(key)) for key in options}
    dead_keys = sorted(
        key for key, status in statuses.items() if status.runtime_status == "dead"
    )
    if dead_keys:
        raise ValueError(f"Unknown configuration option(s): {', '.join(dead_keys)}.")

    validated = {
        str(key): validate_configuration_option_value(str(key), value)
        for key, value in options.items()
    }
    effective_options = {
        name: spec.default for name, spec in CONFIG_FIELD_SPECS.items()
    } | validated
    _validate_hpr_load_options(effective_options, provided_keys=set(validated))
    return validated


def validate_configuration_option_value(name: str, value: Any) -> Any:
    """Validate one supported configuration value."""
    spec = CONFIG_FIELD_SPECS[name]
    if name in _UNIT_TARGETS:
        return _unit_string(name, value, _UNIT_TARGETS[name])
    if name in {"HENS_APPROACH_TEMPERATURES", "HENS_DT_CONT_MULTIPLIERS"}:
        if value is None and name == "HENS_DT_CONT_MULTIPLIERS":
            return None
        return _positive_float_grid(name, value)
    if name == "HENS_DERIVATIVE_THRESHOLDS":
        return _positive_float_grid(name, value)
    if name == "HENS_STAGE_SELECTION":
        return _positive_unique_int_grid(name, value)
    if name == "HENS_METHOD_SEQUENCE":
        return _string_choice_grid(
            name,
            value,
            HENS_METHOD_SEQUENCE_VALUES,
            allow_empty=False,
        )
    if name == "HENS_OUTPUT_FORMATS":
        return _string_choice_grid(
            name,
            value,
            HENS_OUTPUT_FORMAT_VALUES,
            allow_empty=True,
        )
    if name == "HENS_STAGE_PACKING":
        return _string_choice(name, value, HENS_STAGE_PACKING_VALUES)
    if name == "HENS_SOLVE_TOLERANCE":
        return _positive_float(name, value)
    if name in {
        "HENS_MAX_PARALLEL",
        "HENS_BEST_SOLUTIONS_TO_SAVE",
    }:
        return _positive_int(name, value)
    if name in {
        "HENS_EVM_N_AD_BRANCHES",
        "HENS_EVM_N_RM_BRANCHES",
    }:
        return _positive_int_or_none(name, value)
    if name == "HENS_RUN_ID":
        return _run_id(name, value)
    if name in {
        "HENS_SOLVER_PDM",
        "HENS_SOLVER_TDM",
        "HENS_SOLVER_EVM",
        "HENS_LOG_LEVEL",
    }:
        return _non_empty_string(name, value)
    if name in {
        "HENS_SOLVER_OPTIONS_PDM",
        "HENS_SOLVER_OPTIONS_TDM",
        "HENS_SOLVER_OPTIONS_EVM",
    }:
        return _solver_options(name, value)
    if name == "HENS_OUTPUT_FOLDER":
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string.")
        return value
    if name == "HPR_LOAD_MODE":
        return _string_choice(name, value, HPR_LOAD_MODE_VALUES)
    if name == "HPR_LOAD_PERIOD_VALUES":
        return _float_mapping(name, value, numeric_min=0.0)

    if spec.enum_cls is not None:
        value = _enum_value(name, value, spec.enum_cls)
    return _coerce_annotation_value(name, value, spec)


def input_unit_options_to_map(options: Mapping[str, Any]) -> dict[str, str]:
    """Return unit-system input overrides from flat explicit unit options."""
    return {
        "temperature": str(options["INPUT_UNIT_TEMPERATURE"]),
        "pressure": str(options["INPUT_UNIT_PRESSURE"]),
        "enthalpy": str(options["INPUT_UNIT_ENTHALPY"]),
        "heat_flow": str(options["INPUT_UNIT_HEAT_FLOW"]),
        "delta_temperature": str(options["INPUT_UNIT_DELTA_T"]),
        "temperature_difference": str(options["INPUT_UNIT_DELTA_T"]),
        "heat_transfer_coefficient": str(options["INPUT_UNIT_HTC"]),
        "utility_price": str(options["INPUT_UNIT_PRICE"]),
        "price": str(options["INPUT_UNIT_PRICE"]),
    }


def output_unit_options_to_map(options: Mapping[str, Any]) -> dict[str, str]:
    """Return unit-system output overrides from flat explicit unit options."""
    return {
        "heat_flow": str(options["OUTPUT_UNIT_HEAT_FLOW"]),
        "temperature": str(options["OUTPUT_UNIT_TEMPERATURE"]),
        "percent": str(options["OUTPUT_UNIT_PERCENT"]),
        "fraction": str(options["OUTPUT_UNIT_PERCENT"]),
        "utility_cost": str(options["OUTPUT_UNIT_UTILITY_COST"]),
        "work_target": str(options["OUTPUT_UNIT_WORK"]),
        "area": str(options["OUTPUT_UNIT_AREA"]),
        "capital_cost": str(options["OUTPUT_UNIT_CAPITAL_COST"]),
        "currency": str(options["OUTPUT_UNIT_CAPITAL_COST"]),
        "annual_cost": str(options["OUTPUT_UNIT_ANNUAL_COST"]),
        "exergy": str(options["OUTPUT_UNIT_EXERGY"]),
        "cop": str(options["OUTPUT_UNIT_HPR_COP"]),
        "dimensionless": str(options["OUTPUT_UNIT_HPR_COP"]),
    }


def _coerce_annotation_value(
    name: str,
    value: Any,
    spec: ConfigurationFieldSpec,
) -> Any:
    annotation = spec.annotation
    origin = get_origin(annotation)
    args = get_args(annotation)

    if _is_optional(annotation):
        if value is None:
            return None
        annotation = next(arg for arg in args if arg is not type(None))
        return _coerce_annotation_value(
            name,
            value,
            ConfigurationFieldSpec(
                annotation=annotation,
                default=spec.default,
                group=spec.group,
                config_path=spec.config_path,
                enum_cls=None,
                numeric_min=spec.numeric_min,
                numeric_max=spec.numeric_max,
                runtime_status=spec.runtime_status,
            ),
        )

    if annotation is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean.")
        return value
    if annotation is int:
        numeric = _int(name, value)
        return _check_numeric_bounds(name, numeric, spec)
    if annotation is float:
        numeric = _float(name, value)
        return _check_numeric_bounds(name, numeric, spec)
    if annotation is str:
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string.")
        return value
    if origin in {list, List} or "List" in str(annotation):
        item_type = args[0] if args else Any
        return [
            _coerce_list_item(name, item, item_type, spec)
            for item in _sequence(name, value, allow_empty=True)
        ]
    if origin is dict or str(annotation).startswith("dict"):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be provided as a dict.")
        return dict(value)
    return value


def _coerce_list_item(
    name: str,
    value: Any,
    item_type: Any,
    spec: ConfigurationFieldSpec,
) -> Any:
    if item_type is float:
        return _check_numeric_bounds(name, _float(name, value), spec)
    if item_type is int:
        return _check_numeric_bounds(name, _int(name, value), spec)
    if item_type is str:
        if not isinstance(value, str):
            raise ValueError(f"{name} values must be strings.")
        return value
    return value


def _is_optional(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin in {UnionType, None} and type(None) in get_args(annotation)


def _enum_value(name: str, value: Any, enum_cls: type[Enum]) -> Any:
    if isinstance(value, enum_cls):
        return value.value
    allowed = {item.value for item in enum_cls}
    if value not in allowed:
        allowed_str = ", ".join(sorted(str(item) for item in allowed))
        raise ValueError(
            f"Invalid value for configuration option {name}: {value!r}. "
            f"Allowed values are: {allowed_str}."
        )
    return value


def _validate_hpr_load_options(
    options: Mapping[str, Any],
    *,
    provided_keys: set[str],
) -> None:
    mode = options.get("HPR_LOAD_MODE")
    if mode is None:
        return
    if mode == "fraction" and options.get("HPR_LOAD_DUTY") is not None:
        raise ValueError(
            "HPR_LOAD_DUTY cannot be supplied when HPR_LOAD_MODE is 'fraction'."
        )
    if mode == "fraction" and "HPR_LOAD_PERIOD_VALUES" in provided_keys:
        raise ValueError(
            "HPR_LOAD_PERIOD_VALUES cannot be supplied when HPR_LOAD_MODE is 'fraction'."
        )
    if mode == "duty" and options.get("HPR_LOAD_DUTY") is None:
        raise ValueError("HPR_LOAD_DUTY is required when HPR_LOAD_MODE is 'duty'.")
    if mode == "duty" and "HPR_LOAD_FRACTION" in provided_keys:
        raise ValueError(
            "HPR_LOAD_FRACTION cannot be supplied when HPR_LOAD_MODE is 'duty'."
        )
    if mode == "period_values" and not options.get("HPR_LOAD_PERIOD_VALUES"):
        raise ValueError(
            "HPR_LOAD_PERIOD_VALUES is required when HPR_LOAD_MODE is 'period_values'."
        )
    if mode == "period_values" and "HPR_LOAD_FRACTION" in provided_keys:
        raise ValueError(
            "HPR_LOAD_FRACTION cannot be supplied when HPR_LOAD_MODE is 'period_values'."
        )
    if mode == "period_values" and "HPR_LOAD_DUTY" in provided_keys:
        raise ValueError(
            "HPR_LOAD_DUTY cannot be supplied when HPR_LOAD_MODE is 'period_values'."
        )


def _sequence(name: str, value: Any, *, allow_empty: bool) -> Sequence:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be provided as a list.")
    if not allow_empty and not value:
        raise ValueError(f"{name} must contain at least one value.")
    return value


def _positive_float_grid(name: str, value: Any) -> list[float]:
    return [
        _positive_float(name, item)
        for item in _sequence(name, value, allow_empty=False)
    ]


def _solver_options(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be provided as a dict.")
    options: dict[str, Any] = {}
    for key, option_value in value.items():
        option_name = str(key).strip()
        if not option_name:
            raise ValueError(f"{name} cannot contain empty option names.")
        options[option_name] = option_value
    return options


def _float_mapping(
    name: str,
    value: Any,
    *,
    numeric_min: float | None,
) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be provided as a dict.")
    return {
        str(key): _check_numeric_bounds(
            name,
            _float(name, item),
            ConfigurationFieldSpec(
                float,
                None,
                "hpr",
                ("hpr", "load_period_values"),
                numeric_min=numeric_min,
            ),
        )
        for key, item in value.items()
    }


def _positive_unique_int_grid(name: str, value: Any) -> list[int]:
    stages = [
        _positive_int(name, item) for item in _sequence(name, value, allow_empty=False)
    ]
    if len(set(stages)) != len(stages):
        raise ValueError(f"{name} values must be unique.")
    return stages


def _string_choice_grid(
    name: str,
    value: Any,
    choices: frozenset[str],
    *,
    allow_empty: bool,
) -> list[str]:
    values = list(_sequence(name, value, allow_empty=allow_empty))
    invalid = [
        item for item in values if not isinstance(item, str) or item not in choices
    ]
    if invalid:
        choices_text = ", ".join(sorted(choices))
        raise ValueError(f"{name} values must be one of: {choices_text}.")
    return values


def _string_choice(name: str, value: Any, choices: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        choices_text = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {choices_text}.")
    return value


def _positive_float(name: str, value: Any) -> float:
    numeric_value = _float(name, value)
    if numeric_value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return numeric_value


def _positive_int(name: str, value: Any) -> int:
    numeric_value = _int(name, value)
    if numeric_value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return numeric_value


def _positive_int_or_none(name: str, value: Any) -> int | None:
    if value is None:
        return None
    return _positive_int(name, value)


def _float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be a finite number.")
    return numeric_value


def _int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _check_numeric_bounds(
    name: str,
    value: float | int,
    spec: ConfigurationFieldSpec,
) -> Any:
    if spec.numeric_min is not None and value < spec.numeric_min:
        raise ValueError(f"{name} must be greater than or equal to {spec.numeric_min}.")
    if spec.numeric_max is not None and value > spec.numeric_max:
        raise ValueError(f"{name} must be less than or equal to {spec.numeric_max}.")
    return value


def _run_id(name: str, value: Any) -> str:
    value = _non_empty_string(name, value)
    if _HENS_RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{name} must start with an alphanumeric character and contain only "
            "letters, numbers, underscores, hyphens, or periods."
        )
    return value


def _non_empty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _unit_string(name: str, value: Any, target_unit: str) -> str:
    text = _non_empty_string(name, value)
    try:
        from ..classes.value import Value

        Value(1.0, text).to(target_unit)
    except Exception as exc:
        raise ValueError(f"{name} must be compatible with {target_unit}.") from exc
    return text
