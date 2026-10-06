"""Shared declarative metadata for :mod:`OpenPinch` configuration fields."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from functools import partial
from types import UnionType
from typing import Any, List, get_args, get_origin

from .enums import BB_Minimiser, HeatPumpAndRefrigerationCycle, TurbineModel, ZoneType


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
    # When true, the value must be strictly greater than zero.
    positive: bool = False
    runtime_status: str = "supported"
    validator: Callable[[str, Any], Any] | None = dataclass_field(
        default=None, compare=False, repr=False
    )


@dataclass(frozen=True)
class ConfigurationOptionStatus:
    """Classify one incoming configuration option name."""

    name: str
    runtime_status: str


HENS_OUTPUT_FORMAT_VALUES = frozenset({"json", "csv", "xlsx"})
HENS_STAGE_PACKING_VALUES = frozenset({"auto", "none", "pdm", "tdm", "all"})
HPR_LOAD_MODE_VALUES = frozenset({"fraction", "duty", "period_values"})
_HENS_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


# Value validators referenced by the field table below.


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


def _optional_positive_float_grid(name: str, value: Any) -> list[float] | None:
    if value is None:
        return None
    return _positive_float_grid(name, value)


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
    if spec.positive and value <= 0:
        raise ValueError(f"{name} must be greater than 0.")
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


def _string(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    return value


def _non_empty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _unit_string(name: str, value: Any, target_unit: str) -> str:
    text = _non_empty_string(name, value)
    try:
        from ..domain.value import Value

        Value(1.0, text).to(target_unit)
    except Exception as exc:
        raise ValueError(f"{name} must be compatible with {target_unit}.") from exc
    return text


def _spec(
    annotation: Any,
    default: Any,
    group: str,
    field: str,
    *,
    enum_cls: type[Enum] | None = None,
    numeric_min: float | None = None,
    numeric_max: float | None = None,
    positive: bool = False,
    runtime_status: str = "supported",
    validator: Callable[[str, Any], Any] | None = None,
) -> ConfigurationFieldSpec:
    return ConfigurationFieldSpec(
        annotation=annotation,
        default=default,
        group=group,
        config_path=(group, field),
        enum_cls=enum_cls,
        numeric_min=numeric_min,
        numeric_max=numeric_max,
        positive=positive,
        runtime_status=runtime_status,
        validator=validator,
    )


def _unit_spec(default: str, group: str, field: str) -> ConfigurationFieldSpec:
    """Return a unit-string field whose values must convert to ``default``."""
    return _spec(
        str,
        default,
        group,
        field,
        validator=partial(_unit_string, target_unit=default),
    )


# fmt: off
CONFIG_FIELD_SPECS: dict[str, ConfigurationFieldSpec] = {
    # Problem shape and state handling.
    "PROBLEM_TOP_ZONE_NAME": _spec(str, "Site", "problem", "top_zone_name"),
    "PROBLEM_TOP_ZONE_IDENTIFIER": _spec(str, ZoneType.S.value, "problem", "top_zone_identifier", enum_cls=ZoneType),
    "PROBLEM_PERIOD_IDS": _spec(List[str], ["0"], "problem", "period_ids"),
    "PROBLEM_PERIOD_WEIGHTS": _spec(List[float], [1.0], "problem", "period_weights", numeric_min=0.0),

    # Explicit input/output units.
    "INPUT_UNIT_TEMPERATURE": _unit_spec("degC", "input_units", "temperature"),
    "INPUT_UNIT_PRESSURE": _unit_spec("kPa", "input_units", "pressure"),
    "INPUT_UNIT_ENTHALPY": _unit_spec("kJ/kg", "input_units", "enthalpy"),
    "INPUT_UNIT_HEAT_FLOW": _unit_spec("kW", "input_units", "heat_flow"),
    "INPUT_UNIT_DELTA_T": _unit_spec("delta_degC", "input_units", "delta_t"),
    "INPUT_UNIT_HTC": _unit_spec("kW/m^2/delta_degC", "input_units", "htc"),
    "INPUT_UNIT_PRICE": _unit_spec("$/MWh", "input_units", "price"),
    "OUTPUT_UNIT_HEAT_FLOW": _unit_spec("kW", "output_units", "heat_flow"),
    "OUTPUT_UNIT_TEMPERATURE": _unit_spec("degC", "output_units", "temperature"),
    "OUTPUT_UNIT_PERCENT": _unit_spec("%", "output_units", "percent"),
    "OUTPUT_UNIT_UTILITY_COST": _unit_spec("$/h", "output_units", "utility_cost"),
    "OUTPUT_UNIT_WORK": _unit_spec("kW", "output_units", "work"),
    "OUTPUT_UNIT_AREA": _unit_spec("m^2", "output_units", "area"),
    "OUTPUT_UNIT_CAPITAL_COST": _unit_spec("$", "output_units", "capital_cost"),
    "OUTPUT_UNIT_ANNUAL_COST": _unit_spec("$/y", "output_units", "annual_cost"),
    "OUTPUT_UNIT_EXERGY": _unit_spec("kW", "output_units", "exergy"),
    "OUTPUT_UNIT_HPR_COP": _unit_spec("dimensionless", "output_units", "hpr_cop"),

    # General runtime controls.
    "REPORTING_DECIMAL_PLACES": _spec(int, 2, "reporting", "decimal_places", numeric_min=0.0),
    "REPORTING_DEBUG_ENABLED": _spec(bool, False, "reporting", "debug_enabled"),
    "ENV_TEMPERATURE": _spec(float, 15.0, "environment", "temperature"),
    "ENV_PRESSURE": _spec(float, 101.0, "environment", "pressure", numeric_min=0.0),
    "THERMAL_DT_CONT": _spec(float, 5.0, "thermal", "dt_cont", numeric_min=0.0),
    "THERMAL_DT_PHASE_CHANGE": _spec(float, 0.01, "thermal", "dt_phase_change", numeric_min=0.0),
    "THERMAL_HTC": _spec(float, 1.0, "thermal", "htc", numeric_min=0.0, positive=True),

    # Direct integration.
    "DIRECT_BALANCED_CC_ENABLED": _spec(bool, True, "direct", "balanced_cc_enabled"),
    "DIRECT_VERTICAL_GCC_ENABLED": _spec(bool, False, "direct", "vertical_gcc_enabled"),
    "DIRECT_ASSISTED_HT_ENABLED": _spec(bool, False, "direct", "assisted_ht_enabled"),
    "DIRECT_ASSISTED_HT_DT": _spec(float, 10.0, "direct", "assisted_ht_dt", numeric_min=0.0),

    # Costing and economics.
    "COSTING_UTILITY_PRICE": _spec(float, 100.0, "costing", "utility_price", numeric_min=0.0),
    "COSTING_ANNUAL_OP_TIME": _spec(float, 8300.0, "costing", "annual_op_time", numeric_min=0.0, positive=True),
    "COSTING_HX_UNIT_COST": _spec(float, 0.0, "costing", "hx_unit_cost", numeric_min=0.0),
    "COSTING_HX_AREA_COEFF": _spec(float, 10000.0, "costing", "hx_area_coeff", numeric_min=0.0),
    "COSTING_HX_AREA_EXP": _spec(float, 0.6, "costing", "hx_area_exp", numeric_min=0.0),
    "COSTING_DISCOUNT_RATE": _spec(float, 0.07, "costing", "discount_rate", numeric_min=0.0),
    "COSTING_SERVICE_LIFE": _spec(float, 20.0, "costing", "service_life", numeric_min=1.0),
    "COSTING_HPR_ELE_PRICE": _spec(float, 100.0, "costing", "hpr_ele_price", numeric_min=0.0),
    # HPR prices, relative to the electricity price.
    "COSTING_HPR_PRICE_RATIO_HEAT_TO_ELE": _spec(float, 1.0, "costing", "hpr_price_ratio_heat_to_ele", numeric_min=0.0),
    "COSTING_HPR_PRICE_RATIO_COOLING_WATER_TO_ELE": _spec(float, 0.025, "costing", "hpr_price_ratio_cooling_water_to_ele", numeric_min=0.0),
    # Default cold utilities: cooling water, then refrigeration below it.
    "COSTING_HPR_COOLING_WATER_TEMPERATURE": _spec(float, 25.0, "costing", "hpr_cooling_water_temperature"),
    "COSTING_HPR_COOLING_WATER_DT_MIN": _spec(float, 5.0, "costing", "hpr_cooling_water_dt_min", numeric_min=0.0),
    "COSTING_HPR_REFRIGERATION_ETA_II": _spec(float, 0.4, "costing", "hpr_refrigeration_eta_ii", numeric_min=0.0, positive=True),
    "COSTING_HPR_REFRIGERATION_DT": _spec(float, 5.0, "costing", "hpr_refrigeration_dt", numeric_min=0.0),
    # Installed HPR capital: C = F_inst * C_eq * (Q_cap / 1 MW)^exp
    #   * (fixed_share + stage_share * n_stages) * f_T,
    # f_T = 1 + temp_factor * max(0, T_hot_max - temp_base) / 100 K.
    # C_eq is fitted to the IEA HPT Project 68 cost data (2025 USD).
    "COSTING_HPR_EQUIPMENT_COST": _spec(float, 485000.0, "costing", "hpr_equipment_cost", numeric_min=0.0),
    "COSTING_HPR_INSTALLATION_FACTOR": _spec(float, 2.3, "costing", "hpr_installation_factor", numeric_min=0.0),
    "COSTING_HPR_COST_EXP": _spec(float, 0.7, "costing", "hpr_cost_exp", numeric_min=0.0),
    "COSTING_HPR_COST_FIXED_SHARE": _spec(float, 0.7, "costing", "hpr_cost_fixed_share", numeric_min=0.0),
    "COSTING_HPR_COST_STAGE_SHARE": _spec(float, 0.3, "costing", "hpr_cost_stage_share", numeric_min=0.0),
    "COSTING_HPR_COST_TEMP_FACTOR": _spec(float, 0.4, "costing", "hpr_cost_temp_factor", numeric_min=0.0),
    "COSTING_HPR_COST_TEMP_BASE": _spec(float, 75.0, "costing", "hpr_cost_temp_base"),
    "COSTING_HPR_CAPITAL_RECOVERY_ENABLED": _spec(bool, True, "costing", "hpr_capital_recovery_enabled"),
    # Installed capital of the default utilities the heat pump displaces, per
    # kW of capacity, annualised with the same capital recovery factor.
    "COSTING_HPR_UTILITY_CAPITAL_RECOVERY_ENABLED": _spec(bool, True, "costing", "hpr_utility_capital_recovery_enabled"),
    "COSTING_HPR_HOT_UTILITY_CAPITAL_COST": _spec(float, 750.0, "costing", "hpr_hot_utility_capital_cost", numeric_min=0.0),
    "COSTING_HPR_REFRIGERATION_CAPITAL_COST": _spec(float, 1500.0, "costing", "hpr_refrigeration_capital_cost", numeric_min=0.0),
    # ORC prices and installed capital: C = F_inst * C_eq * (W_net / 1 MW)^exp
    # per unit, about $3,000/kW installed at 1 MW (2025 USD, +/-30 %).
    "COSTING_ORC_ELE_PRICE": _spec(float, 100.0, "costing", "orc_ele_price", numeric_min=0.0),
    "COSTING_ORC_COOLING_PRICE": _spec(float, 2.5, "costing", "orc_cooling_price", numeric_min=0.0),
    "COSTING_ORC_EQUIPMENT_COST": _spec(float, 2.3e6, "costing", "orc_equipment_cost", numeric_min=0.0),
    "COSTING_ORC_INSTALLATION_FACTOR": _spec(float, 1.3, "costing", "orc_installation_factor", numeric_min=0.0),
    "COSTING_ORC_COST_EXP": _spec(float, 0.75, "costing", "orc_cost_exp", numeric_min=0.0),
    "COSTING_ORC_CAPITAL_RECOVERY_ENABLED": _spec(bool, True, "costing", "orc_capital_recovery_enabled"),

    # HEN synthesis.
    "HENS_APPROACH_TEMPERATURES": _spec(List[float], [14.0], "hens", "approach_temperatures", validator=_positive_float_grid),
    "HENS_DT_CONT_MULTIPLIERS": _spec(List[float] | None, None, "hens", "dt_cont_multipliers", validator=_optional_positive_float_grid),
    "HENS_DERIVATIVE_THRESHOLDS": _spec(List[float], [0.5], "hens", "derivative_thresholds", validator=_positive_float_grid),
    "HENS_SYNTHESIS_QUALITY_TIER": _spec(int, 1, "hens", "synthesis_quality_tier", numeric_min=0.0, numeric_max=5.0),
    "HENS_PDM_STAGE_PAIR_LIMIT": _spec(int | None, None, "hens", "pdm_stage_pair_limit", numeric_min=0.0, numeric_max=12.0),
    "HENS_TDM_PARENT_LIMIT": _spec(int | None, None, "hens", "tdm_parent_limit", numeric_min=1.0),
    "HENS_STAGE_PACKING": _spec(str, "auto", "hens", "stage_packing", validator=partial(_string_choice, choices=HENS_STAGE_PACKING_VALUES)),
    "HENS_STAGE_SELECTION": _spec(List[int], [1, 2, 3], "hens", "stage_selection", validator=_positive_unique_int_grid),
    "HENS_SOLVER_PDM": _spec(str, "couenne", "hens", "solver_pdm", validator=_non_empty_string),
    "HENS_SOLVER_TDM": _spec(str, "couenne", "hens", "solver_tdm", validator=_non_empty_string),
    "HENS_SOLVER_EVM": _spec(str, "ipopt-pyomo", "hens", "solver_evm", validator=_non_empty_string),
    "HENS_SOLVER_OPTIONS_PDM": _spec(dict[str, Any], {}, "hens", "solver_options_pdm", validator=_solver_options),
    "HENS_SOLVER_OPTIONS_TDM": _spec(dict[str, Any], {}, "hens", "solver_options_tdm", validator=_solver_options),
    "HENS_SOLVER_OPTIONS_EVM": _spec(dict[str, Any], {}, "hens", "solver_options_evm", validator=_solver_options),
    "HENS_SOLVE_TOLERANCE": _spec(float, 1e-3, "hens", "solve_tolerance", numeric_min=0.0, validator=_positive_float),
    "HENS_MAX_PARALLEL": _spec(int, 1, "hens", "max_parallel", numeric_min=1.0, validator=_positive_int),
    "HENS_EVM_N_AD_BRANCHES": _spec(int | None, None, "hens", "evm_n_ad_branches", numeric_min=1.0, validator=_positive_int_or_none),
    "HENS_EVM_N_RM_BRANCHES": _spec(int | None, None, "hens", "evm_n_rm_branches", numeric_min=1.0, validator=_positive_int_or_none),
    "HENS_EVM_BEAM_WIDTH": _spec(int, 4, "hens", "evm_beam_width", numeric_min=1.0, validator=_positive_int),
    "HENS_LOG_LEVEL": _spec(str, "INFO", "hens", "log_level", validator=_non_empty_string),
    "HENS_OUTPUT_FOLDER": _spec(str, "", "hens", "output_folder", validator=_string),
    "HENS_OUTPUT_FORMATS": _spec(List[str], [], "hens", "output_formats", validator=partial(_string_choice_grid, choices=HENS_OUTPUT_FORMAT_VALUES, allow_empty=True)),
    "HENS_RUN_ID": _spec(str, "default", "hens", "run_id", validator=_run_id),
    "HENS_BEST_SOLUTIONS_TO_SAVE": _spec(int, 1, "hens", "best_solutions_to_save", numeric_min=1.0, validator=_positive_int),
    # Heat pump and refrigeration.
    "HPR_TYPE": _spec(str, HeatPumpAndRefrigerationCycle.CascadeCarnot.value, "hpr", "type", enum_cls=HeatPumpAndRefrigerationCycle),
    "HPR_LOAD_MODE": _spec(str, "fraction", "hpr", "load_mode", validator=partial(_string_choice, choices=HPR_LOAD_MODE_VALUES)),
    "HPR_LOAD_FRACTION": _spec(float, 1.0, "hpr", "load_fraction", numeric_min=0.0, numeric_max=1.0),
    "HPR_LOAD_DUTY": _spec(float | None, None, "hpr", "load_duty", numeric_min=0.0),
    "HPR_LOAD_PERIOD_VALUES": _spec(dict[str, float], {}, "hpr", "load_period_values", validator=partial(_float_mapping, numeric_min=0.0)),
    "HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": _spec(bool, False, "hpr", "multiperiod_optimization_enabled"),
    "HPR_REFRIGERANTS": _spec(List[str], ["water", "ammonia"], "hpr", "refrigerants"),
    "HPR_REFRIGERANT_SORT_ENABLED": _spec(bool, True, "hpr", "refrigerant_sort_enabled"),
    "HPR_MVR_FLUIDS": _spec(List[str], ["Water"], "hpr", "mvr_fluids"),
    "HPR_MVR_COUNT": _spec(int, 1, "hpr", "mvr_count", numeric_min=1.0),
    "HPR_MVR_ETA_COMP": _spec(float, 0.7, "hpr", "mvr_eta_comp", numeric_min=0.0, numeric_max=1.0, positive=True),
    "HPR_MVR_ETA_MOTOR": _spec(float, 0.95, "hpr", "mvr_eta_motor", numeric_min=0.0, numeric_max=1.0, positive=True),
    "HPR_N_COND": _spec(int, 3, "hpr", "n_cond", numeric_min=0.0),
    "HPR_N_EVAP": _spec(int, 2, "hpr", "n_evap", numeric_min=0.0),
    "HPR_ETA_COMP": _spec(float, 0.7, "hpr", "eta_comp", numeric_min=0.0, numeric_max=1.0, positive=True),
    "HPR_ETA_EXP": _spec(float, 0.7, "hpr", "eta_exp", numeric_min=0.0, numeric_max=1.0),
    "HPR_ETA_II_CARNOT": _spec(float, 0.5, "hpr", "eta_ii_carnot", numeric_min=0.0, numeric_max=1.0, positive=True),
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

    # Organic Rankine cycle (ORC) on the surplus below the pinch.
    "ORC_N_STAGES": _spec(int, 1, "orc", "n_stages", numeric_min=1.0),
    "ORC_ETA_II_CARNOT": _spec(float, 0.5, "orc", "eta_ii_carnot", numeric_min=0.0, numeric_max=1.0, positive=True),
    "ORC_DT_CONT": _spec(float, 5.0, "orc", "dt_cont", numeric_min=0.0),
    "ORC_T_COND": _spec(float, 30.0, "orc", "t_cond"),
    "ORC_MIN_LIFT": _spec(float, 10.0, "orc", "min_lift", numeric_min=0.0),
    "ORC_LOAD_FRACTION": _spec(float, 1.0, "orc", "load_fraction", numeric_min=0.0, numeric_max=1.0),
    "ORC_MAX_MULTISTART": _spec(int, 5, "orc", "max_multistart", numeric_min=1.0),
    "ORC_BB_MINIMISER": _spec(str, BB_Minimiser.DA.value, "orc", "bb_minimiser", enum_cls=BB_Minimiser),

    # Direct process MVR and power cogeneration.
    "PROCESS_MVR_ETA_COMP": _spec(float, 0.7, "process_mvr", "eta_comp", numeric_min=0.0, numeric_max=1.0, positive=True),
    "PROCESS_MVR_ETA_MOTOR": _spec(float, 0.95, "process_mvr", "eta_motor", numeric_min=0.0, numeric_max=1.0, positive=True),
    "POWER_TURBINE_WORK_ENABLED": _spec(bool, False, "power", "turbine_work_enabled"),
    "POWER_TURB_T_IN": _spec(float, 450.0, "power", "turb_t_in"),
    "POWER_TURB_P_IN": _spec(float, 90.0, "power", "turb_p_in", numeric_min=0.0),
    "POWER_MIN_EFF": _spec(float, 0.1, "power", "min_eff", numeric_min=0.0),
    "POWER_LOAD_FRACTION": _spec(float, 1.0, "power", "load_fraction", numeric_min=0.0),
    "POWER_ETA_MECH": _spec(float, 1.0, "power", "eta_mech", numeric_min=0.0, positive=True),
    "POWER_TURB_MODEL": _spec(str, TurbineModel.MEDINA_FLORES.value, "power", "turb_model", enum_cls=TurbineModel),
    "POWER_HIGH_P_COND_FLASH_ENABLED": _spec(bool, False, "power", "high_p_cond_flash_enabled"),
}
# fmt: on

INTERNAL_METHOD_OPTION_KEYS = frozenset({"HPR_TYPE", "POWER_TURB_MODEL"})
USER_CONFIG_FIELD_SPECS: dict[str, ConfigurationFieldSpec] = {
    name: spec
    for name, spec in CONFIG_FIELD_SPECS.items()
    if name not in INTERNAL_METHOD_OPTION_KEYS
}


def configuration_group(name: str) -> str:
    """Return the workspace group name for a configuration field."""
    spec = CONFIG_FIELD_SPECS.get(name)
    if spec is None:
        return "problem"
    return spec.group


def configuration_option_status(name: str) -> ConfigurationOptionStatus:
    """Classify one configuration option key by runtime support status."""
    if name in USER_CONFIG_FIELD_SPECS:
        spec = USER_CONFIG_FIELD_SPECS[name]
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
    return _validate_configuration_options(options, allow_method_selectors=False)


def validate_internal_configuration_options(options: dict) -> dict:
    """Validate call-local overrides, including private method selectors."""
    return _validate_configuration_options(options, allow_method_selectors=True)


def _validate_configuration_options(
    options: dict,
    *,
    allow_method_selectors: bool,
) -> dict:
    if not isinstance(options, dict):
        raise ValueError("Configuration options must be provided as a dict.")

    supported = (
        CONFIG_FIELD_SPECS if allow_method_selectors else USER_CONFIG_FIELD_SPECS
    )
    dead_keys = sorted(str(key) for key in options if str(key) not in supported)
    if dead_keys:
        raise ValueError(f"Unknown configuration option(s): {', '.join(dead_keys)}.")

    validated = {
        str(key): _validate_configuration_option_value(str(key), value)
        for key, value in options.items()
    }
    effective_options = {
        name: spec.default for name, spec in CONFIG_FIELD_SPECS.items()
    } | validated
    _validate_hpr_load_options(effective_options, provided_keys=set(validated))
    _validate_period_options(effective_options)
    return validated


def _validate_period_options(options: dict) -> None:
    """Require unique, non-empty period ids and no more weights than periods."""
    period_ids = [str(item) for item in options["PROBLEM_PERIOD_IDS"]]
    if not period_ids:
        raise ValueError("PROBLEM_PERIOD_IDS must list at least one period.")
    if any(not period_id.strip() for period_id in period_ids):
        raise ValueError("PROBLEM_PERIOD_IDS must not contain empty ids.")
    duplicates = sorted({item for item in period_ids if period_ids.count(item) > 1})
    if duplicates:
        raise ValueError(
            f"PROBLEM_PERIOD_IDS must be unique; repeated: {', '.join(duplicates)}."
        )
    weights = options["PROBLEM_PERIOD_WEIGHTS"]
    if len(weights) > len(period_ids):
        raise ValueError(
            f"PROBLEM_PERIOD_WEIGHTS has {len(weights)} values for "
            f"{len(period_ids)} periods."
        )


def validate_configuration_option_value(name: str, value: Any) -> Any:
    """Validate one supported configuration value."""
    if name not in USER_CONFIG_FIELD_SPECS:
        raise ValueError(f"Unknown configuration option: {name}.")
    return _validate_configuration_option_value(name, value)


def _validate_configuration_option_value(name: str, value: Any) -> Any:
    spec = CONFIG_FIELD_SPECS[name]
    if spec.validator is not None:
        return spec.validator(name, value)
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
                positive=spec.positive,
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
            "HPR_LOAD_PERIOD_VALUES cannot be supplied when HPR_LOAD_MODE "
            "is 'fraction'."
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
            "HPR_LOAD_FRACTION cannot be supplied when HPR_LOAD_MODE "
            "is 'period_values'."
        )
    if mode == "period_values" and "HPR_LOAD_DUTY" in provided_keys:
        raise ValueError(
            "HPR_LOAD_DUTY cannot be supplied when HPR_LOAD_MODE is 'period_values'."
        )
