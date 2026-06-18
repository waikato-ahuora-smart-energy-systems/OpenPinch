"""Shared declarative metadata for :mod:`OpenPinch` configuration fields."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List

from .enums import ZT, BB_Minimiser, HPRcycle, TurbineModel


@dataclass(frozen=True)
class ConfigurationFieldSpec:
    """Describe one editable configuration field and its workspace metadata."""

    annotation: Any
    default: Any
    group: str
    enum_cls: type[Enum] | None = None
    numeric_min: float | None = None
    runtime_status: str = "supported"


@dataclass(frozen=True)
class ConfigurationOptionStatus:
    """Classify one incoming configuration option name."""

    name: str
    runtime_status: str


# fmt: off
CONFIG_FIELD_SPECS: dict[str, ConfigurationFieldSpec] = {
    "TOP_ZONE_NAME": ConfigurationFieldSpec(str, "Site", "general"),
    "TOP_ZONE_IDENTIFIER": ConfigurationFieldSpec(str, ZT.S.value, "general", enum_cls=ZT),
    "DT_CONT": ConfigurationFieldSpec(float, 5, "general", numeric_min=0.0),
    "DT_PHASE_CHANGE": ConfigurationFieldSpec(float, 0.01, "general", numeric_min=0.0),
    "DT_ASSISTED_HT": ConfigurationFieldSpec(float, 10.0, "general", numeric_min=0.0),
    "HTC": ConfigurationFieldSpec(float, 1.0, "general", numeric_min=0.0),
    "T_ENV": ConfigurationFieldSpec(float, 15, "general"),
    "DT_ENV_CONT": ConfigurationFieldSpec(float, 10, "general", numeric_min=0.0),
    "P_ENV": ConfigurationFieldSpec(float, 101, "general"),
    "DECIMAL_PLACES": ConfigurationFieldSpec(int, 2, "general", numeric_min=0.0),
    "WEIGHTS": ConfigurationFieldSpec(List[float], [1], "general"),
    "STATE_IDS": ConfigurationFieldSpec(List[str], ["0"], "general"),
    "INPUT_UNITS": ConfigurationFieldSpec(dict[str, str], {}, "general"),
    "OUTPUT_UNITS": ConfigurationFieldSpec(dict[str, str], {}, "general"),
    "DO_DIRECT_OPERATION_TARGETING": ConfigurationFieldSpec(bool, False, "targeting"),
    "DO_DIRECT_SITE_TARGETING": ConfigurationFieldSpec(bool, True, "targeting"),
    "DO_INDIRECT_PROCESS_TARGETING": ConfigurationFieldSpec(bool, False, "targeting"),
    "DO_BALANCED_CC": ConfigurationFieldSpec(bool, True, "targeting"),
    "DO_AREA_TARGETING": ConfigurationFieldSpec(bool, False, "targeting"),
    "DO_PROCESS_HP_TARGETING": ConfigurationFieldSpec(bool, False, "targeting"),
    "DO_PROCESS_RFRG_TARGETING": ConfigurationFieldSpec(bool, False, "targeting"),
    "DO_UTILITY_HP_TARGETING": ConfigurationFieldSpec(bool, False, "targeting"),
    "DO_UTILITY_RFRG_TARGETING": ConfigurationFieldSpec(bool, False, "targeting"),
    "DO_TURBINE_TARGETING": ConfigurationFieldSpec(bool, False, "targeting"),
    "DO_EXERGY_TARGETING": ConfigurationFieldSpec(bool, False, "targeting", runtime_status="experimental"),
    "DO_VERTICAL_GCC": ConfigurationFieldSpec(bool, False, "targeting"),
    "DO_ASSITED_HT": ConfigurationFieldSpec(bool, False, "targeting"),
    "DO_TURBINE_WORK": ConfigurationFieldSpec(bool, False, "targeting"),
    "HPR_TYPE": ConfigurationFieldSpec(str, HPRcycle.CascadeCarnot.value, "hpr", enum_cls=HPRcycle),
    "HPR_LOAD_VALUE": ConfigurationFieldSpec(float | str | dict, 1.0, "hpr"),
    "HPR_LOAD_VALUE_TYPE": ConfigurationFieldSpec(str, "fraction", "hpr"),
    "REFRIGERANTS": ConfigurationFieldSpec(List[str], ["water", "ammonia"], "hpr"),
    "MVR_FLUIDS": ConfigurationFieldSpec(List[str], ["Water"], "hpr"),
    "DO_REFRIGERANT_SORT": ConfigurationFieldSpec(bool, True, "hpr"),
    "PRICE_RATIO_HEAT_TO_ELE": ConfigurationFieldSpec(float, 1.0, "hpr", numeric_min=0.0),
    "PRICE_RATIO_COLD_TO_ELE": ConfigurationFieldSpec(float, 1.0, "hpr", numeric_min=0.0),
    "MAX_HP_MULTISTART": ConfigurationFieldSpec(int, 10, "hpr", numeric_min=0.0),
    "N_COND": ConfigurationFieldSpec(int, 3, "hpr", numeric_min=0.0),
    "N_EVAP": ConfigurationFieldSpec(int, 2, "hpr", numeric_min=0.0),
    "N_MVR": ConfigurationFieldSpec(int, 1, "hpr", numeric_min=1.0),
    "ETA_COMP": ConfigurationFieldSpec(float, 0.7, "hpr", numeric_min=0.0),
    "ETA_MVR_COMP": ConfigurationFieldSpec(float, 0.7, "hpr", numeric_min=0.0),
    "ETA_EXP": ConfigurationFieldSpec(float, 0.7, "hpr", numeric_min=0.0),
    "ETA_MOTOR": ConfigurationFieldSpec(float, 0.95, "hpr", numeric_min=0.0),
    "ETA_II_HPR_CARNOT": ConfigurationFieldSpec(float, 0.5, "hpr", numeric_min=0.0),
    "ETA_II_HE_CARNOT": ConfigurationFieldSpec(float, 0.5, "hpr", numeric_min=0.0),
    "DT_CONT_HP": ConfigurationFieldSpec(float, 0.0, "hpr", numeric_min=0.0),
    "DT_HPR_IHX": ConfigurationFieldSpec(float, 0.0, "hpr", numeric_min=0.0),
    "DT_HPR_CASCADE_HX": ConfigurationFieldSpec(float, 0.0, "hpr", numeric_min=0.0),
    "ETA_PENALTY": ConfigurationFieldSpec(float, 0.001, "hpr", numeric_min=0.0),
    "RHO_PENALTY": ConfigurationFieldSpec(float, 10.0, "hpr", numeric_min=0.0),
    "BB_MINIMISER": ConfigurationFieldSpec(str, BB_Minimiser.CMAES.value, "hpr", enum_cls=BB_Minimiser),
    "INITIALISE_SIMULATED_CYCLE": ConfigurationFieldSpec(bool, True, "hpr"),
    "ALLOW_INTEGRATED_EXPANDER": ConfigurationFieldSpec(bool, False, "hpr"),
    "HPR_COMP_FIXED_COST": ConfigurationFieldSpec(float, 0, "hpr", numeric_min=0.0),
    "HPR_COMP_VARIABLE_COST": ConfigurationFieldSpec(float, 10000, "hpr", numeric_min=0.0),
    "HPR_COMP_COST_EXP": ConfigurationFieldSpec(float, 1.0, "hpr", numeric_min=0.0),
    "HPR_HX_FIXED_COST": ConfigurationFieldSpec(float, 0, "hpr", numeric_min=0.0),
    "HPR_HX_VARIABLE_COST": ConfigurationFieldSpec(float, 10000, "hpr", numeric_min=0.0),
    "HPR_HX_COST_EXP": ConfigurationFieldSpec(float, 1.0, "hpr", numeric_min=0.0),
    "ELE_PRICE": ConfigurationFieldSpec(float, 200, "costing", numeric_min=0.0),
    "UTILITY_PRICE": ConfigurationFieldSpec(float, 100, "costing", numeric_min=0.0),
    "ANNUAL_OP_TIME": ConfigurationFieldSpec(float, 8300, "costing", numeric_min=0.0),
    "FIXED_COST": ConfigurationFieldSpec(float, 0, "costing", numeric_min=0.0),
    "VARIABLE_COST": ConfigurationFieldSpec(float, 10000, "costing", numeric_min=0.0),
    "COST_EXP": ConfigurationFieldSpec(float, 0.6, "costing", numeric_min=0.0),
    "DISCOUNT_RATE": ConfigurationFieldSpec(float, 0.07, "costing", numeric_min=0.0),
    "SERV_LIFE": ConfigurationFieldSpec(float, 20, "costing", numeric_min=0.0),
    "TURB_T_IN": ConfigurationFieldSpec(float, 450, "turbine"),
    "TURB_P_IN": ConfigurationFieldSpec(float, 90, "turbine", numeric_min=0.0),
    "MIN_EFF": ConfigurationFieldSpec(float, 0.1, "turbine", numeric_min=0.0),
    "LOAD_FRACTION": ConfigurationFieldSpec(float, 1, "turbine", numeric_min=0.0),
    "ETA_MECH": ConfigurationFieldSpec(float, 1, "turbine", numeric_min=0.0),
    "TURB_MODEL": ConfigurationFieldSpec(str, TurbineModel.MEDINA_FLORES.value, "turbine", enum_cls=TurbineModel),
    "IS_HIGH_P_COND_FLASH": ConfigurationFieldSpec(bool, False, "turbine"),
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
        return "general"
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
    return "stable" if group in {"general", "targeting"} else "advanced"
