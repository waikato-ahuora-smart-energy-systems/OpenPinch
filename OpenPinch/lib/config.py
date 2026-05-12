"""Configuration defaults and global numerical constants for OpenPinch.

The :class:`Configuration` object centralizes option flags and numerical
settings used across direct integration, utility targeting, and optional
advanced routines such as heat pump and cost targeting.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List

from .enums import BB_Minimiser, HPRcycle, TurbineModel, ZT

# TODO: This file needs a refactor once the purpose of it is well defined.
# At present, the config includes many options corresponding to the Excel
# workbook, but they are not considered during analysis.

C_to_K: float = 273.15  # degrees
tol: float = 1e-6
T_CRIT: float = 373.9  # C
ACTIVATE_TIMING = False
LOG_TIMING = False

__all__ = [
    "ACTIVATE_TIMING",
    "C_to_K",
    "Configuration",
    "LOG_TIMING",
    "T_CRIT",
    "tol",
]


class Configuration:
    """Runtime configuration defaults used throughout OpenPinch.

    The attributes on this class combine global numerical settings, workbook-
    compatible feature flags, and advanced-analysis parameters such as heat pump
    or costing options. A ``Configuration`` instance is attached to each
    :class:`~OpenPinch.classes.zone.Zone` so workflows can vary behaviour by
    hierarchy level if needed.
    """

    ### General parameters ###
    TOP_ZONE_NAME: str = "Site"
    TOP_ZONE_IDENTIFIER: str = ZT.S.value
    DT_CONT: float = 5
    DT_PHASE_CHANGE: float = 0.01
    HTC: float = 1.0
    T_ENV: float = 15
    DT_ENV_CONT: float = 10
    P_ENV: float = 101
    DECIMAL_PLACES: int = 2

    ### Targeting analysis flags ###
    DO_DIRECT_OPERATION_TARGETING: bool = False
    DO_DIRECT_SITE_TARGETING: bool = True
    DO_INDIRECT_PROCESS_TARGETING: bool = False
    DO_BALANCED_CC: bool = True
    DO_AREA_TARGETING: bool = False
    DO_PROCESS_HP_TARGETING: bool = False
    DO_PROCESS_RFRG_TARGETING: bool = False
    DO_UTILITY_HP_TARGETING: bool = False
    DO_UTILITY_RFRG_TARGETING: bool = False
    DO_TURBINE_TARGETING: bool = False
    DO_EXERGY_TARGETING: bool = False
    DO_VERTICAL_GCC: bool = False
    DO_ASSITED_HT: bool = False
    DO_TURBINE_WORK: bool = False

    ### Heat pump and refrigeration targeting parameters ###
    HPR_TYPE: str = HPRcycle.MultiTempCarnot.value
    HPR_LOAD_VALUE: float | str | dict = 1.0
    HPR_LOAD_VALUE_TYPE: str = "fraction"
    REFRIGERANTS: List[str] = ["water", "ammonia"]
    DO_REFRIGERANT_SORT: bool = True
    PRICE_RATIO_HEAT_TO_ELE: float = 1.0
    PRICE_RATIO_COLD_TO_ELE: float = 1.0
    MAX_HP_MULTISTART: int = 10
    N_COND: int = 3
    N_EVAP: int = 2
    ETA_COMP: float = 0.7
    ETA_EXP: float = 0.7
    ETA_MOTOR: float = 0.95
    ETA_II_HPR_CARNOT: float = 0.5
    ETA_II_HE_CARNOT: float = 0.5
    DT_CONT_HP: float = 0.0
    DT_HPR_IHX: float = 0.0
    DT_HPR_CASCADE_HX: float = 0.0
    BB_MINIMISER: str = BB_Minimiser.CMAES.value
    INITIALISE_SIMULATED_CYCLE: bool = True
    ALLOW_INTEGRATED_EXPANDER: bool = False

    ### Cost targeting parameters ###
    ELE_PRICE: float = 200  # $/MWh
    UTILITY_PRICE: float = 100  # $/MWh
    ANNUAL_OP_TIME: float = 8300
    FIXED_COST: float = 0
    VARIABLE_COST: float = 10000
    COST_EXP: float = 0.6
    DISCOUNT_RATE: float = 0.07
    SERV_LIFE: float = 20  # years

    ### Turbine parameters ###
    TURB_T_IN: float = 450  # degC
    TURB_P_IN: float = 90  # bar
    MIN_EFF: float = 0.1  # minimum isentropic efficiency
    LOAD_FRACTION: float = 1
    ETA_MECH: float = 1
    TURB_MODEL: str = TurbineModel.MEDINA_FLORES.value
    IS_HIGH_P_COND_FLASH: bool = False

    _LEGACY_OPTION_GATEWAYS = {"main", "turbine"}
    _RENAMED_OPTIONS = {
        "HP_CONDESATE": "IS_HIGH_P_COND_FLASH",
        "IS_HP_CONDESATE": "IS_HIGH_P_COND_FLASH",
        "IS_HP_CONDENSATE": "IS_HIGH_P_COND_FLASH",
        "CONDENSATE_FLASH_CORRECTION": "IS_HIGH_P_COND_FLASH",
    }

    def __init__(
        self,
        options: dict | None = None,
        top_zone_name: str = "Site",
        top_zone_identifier: str = ZT.S.value,
    ):
        """Initialise defaults and optionally apply user-provided options."""
        for key in type(self).__annotations__:
            if key.startswith("_"):
                continue
            setattr(self, key, deepcopy(getattr(type(self), key)))

        self.TOP_ZONE_NAME = top_zone_name
        self.TOP_ZONE_IDENTIFIER = top_zone_identifier

        if options is None:
            return

        if not isinstance(options, dict):
            raise TypeError("Configuration options must be provided as a dict.")

        for key, value in self._validate_option_keys(options).items():
            if key == "REFRIGERANTS":
                ref_ls = (
                    value.replace(";", ",").split(",")
                    if isinstance(value, str)
                    else list(value)
                )
                setattr(self, key, ref_ls)
                continue
            setattr(self, key, value)

    @classmethod
    def _known_option_keys(cls) -> set[str]:
        """Return the supported configuration keys accepted by ``options``."""
        return {key for key in cls.__annotations__ if not key.startswith("_")}

    @classmethod
    def _validate_option_keys(cls, options: dict) -> dict:
        """Fail fast on unsupported workbook gateways and unknown option names."""
        legacy_gateways = sorted(
            key
            for key in options
            if key in cls._LEGACY_OPTION_GATEWAYS or str(key).startswith("PROP_TOP_")
        )
        if legacy_gateways:
            raise ValueError(
                "Legacy workbook option gateways are no longer supported: "
                f"{', '.join(legacy_gateways)}. Set canonical turbine fields directly "
                "on zone.config or pass them through Configuration(options=...), e.g. "
                "TURB_T_IN, TURB_P_IN, MIN_EFF, LOAD_FRACTION, ETA_MECH, TURB_MODEL, "
                "and IS_HIGH_P_COND_FLASH."
            )

        renamed_keys = sorted(key for key in options if key in cls._RENAMED_OPTIONS)
        if renamed_keys:
            rename_map = ", ".join(
                f"{key} -> {cls._RENAMED_OPTIONS[key]}" for key in renamed_keys
            )
            raise ValueError(
                "Unsupported configuration option name(s): "
                f"{rename_map}. Use the canonical zone.config field names instead."
            )

        known_keys = cls._known_option_keys()
        unknown_keys = sorted(key for key in options if key not in known_keys)
        if unknown_keys:
            raise ValueError(
                f"Unknown configuration option(s): {', '.join(unknown_keys)}."
            )

        return options
