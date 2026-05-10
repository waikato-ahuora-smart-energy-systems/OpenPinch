"""Configuration defaults and global numerical constants for OpenPinch.

The :class:`Configuration` object centralizes option flags and numerical
settings used across direct integration, utility targeting, and optional
advanced routines such as heat pump and cost targeting.
"""

from __future__ import annotations

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
    TURB_T_IN: float = 450 # degC
    TURB_P_IN: float = 90 # bar
    MIN_EFF: float = 0.1 # minimum isentropic efficiency
    LOAD_FRACTION: float = 1
    ETA_MECH: float = 1
    TURB_MODEL: str = TurbineModel.MEDINA_FLORES.value
    HP_CONDESATE: bool = False

    def __init__(
        self,
        options: dict = None,
        top_zone_name: str = "Site",
        top_zone_identifier: str = ZT.S.value,
    ):
        """Initialise defaults and optionally apply user-provided options."""
        self.TOP_ZONE_NAME = top_zone_name
        self.TOP_ZONE_IDENTIFIER = top_zone_identifier
        if isinstance(options, dict):
            for key in options.keys():
                if key != "REFRIGERANTS":
                    setattr(self, key, options[key])
                else:
                    ref_ls = options[key].replace(";", ",").split(",")
                    setattr(self, key, ref_ls)
