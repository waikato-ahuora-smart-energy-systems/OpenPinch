"""Configuration defaults and global numerical constants for OpenPinch.

The :class:`Configuration` object centralizes option flags and numerical
settings used across direct integration, utility targeting, and optional
advanced routines such as heat pump and cost targeting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from .enums import *

if TYPE_CHECKING:
    from .schema import *

# TODO: This file needs a refactor once the purpose of it is well defined.
# At present, the config includes many options corresponding to the Excel
# workbook, but they are not considered during analysis.

C_to_K: float = 273.15  # degrees
tol: float = 1e-6
T_CRIT: float = 373.9  # C
ACTIVATE_TIMING = False
LOG_TIMING = False


class Configuration:
    """Runtime configuration defaults used throughout OpenPinch.

    The attributes on this class combine global numerical settings, workbook-
    compatible feature flags, and advanced-analysis parameters such as heat-pump
    or costing options. A ``Configuration`` instance is attached to each
    :class:`~OpenPinch.classes.zone.Zone` so workflows can vary behaviour by
    hierarchy level if needed.
    """

    ### General parameters ###
    TOP_ZONE_NAME: str = "Site"
    TOP_ZONE_IDENTIFIER: str = ZoneType.S.value
    DT_CONT: float = 5
    DT_PHASE_CHANGE: float = 0.1
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
    DO_PROCESS_RFRG_TARGETING: bool = False  ### add to the template
    DO_UTILITY_HP_TARGETING: bool = False
    DO_UTILITY_RFRG_TARGETING: bool = False  ### add to the template
    DO_TURBINE_TARGETING: bool = False
    DO_EXERGY_TARGETING: bool = False
    DO_VERTICAL_GCC: bool = False
    DO_ASSITED_HT: bool = False
    DO_TURBINE_WORK: bool = False

    ### Heat pump and refrigeration targeting parameters ###
    HPR_TYPE: str = HPRcycle.MultiTempCarnot.value  ### add to the template (modified)
    HPR_LOAD_VALUE: float | str | dict = 1.0  ### add to the template (modified)
    HPR_LOAD_VALUE_TYPE: str = "fraction"  ### add to the template (modified)
    REFRIGERANTS: List[str] = ["water", "ammonia"]
    DO_REFRIGERANT_SORT: bool = True
    PRICE_RATIO_HEAT_TO_ELE: float = 1.0  ### add to the template (modified)
    PRICE_RATIO_COLD_TO_ELE: float = 1.0  ### add to the template (modified)
    MAX_HP_MULTISTART: int = 10
    N_COND: int = 3
    N_EVAP: int = 2
    ETA_COMP: float = 0.7
    ETA_EXP: float = 0.7
    ETA_MOTOR: float = 0.95  ### add to the template (modified)
    ETA_II_HPR_CARNOT: float = 0.5  ### add to the template (modified)
    ETA_II_HE_CARNOT: float = 0.5  ### add to the template (modified)
    DT_CONT_HP: float = 0.0  ### add to the template (modified)
    DT_HPR_IHX: float = 0.0  ### add to the template (modified)
    DT_HPR_CASCADE_HX: float = 0.0  ### add to the template (modified)
    BB_MINIMISER: str = BB_Minimiser.CMAES.value  ### add to the template
    INITIALISE_SIMULATED_CYCLE: bool = True  ### add to the template
    ALLOW_INTEGRATED_EXPANDER: bool = False  ### add to the template

    ### Cost targeting parameters ###
    ELE_PRICE: float = 200  # $/MWh   ### add to the template
    UTILITY_PRICE: float = 100  # $/MWh
    ANNUAL_OP_TIME: float = 8300
    FIXED_COST: float = 0
    VARIABLE_COST: float = 10000
    COST_EXP: float = 0.6
    DISCOUNT_RATE: float = 0.07
    SERV_LIFE: float = 20  # years

    ### OLD CONFIG -- TODO: Review ###

    T_TURBINE_BOX: float = 450
    P_TURBINE_BOX: float = 90
    MIN_EFF: float = 0.1
    ELECTRICITY_PRICE: float = 100
    LOAD: float = 1
    MECH_EFF: float = 1
    COMBOBOX: str = TurbineModel.MEDINA_FLORES.value
    ABOVE_PINCH_CHECKBOX: bool = False
    BELOW_PINCH_CHECKBOX: bool = False
    CONDESATE_FLASH_CORRECTION: bool = False

    # HHT_OPTION: bool = True
    # VHT_OPTION: bool = False

    # GCC_REG_FULL_POCKET: bool = True
    # GCC_VERT_CUT_KINK_OPTION: bool = False
    # SET_MIN_DH_THRES: bool = False
    # SET_MIN_TH_AREA: bool = False
    # AUTOMATED_RETROFIT_TARGETING_BUTTON: bool = False
    # THRESHOLD: float = 0
    # AREA_THRESHOLD: float = 0

    # Q_MIN: float = 400
    # NUM_HX: float = 6
    # AREA_RATIO: float = 0
    # PRICE_UTILITIES: float = 220
    # MAX_RBBRIDGE: float = 100000
    # RECORD: bool = True
    # PARETO: bool = False
    # QUANTIFY_AREA: bool = False
    # ACCELERATION: bool = True
    # HEURISTICS: bool = True

    def __init__(
        self,
        options: dict = None,
        top_zone_name: str = "Site",
        top_zone_identifier: str = ZoneType.S.value,
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

    # def set_parameters(self, options: Options) -> None:
    #     """Apply checkbox- and turbine-related configuration from :class:`Options`."""
    #     # Main properties
    #     main_props = set(options.main)
    #     self.TIT_BUTTON_SELECTED = "PROP_MOP_0" in main_props
    #     self.TS_BUTTON_SELECTED = "PROP_MOP_1" in main_props
    #     self.DO_TURBINE_WORK = "PROP_MOP_2" in main_props
    #     self.DO_AREA_TARGETING = "PROP_MOP_3" in main_props
    #     self.ENERGY_RETROFIT_BUTTON = "PROP_MOP_4" in main_props
    #     self.DO_EXERGY_TARGETING = "PROP_MOP_5" in main_props
    #     self.PRINT_PTS = "PROP_MOP_6" in main_props
    #     self.PLOT_GRAPHS = True

    #     # Graph properties
    #     if self.PLOT_GRAPHS:
    #         for checkbox in (
    #             "CC_CHECKBOX",
    #             "SCC_CHECKBOX",
    #             "BCC_CHECKBOX",
    #             "GCC_CHECKBOX",
    #             "GCC_N_CHECKBOX",
    #             "GCCU_CHECKBOX",
    #             "GCC_Lim_CHECKBOX",
    #             "TSC_CHECKBOX",
    #             "ERC_CHECKBOX",
    #             "NLC_CHECKBOX",
    #         ):
    #             setattr(self, checkbox, True)

    #     # Turbine options
    #     turbine_options = options.turbine
    #     self._set_turbine_parameters(turbine_options)

    # def _set_turbine_parameters(self, turbine_options: List["TurbineOption"]) -> None:
    #     """Populate turbine settings when the turbine work toggle is active."""

    #     option_map = {opt.key: opt.value for opt in turbine_options}

    #     def get_turbine_value(key: str, default=None):
    #         value = option_map.get(key, default)
    #         return default if value is None else value

    #     if self.DO_TURBINE_WORK:
    #         self.T_TURBINE_BOX = get_turbine_value("PROP_TOP_0", self.T_TURBINE_BOX)
    #         self.P_TURBINE_BOX = get_turbine_value("PROP_TOP_1", self.P_TURBINE_BOX)
    #         self.MIN_EFF = get_turbine_value("PROP_TOP_2", self.MIN_EFF)
    #         self.ELECTRICITY_PRICE = get_turbine_value(
    #             "PROP_TOP_3", self.ELECTRICITY_PRICE
    #         )
    #         self.LOAD = get_turbine_value("PROP_TOP_4", self.LOAD)
    #         self.MECH_EFF = get_turbine_value("PROP_TOP_5", self.MECH_EFF)
    #         self.COMBOBOX = get_turbine_value("PROP_TOP_6", self.COMBOBOX)
    #         self.ABOVE_PINCH_CHECKBOX = get_turbine_value(
    #             "PROP_TOP_7", self.ABOVE_PINCH_CHECKBOX
    #         )
    #         self.BELOW_PINCH_CHECKBOX = get_turbine_value(
    #             "PROP_TOP_8", self.BELOW_PINCH_CHECKBOX
    #         )
    #         self.CONDESATE_FLASH_CORRECTION = get_turbine_value(
    #             "PROP_TOP_9", self.CONDESATE_FLASH_CORRECTION
    #         )
