from __future__ import annotations

from typing import TYPE_CHECKING, List

from .enums import *

if TYPE_CHECKING:
    from .schema import *

# TODO: This file needs a refactor once the purpose of it is well defined.
# At present, the config includes many options corresponding to the Excel
# workbook, but they are not considered during analysis.

"""Global parameters."""
C_to_K: float = 273.15  # degrees
tol: float = 1e-6
T_CRIT: float = 373.9  # C
ACTIVATE_TIMING = True
LOG_TIMING = False


class Configuration:
    """Runtime configuration."""
    
    ### General parameters ###
    TOP_ZONE_NAME: str = "Site"
    TOP_ZONE_IDENTIFIER = ZoneType.S.value
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
    DO_PROCESS_HP_TARGETING: bool = True
    DO_UTILITY_HP_TARGETING: bool = False
    DO_TURBINE_TARGETING: bool = False
    DO_EXERGY_TARGETING: bool = False
    DO_VERTICAL_GCC: bool = False
    DO_ASSITED_HT: bool = False
    DO_TURBINE_WORK: bool = False

    ### Heat pump targeting parameters ###
    MULTI_TEMPERATURE_HP: bool = True
    HP_LOAD_FRACTION: float = 1.0
    DO_HP_SIM: bool = False
    REFRIGERANTS: List[str] = ["propane", "butane", "water"]
    Y_COND_MIN: float = 0.05 # Minimum load on a heat pump condenser, enforced for simulated heat pump
    PRICE_RATIO_ELE_TO_FUEL: float = 1.0
    MAX_HP_MULTISTART: int = 10
    N_COND: int = 2
    N_EVAP: int = 2
    ETA_COMP: float = 0.7
    DTMIN_HP: float = 0.0  

    ### Cost targeting parameters ### 
    UTILITY_PRICE: float = 40
    ANNUAL_OP_TIME: float = 8300
    FIXED_COST: float = 0
    VARIABLE_COST: float = 10000
    COST_EXP: float = 0.7
    DISCOUNT_RATE: float = 0.07
    SERV_LIFE: float = 10

    ### OLD CONFIG -- Review ###

    # T_TURBINE_BOX: float = 450
    # P_TURBINE_BOX: float = 90
    # MIN_EFF: float = 0.1
    # ELECTRICITY_PRICE: float = 100
    # LOAD: float = 1
    # MOTOR_MECH_EFF: float = 1
    # COMBOBOX: str = "Medina-Flores et al. (2010)"
    # ABOVE_PINCH_CHECKBOX: bool = False
    # BELOW_PINCH_CHECKBOX: bool = False
    # CONDESATE_FLASH_CORRECTION: bool = False

    # HHT_OPTION: bool = True
    # VHT_OPTION: bool = False
    # AUTOREORDER: bool = False
    # AUTOREORDER_1: bool = False
    # AUTOREORDER_2: bool = False
    # AUTOREORDER_3: bool = False
    # AUTOREORDER_4: bool = False

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
        options: Options = None,
        top_zone_name: str = "Site",
        top_zone_identifier: str = ZoneType.S.value,
    ):
        """Initialise defaults and optionally apply user-provided options."""
        self.TOP_ZONE_NAME = top_zone_name
        self.TOP_ZONE_IDENTIFIER = top_zone_identifier
        if options:
            self.set_parameters(options)

    def set_parameters(self, options: Options) -> None:
        """Apply checkbox- and turbine-related configuration from :class:`Options`."""
        # Main properties
        main_props = set(options.main)
        self.TIT_BUTTON_SELECTED = "PROP_MOP_0" in main_props
        self.TS_BUTTON_SELECTED = "PROP_MOP_1" in main_props
        self.DO_TURBINE_WORK = "PROP_MOP_2" in main_props
        self.DO_AREA_TARGETING = "PROP_MOP_3" in main_props
        self.ENERGY_RETROFIT_BUTTON = "PROP_MOP_4" in main_props
        self.DO_EXERGY_TARGETING = "PROP_MOP_5" in main_props
        self.PRINT_PTS = "PROP_MOP_6" in main_props
        self.PLOT_GRAPHS = True

        # Graph properties
        if self.PLOT_GRAPHS:
            for checkbox in (
                "CC_CHECKBOX",
                "SCC_CHECKBOX",
                "BCC_CHECKBOX",
                "GCC_CHECKBOX",
                "GCC_N_CHECKBOX",
                "GCCU_CHECKBOX",
                "GCC_Lim_CHECKBOX",
                "TSC_CHECKBOX",
                "ERC_CHECKBOX",
                "NLC_CHECKBOX",
            ):
                setattr(self, checkbox, True)

        # Turbine options
        turbine_options = options.turbine
        self._set_turbine_parameters(turbine_options)

    def _set_turbine_parameters(self, turbine_options: List["TurbineOption"]) -> None:
        """Populate turbine settings when the turbine work toggle is active."""

        option_map = {opt.key: opt.value for opt in turbine_options}

        def get_turbine_value(key: str, default=None):
            value = option_map.get(key, default)
            return default if value is None else value

        if self.DO_TURBINE_WORK:
            self.T_TURBINE_BOX = get_turbine_value("PROP_TOP_0", self.T_TURBINE_BOX)
            self.P_TURBINE_BOX = get_turbine_value("PROP_TOP_1", self.P_TURBINE_BOX)
            self.MIN_EFF = get_turbine_value("PROP_TOP_2", self.MIN_EFF)
            self.ELECTRICITY_PRICE = get_turbine_value(
                "PROP_TOP_3", self.ELECTRICITY_PRICE
            )
            self.LOAD = get_turbine_value("PROP_TOP_4", self.LOAD)
            self.MECH_EFF = get_turbine_value("PROP_TOP_5", self.MECH_EFF)
            self.COMBOBOX = get_turbine_value("PROP_TOP_6", self.COMBOBOX)
            self.ABOVE_PINCH_CHECKBOX = get_turbine_value(
                "PROP_TOP_7", self.ABOVE_PINCH_CHECKBOX
            )
            self.BELOW_PINCH_CHECKBOX = get_turbine_value(
                "PROP_TOP_8", self.BELOW_PINCH_CHECKBOX
            )
            self.CONDESATE_FLASH_CORRECTION = get_turbine_value(
                "PROP_TOP_9", self.CONDESATE_FLASH_CORRECTION
            )
