from __future__ import annotations

from typing import List, TYPE_CHECKING
from .enums import *
if TYPE_CHECKING:
    from .schema import *

# TODO: This file needs a refactor once the purpose of it is well defined.
# At present, the config includes many options corresponding to the Excel 
# workbook, but they are not considered during analysis.

"""Global parameters."""
C_to_K: float = 273.15 # degrees
tol: float = 1e-6
T_CRIT: float = 373.9 # C
ACTIVATE_TIMING = True
LOG_TIMING = False

class Configuration:
    """Runtime configuration flags mirroring options from the legacy Excel workbook."""
    TOP_ZONE_NAME: str = "Site"
    TOP_ZONE_IDENTIFIER = ZoneType.S.value
    # TIT_BUTTON_SELECTED: bool = False
    # TS_BUTTON_SELECTED: bool = False
    TURBINE_WORK_BUTTON: bool = False
    AREA_BUTTON: bool = False
    ENERGY_RETROFIT_BUTTON: bool = False
    EXERGY_BUTTON: bool = False
    # PRINT_PTS: bool = False
    PLOT_GRAPHS: bool = True
    DECIMAL_PLACES: int = 2

    CC_CHECKBOX: bool = True
    SCC_CHECKBOX: bool = True
    BCC_CHECKBOX: bool = False
    GCC_CHECKBOX: bool = True
    GCC_NP_CHECKBOX: bool = True
    GCCU_CHECKBOX: bool = False
    LGCC_CHECKBOX: bool = False
    TSC_CHECKBOX: bool = False
    ERC_CHECKBOX: bool = False
    NLC_CHECKBOX: bool = False
    MAX_GRAPHS: int = 128

    T_TURBINE_BOX: float = 450
    P_TURBINE_BOX: float = 90
    MIN_EFF: float = 0.1
    ELECTRICITY_PRICE: float = 100
    LOAD: float = 1
    MECH_EFF: float = 1
    COMBOBOX: str = "Medina-Flores et al. (2010)"
    ABOVE_PINCH_CHECKBOX: bool = False
    BELOW_PINCH_CHECKBOX: bool = False
    CONDESATE_FLASH_CORRECTION: bool = False

    # CF_SELECTED: bool = True
    # PF_SELECTED: bool = False
    # SHELL_AND_TUBE: bool = False

    HHT_OPTION: bool = True
    VHT_OPTION: bool = False
    AUTOREORDER: bool = False
    AUTOREORDER_1: bool = False
    AUTOREORDER_2: bool = False
    AUTOREORDER_3: bool = False
    AUTOREORDER_4: bool = False
    GCC_REG_FULL_POCKET: bool = True
    GCC_VERT_CUT_KINK_OPTION: bool = False
    SET_MIN_DH_THRES: bool = False
    SET_MIN_TH_AREA: bool = False
    AUTOMATED_RETROFIT_TARGETING_BUTTON: bool = False
    THRESHOLD: float = 0
    AREA_THRESHOLD: float = 0

    Q_MIN: float = 400
    NUM_HX: float = 6
    AREA_RATIO: float = 0
    PRICE_UTILITIES: float = 220
    MAX_RBBRIDGE: float = 100000
    RECORD: bool = True
    PARETO: bool = False
    QUANTIFY_AREA: bool = False
    ACCELERATION: bool = True
    HEURISTICS: bool = True

    FC: float = 0
    VC: float = 10000
    EXP: float = 0.7
    DISCOUNT_RATE: float = 0.07
    SERV_LIFE: float = 10
    ANNUAL_OP_TIME: float = 8300

    DTCONT: float = 5
    DTGLIDE: float = 0.01
    HTC: float = 1.0
    UTILITY_PRICE: float = 40
    TEMP_REF: float = 15
    PRESSURE_REF: float = 101
    AHT_BUTTON_SELECTED: bool = False
    OVERRIDEDT_BUTTON_SELECTED: bool = False

    def __init__(self, options: Options = None, top_zone_name: str = "Site", top_zone_identifier: str = ZoneType.S.value):
        """Initialise defaults and optionally apply user-provided options."""
        self.TOP_ZONE_NAME = top_zone_name
        self.TOP_ZONE_IDENTIFIER = top_zone_identifier
        if options:
            self.set_parameters(options)

    def set_parameters(self, options: Options) -> None:
        """Apply checkbox- and turbine-related configuration from :class:`Options`."""
        # Main properties
        main_props = options.main
        self.TIT_BUTTON_SELECTED = "PROP_MOP_0" in main_props
        self.TS_BUTTON_SELECTED = "PROP_MOP_1" in main_props
        self.TURBINE_WORK_BUTTON = "PROP_MOP_2" in main_props
        self.AREA_BUTTON = "PROP_MOP_3" in main_props
        self.ENERGY_RETROFIT_BUTTON = "PROP_MOP_4" in main_props
        self.EXERGY_BUTTON = "PROP_MOP_5" in main_props
        self.PRINT_PTS = "PROP_MOP_6" in main_props
        self.PLOT_GRAPHS = True

        # Graph properties
        if self.PLOT_GRAPHS:
            self.CC_CHECKBOX = True
            self.SCC_CHECKBOX = True
            self.BCC_CHECKBOX = True
            self.GCC_CHECKBOX = True
            self.GCC_NP_CHECKBOX = True
            self.GCCU_CHECKBOX = True
            self.LGCC_CHECKBOX = True
            self.TSC_CHECKBOX = True
            self.ERC_CHECKBOX = True
            self.NLC_CHECKBOX = True

        # Turbine options
        turbine_options = options.turbine
        self._set_turbine_parameters(turbine_options)

    def _set_turbine_parameters(self, turbine_options: List["TurbineOption"]) -> None:
        """Populate turbine settings when the turbine work toggle is active."""
        def get_turbine_value(key: str, default=None):
            for option in turbine_options:
                if option.key == key:  # Access the key attribute directly
                    return option.value if option.value is not None else default
            return default

        if self.TURBINE_WORK_BUTTON:
            self.T_TURBINE_BOX = get_turbine_value("PROP_TOP_0", self.T_TURBINE_BOX)
            self.P_TURBINE_BOX = get_turbine_value("PROP_TOP_1", self.P_TURBINE_BOX)
            self.MIN_EFF = get_turbine_value("PROP_TOP_2", self.MIN_EFF)
            self.ELECTRICITY_PRICE = get_turbine_value("PROP_TOP_3", self.ELECTRICITY_PRICE)
            self.LOAD = get_turbine_value("PROP_TOP_4", self.LOAD)
            self.MECH_EFF = get_turbine_value("PROP_TOP_5", self.MECH_EFF)
            self.COMBOBOX = get_turbine_value("PROP_TOP_6", self.COMBOBOX)
            self.ABOVE_PINCH_CHECKBOX = get_turbine_value("PROP_TOP_7", self.ABOVE_PINCH_CHECKBOX)
            self.BELOW_PINCH_CHECKBOX = get_turbine_value("PROP_TOP_8", self.BELOW_PINCH_CHECKBOX)
            self.CONDESATE_FLASH_CORRECTION = get_turbine_value("PROP_TOP_9", self.CONDESATE_FLASH_CORRECTION)
