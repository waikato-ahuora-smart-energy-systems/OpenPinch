from enum import Enum


class ZoneType(Enum):
    """Types of zones used to divide the problem."""

    R = "Region"
    C = "Community"
    S = "Site"
    P = "Process Zone"
    U = "Utility Zone"
    O = "Unit Operation"

    def __str__(self):
        return self.value


class TargetType(Enum):
    """Different target calculation categories."""

    TL = "Thermodynamic Limit Target"
    DI = "Direct Integration"
    TZ = "Total Process Target"
    TS = "Total Site Target"  # Also indirect integration
    RT = "Regional Target"  # Currently the same as TS
    ET = "Energy Transfer Analysis"

    def __str__(self):
        return self.value


class HeatExchangerTypes(Enum):
    """Heat exchanger flow arrangements"""

    CF = "Counter Flow"
    PF = "Parallel Flow"
    CrFUU = "Crossflow - Both Unmixed"
    CrFMM = "Crossflow - Both Mixed"
    CrFMUmax = "Crossflow - Cmax Unmixed"
    CrFMUmin = "Crossflow - Cmin Unmixed"
    ShellTube = "1-n Shell and Tube"
    CondEvap = "Condensing or Evaporating"


class HeatFlowUnits(Enum):
    """Heat flow units"""

    W = "W"
    kW = "kW"
    MW = "MW"
    GW = "GW"


class StreamType(Enum):
    """Steam type"""

    Hot = "Hot"
    Cold = "Cold"
    Both = "Both"
    Unassigned = ""


class StreamID(Enum):
    """Stream identity"""

    Process = "Process"
    Utility = "Utility"
    Unassigned = "Unassigned"


class StreamLoc(Enum):
    """Stream set identity"""

    HotS = "Hot Streams"
    ColdS = "Cold Streams"
    HotU = "Hot Utility"
    ColdU = "Cold Utility"
    Unassigned = "Unassigned"


class ProblemTableLabel(Enum):
    """Problem table column header labels"""

    T = "T"
    DELTA_T = "\N{GREEK CAPITAL LETTER DELTA}T"
    CP_HOT = "mcp_hot_tot"
    DELTA_H_HOT = "\N{GREEK CAPITAL LETTER DELTA}H_hot"
    H_HOT = "H_hot"
    CP_COLD = "mcp_cold_tot"
    DELTA_H_COLD = "\N{GREEK CAPITAL LETTER DELTA}H_cold"
    H_COLD = "H_cold"
    CP_NET = "CP_NET"
    DELTA_H_NET = "\N{GREEK CAPITAL LETTER DELTA}H_net"
    H_NET = "H_net"

    H_NET_NP = "H_net_np"
    H_NET_V = "H_net_vert"
    H_NET_PK = "H_net_pockets"
    H_NET_AI = "H_net_assisted"
    H_NET_A = "H_net_actual"
    H_UT_NET = "H_net_ut"
    H_HOT_NET = "H_hot_net"
    H_COLD_NET = "H_cold_net"

    H_HOT_UT = "H_hot_utility"
    H_COLD_UT = "H_cold_utility"
    H_HOT_BAL = "H_hot_balanced"
    H_COLD_BAL = "H_cold_balanced"

    RCP_HOT = "rCP_hot"
    RCP_COLD = "rCP_cold"
    RCP_HOT_NET = "rcp_hot_net"
    RCP_COLD_NET = "rcp_cold_net"
    RCP_UT_NET = "rcp_ut_net"
    RCP_HOT_UT = "rcp_hot_ut"
    RCP_COLD_UT = "rcp_cold_ut"
    RCP_HOT_BAL = "rCP_hot_balanced"
    RCP_COLD_BAL = "rCP_cold_balanced"
    R_HOT_BAL = "HTC_hot_balanced"
    R_COLD_BAL = "HTC_cold_balanced"


PT = ProblemTableLabel


class StreamDataLabel(Enum):
    """Stream data column header labels"""

    TS = "T_supply"
    TT = "T_target"
    TYPE = "stream_type"
    CP = "heat_capacity_flowrate"
    H = "heat_flow"
    DT_CONT = "\N{GREEK CAPITAL LETTER DELTA}T_cont"
    HTC = "heat_transfer_coefficient"


SD = StreamDataLabel


class ArrowHead(Enum):
    """Position of arrow head"""

    START = "Start"
    END = "End"
    NO_ARROW = "None"


class LineColour(Enum):
    """Line colour selection"""

    HotS = 0
    ColdS = 1
    HotU = 2
    ColdU = 3
    Black = 5
    Other = 4


class GraphType(Enum):
    CC = "Composite Curves"
    SCC = "Shifted Composite Curves"
    BCC = "Balanced Composite Curves"
    GCC = "Grand Composite Curve"
    GCC_R = "Grand Composite Curve (Real)"
    GCC_X = "Exergetic Grand Composite Curve"
    # GCC_N = "Grand Composite Curve (No Pockets)"
    # GCC_V = "Vertical Grand Composite Curve"
    # GCC_A = "Actual Grand Composite Curve"    
    # GCC_U = "Utility Grand Composite Curve"
    # GCC_U_real = "Utility Grand Composite Curve (Real)"
    # GCC_Lim = "Thermodynamic Limiting GCC"
    NLC = "Net Load Curves"

    TSP = "Total Site Profiles"
    TSU = "Total Site Utility"
    # TSU_real = "Total Site Utility"
    SUGCC = "Site Utility Grand Composite Curve"


ResultsType = GT = GraphType


class LegendSeries(Enum):
    """Legend labels for multi-series graphs."""

    GCC = "GCC"
    GCC_N = "GCC (No Pockets)"
    GCC_V = "Vertical GCC"
    GCC_A = "Assisted GCC"
    GCC_U = "Utility GCC"


class SummaryRowType(Enum):
    CONTENT = "content"
    FOOTER = "footer"


class TurbineModel(Enum):
    MEDINA_FLORES = "Medina-Flores et al. (2010)"
    SUN_SMITH = "Sun & Smith (2015)"
    VARBANOV = "Varbanov et al. (2004)"
    ISENTROPIC = "Fixed Isentropic Turbine"


class MainOptionsPropKeys(Enum):
    Totally_Integrated_Site = "PROP_MOP_0"
    Total_Site = "PROP_MOP_1"
    Turbine_Work = "PROP_MOP_2"
    Target_Area = "PROP_MOP_3"
    Energy_Retrofit = "PROP_MOP_4"
    Thermal_Exergy = "PROP_MOP_5"
    Problem_Tables = "PROP_MOP_6"


class TurbineOptionsPropKeys(Enum):
    TURBINEFORM_T_TURBINE_BOX = "PROP_TOP_0"
    TURBINEFORM_P_TURBINE_BOX = "PROP_TOP_1"
    TURBINEFORM_MIN_EFF = "PROP_TOP_2"
    TURBINEFORM_ELECTRICITY_PRICE = "PROP_TOP_3"
    TURBINEFORM_LOAD = "PROP_TOP_4"
    TURBINEFORM_MECH_EFF = "PROP_TOP_5"
    TURBINEFORM_COMBOBOX = "PROP_TOP_6"
    TURBINEFORM_ABOVE_PINCH_CHECKBOX = "PROP_TOP_7"
    TURBINEFORM_BELOW_PINCH_CHECKBOX = "PROP_TOP_8"
    TURBINEFORM_CONDESATE_FLASH_CORRECTION = "PROP_TOP_9"
