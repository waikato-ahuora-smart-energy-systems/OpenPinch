"""Enumerations and lightweight typed contracts used across OpenPinch.

These enums standardize zone types, stream classifications, Problem Table
column names, graph labels, and options keys used by configuration and schemas.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..domain.stream_collection import StreamCollection
    from .config import Configuration


class CogenerationTarget(Protocol):
    """Compatible target surface required by cogeneration analysis helpers."""

    config: Configuration
    hot_utilities: StreamCollection
    work_target: float | None
    turbine_efficiency_target: float | None
    period_ids: dict[str, int] | None


class ZoneType(Enum):
    """Types of zones used to divide the problem."""

    R = "Region"
    C = "Community"
    S = "Site"
    P = "Process Zone"
    U = "Utility Zone"
    O = "Unit Operation"  # noqa: E741

    def __str__(self):
        return self.value


ZT = ZoneType


class TargetType(Enum):
    """Different target calculation categories."""

    TL = "Thermodynamic Limit Target"
    DI = "Direct Integration"
    TZ = "Total Process Target"
    TS = "Total Site Target"  # Also indirect integration
    RT = "Regional Target"  # Currently the same as TS
    ET = "Energy Transfer Analysis"
    DHP = "Direct Heat Pump"
    IHP = "Indirect Heat Pump"
    DR = "Direct Refrigeration"
    IR = "Indirect Refrigeration"

    def __str__(self):
        return self.value


TT = TargetType


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


class HeatExchangerKind(str, Enum):
    """Supported heat-transfer link families in a heat exchanger network design."""

    RECOVERY = "recovery"
    HOT_UTILITY = "hot_utility"
    COLD_UTILITY = "cold_utility"


class HeatExchangerStreamRole(str, Enum):
    """Identity class for the source and sink streams on an exchanger link."""

    PROCESS = "process"
    UTILITY = "utility"


class HeatExchangerNetworkDesignMethod(str, Enum):
    """User-selectable heat exchanger network design service methods."""

    OpenHENS = "open_hens_method"
    PinchDesign = "pinch_design_method"
    ThermalDerivative = "thermal_derivative_method"
    NetworkEvolution = "network_evolution_method"

    def __str__(self):
        return self.value


HENDesignMethod = HeatExchangerNetworkDesignMethod


class HeatPump(Enum):
    """Heat pump components"""

    Cond = "Condenser"
    Evap = "Evaporator"
    Comp = "Compressor"
    Expd = "Expansion"
    IHX = "Internal Heat Exchanger"


class HeatPumpAndRefrigerationCycle(str, Enum):
    """Supported heat pump targeting model families."""

    CascadeCarnot = "Cascade Carnot cycles"
    ParallelCarnot = "Parallel Carnot cycles"
    Brayton = "Brayton cycle"
    CascadeVapourComp = "Cascade vapour compression cycles"
    ParallelVapourComp = "Parallel vapour compression cycles"
    VapourCompMVR = "Vapour compression with MVR cascade"


HPRcycle = HeatPumpAndRefrigerationCycle


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
    Neutral = "Neutral"
    Unassigned = ""


ST = StreamType


class FluidPhase(str, Enum):
    """Supported stream fluid-phase flags."""

    sol = "solid"
    sle = "solid-liquid equilibrium"
    liq = "liquid"
    vle = "vapour-liquid equilibrium"
    vapour = "vapour"
    sve = "solid-vapour equilibrium"
    gas = "gas"

    @classmethod
    def from_code_or_description(cls, value: str | "FluidPhase") -> "FluidPhase":
        """Resolve a phase from its short code or descriptive label."""
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        for phase in cls:
            aliases = {phase.name, phase.value.lower()}
            if phase is cls.vapour:
                aliases.add("vapor")
            if text in aliases:
                return phase
        raise ValueError(f"Unknown fluid phase: {value!r}.")


class StreamID(Enum):
    """Stream identity"""

    Process = "Process"
    Utility = "Utility"
    Unassigned = "Unassigned"


SID = StreamID


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
    CP_HOT = "\N{GREEK CAPITAL LETTER SIGMA}CP(hot)"
    DELTA_H_HOT = "\N{GREEK CAPITAL LETTER DELTA}H(hot)"
    H_HOT = "H(hot)"
    CP_COLD = "\N{GREEK CAPITAL LETTER SIGMA}CP(cold)"
    DELTA_H_COLD = "\N{GREEK CAPITAL LETTER DELTA}H(cold)"
    H_COLD = "H(cold)"
    CP_NET = "CP(net)"
    DELTA_H_NET = "\N{GREEK CAPITAL LETTER DELTA}H(net)"
    H_NET = "H(net)"

    H_NET_NP = "H(net)-no pockets"
    H_NET_V = "H(net)-vertical heat transfer"
    H_NET_PK = "H(net)-pockets only"
    H_NET_AI = "H(net)-assisted"
    H_NET_A = "H(net)-actual"
    H_NET_UT = "H(net)-ut"
    H_NET_HOT = "H(net-hot)"
    H_NET_COLD = "H(net-cold)"
    H_NET_HP = "H(net)-heat pump"
    H_NET_RFRG = "H(net)-refrigeration"
    H_NET_W_AIR = "H(net)-with air"

    H_HOT_UT = "H(hot)-utility"
    H_COLD_UT = "H(cold)-utility"
    H_NET_HOT_UT = "H(hot)-net_utility"
    H_NET_COLD_UT = "H(cold)-net_utility"
    H_HOT_BAL = "H(hot)-balanced"
    H_COLD_BAL = "H(cold)-balanced"

    H_HOT_HP = "H(hot)-hp_ut"
    H_COLD_HP = "H(cold)-hp_ut"
    H_NET_HOT_AFTR_HP = "H(hot)-net_after_hp"
    H_NET_COLD_AFTR_HP = "H(cold)-net_after_hp"
    H_NET_HOT_UT_AFTR_HP = "H(hot)-net_utility_after_hp"
    H_NET_COLD_UT_AFTR_HP = "H(cold)-net_utility_after_hp"

    H_HOT_RFRG = "H(hot)-rfrg_ut"
    H_COLD_RFRG = "H(cold)-rfrg_ut"
    H_NET_HOT_AFTR_RFRG = "H(hot)-net_after_rfrg"
    H_NET_COLD_AFTR_RFRG = "H(cold)-net_after_rfrg"
    H_NET_HOT_UT_AFTR_RFRG = "H(hot)-net_utility_after_rfrg"
    H_NET_COLD_UT_AFTR_RFRG = "H(cold)-net_utility_after_rfrg"

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

    X_GCC = "X(net)"
    X_SUR = "X(surplus)"
    X_DEF = "X(deficit)"


PT = ProblemTableLabel


class HeatExchangerNetworkLabel(Enum):
    """Heat exchanger network metric labels for labelled accessors."""

    RECOVERY_DUTY = "recovery_duty"
    HOT_UTILITY_DUTY = "hot_utility_duty"
    COLD_UTILITY_DUTY = "cold_utility_duty"
    RECOVERY_AREA = "recovery_area"
    HOT_UTILITY_AREA = "hot_utility_area"
    COLD_UTILITY_AREA = "cold_utility_area"
    HOT_RECOVERY_OUTLET_TEMPERATURE = "hot_recovery_outlet_temperature"
    COLD_RECOVERY_OUTLET_TEMPERATURE = "cold_recovery_outlet_temperature"
    MATCH_ACTIVE = "match_active"
    MATCH_ALLOWED = "match_allowed"


HEN = HeatExchangerNetworkLabel


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
    """Graph groups available in OpenPinch reporting data."""

    CC = "Composite Curves"
    SCC = "Shifted Composite Curves"
    BCC = "Balanced Composite Curves"
    GCC = "Grand Composite Curve"
    GCC_R = "Grand Composite Curve (Real)"
    GCC_X = "Exergetic Grand Composite Curve"
    NLP_X = "Exergetic Net Load Profiles"
    GCC_HP = "Grand Composite Curve with Heat Pump"
    NLP_HP = "Net Load Profiles with Heat Pump"
    NLP = "Net Load Profiles"
    ETD = "Energy Transfer Diagram"
    TSP = "Total Site Profiles"
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
    """Row semantics for tabular summary output."""

    CONTENT = "content"
    FOOTER = "footer"


class TurbineModel(Enum):
    """Alternative turbine performance correlations used in power targeting."""

    MEDINA_FLORES = "Medina-Flores et al. (2010)"
    SUN_SMITH = "Sun & Smith (2015)"
    VARBANOV = "Varbanov et al. (2004)"
    ISENTROPIC = "Fixed Isentropic Turbine"


class BB_Minimiser(str, Enum):
    """Supported optimisation backends for multistart black-box search."""

    DA = "dual_annealing"
    CMAES = "cmaes"
    BO = "bo"
    RBF = "rbf_surrogate"


__all__ = [
    "ArrowHead",
    "BB_Minimiser",
    "CogenerationTarget",
    "FluidPhase",
    "GraphType",
    "GT",
    "HeatExchangerKind",
    "HEN",
    "HeatExchangerNetworkLabel",
    "HeatExchangerStreamRole",
    "HeatExchangerTypes",
    "HeatFlowUnits",
    "HeatPump",
    "HeatPumpAndRefrigerationCycle",
    "HPRcycle",
    "LegendSeries",
    "LineColour",
    "ProblemTableLabel",
    "PT",
    "ResultsType",
    "SID",
    "SD",
    "ST",
    "StreamDataLabel",
    "StreamID",
    "StreamLoc",
    "StreamType",
    "SummaryRowType",
    "TargetType",
    "TT",
    "TurbineModel",
    "ZoneType",
    "ZT",
]
