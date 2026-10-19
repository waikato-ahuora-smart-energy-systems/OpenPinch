"""Display text for the editable configuration fields.

Each field has a label, the unit its value is given in (``None`` when it has
none) and a one-sentence description, for option editors.
"""

from __future__ import annotations

from typing import NamedTuple


class FieldText(NamedTuple):
    label: str
    unit: str | None
    description: str


# fmt: off
CONFIGURATION_FIELD_TEXT: dict[str, FieldText] = {
    "PROBLEM_TOP_ZONE_NAME": FieldText(
        "Top zone name", None,
        "Name given to the top zone of the zone tree.",
    ),
    "PROBLEM_TOP_ZONE_IDENTIFIER": FieldText(
        "Top zone type", None,
        (
            "Zone type of the top zone: Region, Community, Site, Process Zone, Utility "
            "Zone or Unit Operation."
        ),
    ),
    "PROBLEM_PERIOD_IDS": FieldText(
        "Operating period IDs", None,
        (
            "Identifiers of the operating periods, in order. Multiperiod values give "
            "one value per period."
        ),
    ),
    "PROBLEM_PERIOD_WEIGHTS": FieldText(
        "Operating period weights", None,
        (
            "Weight of each operating period in multiperiod totals and costs. A single "
            "value weights every period equally."
        ),
    ),
    "INPUT_UNIT_TEMPERATURE": FieldText(
        "Temperature input unit", None,
        "Unit assumed for input temperatures given without a unit.",
    ),
    "INPUT_UNIT_PRESSURE": FieldText(
        "Pressure input unit", None,
        "Unit assumed for input pressures given without a unit.",
    ),
    "INPUT_UNIT_ENTHALPY": FieldText(
        "Specific enthalpy input unit", None,
        "Unit assumed for input specific enthalpies given without a unit.",
    ),
    "INPUT_UNIT_HEAT_FLOW": FieldText(
        "Heat flow input unit", None,
        "Unit assumed for input heat flows given without a unit.",
    ),
    "INPUT_UNIT_DELTA_T": FieldText(
        "Temperature difference input unit", None,
        (
            "Unit assumed for input temperature differences, such as dt_cont, given "
            "without a unit."
        ),
    ),
    "INPUT_UNIT_HTC": FieldText(
        "Heat transfer coefficient input unit", None,
        "Unit assumed for input heat transfer coefficients given without a unit.",
    ),
    "INPUT_UNIT_PRICE": FieldText(
        "Utility price input unit", None,
        "Unit assumed for input utility prices given without a unit.",
    ),
    "OUTPUT_UNIT_HEAT_FLOW": FieldText(
        "Heat flow output unit", None,
        "Unit used to report heat flows.",
    ),
    "OUTPUT_UNIT_TEMPERATURE": FieldText(
        "Temperature output unit", None,
        "Unit used to report temperatures.",
    ),
    "OUTPUT_UNIT_PERCENT": FieldText(
        "Percentage output unit", None,
        "Unit used to report percentages and ratios.",
    ),
    "OUTPUT_UNIT_UTILITY_COST": FieldText(
        "Utility cost output unit", None,
        "Unit used to report utility cost rates.",
    ),
    "OUTPUT_UNIT_WORK": FieldText(
        "Work output unit", None,
        "Unit used to report work and power.",
    ),
    "OUTPUT_UNIT_AREA": FieldText(
        "Area output unit", None,
        "Unit used to report heat transfer area.",
    ),
    "OUTPUT_UNIT_CAPITAL_COST": FieldText(
        "Capital cost output unit", None,
        "Unit used to report installed capital costs.",
    ),
    "OUTPUT_UNIT_ANNUAL_COST": FieldText(
        "Annual cost output unit", None,
        "Unit used to report annual and annualised costs.",
    ),
    "OUTPUT_UNIT_EXERGY": FieldText(
        "Exergy output unit", None,
        "Unit used to report exergy flows.",
    ),
    "OUTPUT_UNIT_HPR_COP": FieldText(
        "COP output unit", None,
        "Unit used to report heat pump and refrigeration COPs.",
    ),
    "REPORTING_DECIMAL_PLACES": FieldText(
        "Decimal places", None,
        (
            "Number of decimal places for reported results. No current code reads this "
            "setting."
        ),
    ),
    "REPORTING_DEBUG_ENABLED": FieldText(
        "Debug reporting", None,
        (
            "Report extra validation warnings on loading, such as streams renamed "
            "because they share a name."
        ),
    ),
    "ENV_TEMPERATURE": FieldText(
        "Ambient temperature", "degC",
        (
            "Dead-state temperature for exergy, the ambient air level for heat pumps, "
            "and the sink for below-pinch turbines."
        ),
    ),
    "ENV_PRESSURE": FieldText(
        "Ambient pressure", "kPa",
        "Dead-state pressure. No current calculation reads this setting.",
    ),
    "THERMAL_DT_CONT": FieldText(
        "Minimum contribution temperature difference", "delta_degC",
        "The dt_cont of any stream or utility that does not give its own.",
    ),
    "THERMAL_DT_PHASE_CHANGE": FieldText(
        "Phase change temperature span", "delta_degC",
        (
            "Smallest temperature span of a stream. Narrower streams, such as phase "
            "changes, are widened to it with the same duty."
        ),
    ),
    "THERMAL_HTC": FieldText(
        "Heat transfer coefficient", "kW/m^2/delta_degC",
        (
            "The heat transfer coefficient of any stream or utility that does not give "
            "its own."
        ),
    ),
    "DIRECT_BALANCED_CC_ENABLED": FieldText(
        "Balanced composite curves", None,
        (
            "Build the balanced composite curves in direct integration. They are always"
            " built when area or capital cost is targeted."
        ),
    ),
    "DIRECT_VERTICAL_GCC_ENABLED": FieldText(
        "Vertical heat transfer GCC", None,
        (
            "Also build the grand composite curve assuming vertical heat transfer "
            "between the composite curves."
        ),
    ),
    "DIRECT_ASSISTED_HT_ENABLED": FieldText(
        "Assisted heat transfer GCC", None,
        (
            "Build an assisted-integration grand composite curve by partly cutting "
            "pockets; it then sets the utility targets."
        ),
    ),
    "DIRECT_ASSISTED_HT_DT": FieldText(
        "Assisted heat transfer pocket span", "delta_degC",
        (
            "Pockets spanning at least this temperature range are partly cut for "
            "assisted heat transfer; narrower pockets are kept."
        ),
    ),
    "COSTING_UTILITY_PRICE": FieldText(
        "Default utility price", "$/MWh",
        (
            "Price of the default utilities and of any utility that does not give its "
            "own price."
        ),
    ),
    "COSTING_ANNUAL_OP_TIME": FieldText(
        "Annual operating time", "h/y",
        "Operating hours per year, used to turn duties and prices into annual costs.",
    ),
    "COSTING_HX_UNIT_COST": FieldText(
        "Heat exchanger fixed cost", "$",
        "Fixed installed cost per heat exchanger in area-based capital costs.",
    ),
    "COSTING_HX_AREA_COEFF": FieldText(
        "Heat exchanger area cost coefficient", "$",
        (
            "Multiplies (area per exchanger in m^2) raised to the area exponent, per "
            "exchanger, in capital costs."
        ),
    ),
    "COSTING_HX_AREA_EXP": FieldText(
        "Heat exchanger area cost exponent", None,
        "Exponent on the area per exchanger in heat exchanger capital costs.",
    ),
    "COSTING_DISCOUNT_RATE": FieldText(
        "Discount rate", None,
        (
            "Annual discount rate as a fraction (0.07 is 7 %), used with the service "
            "life to annualise capital."
        ),
    ),
    "COSTING_SERVICE_LIFE": FieldText(
        "Service life", "year",
        "Equipment life in years, used with the discount rate to annualise capital.",
    ),
    "COSTING_HPR_ELE_PRICE": FieldText(
        "HPR electricity price", "$/MWh",
        (
            "Electricity price for heat pump and refrigeration costing. HPR heating and"
            " cooling prices are set relative to it."
        ),
    ),
    "COSTING_HPR_PRICE_RATIO_HEAT_TO_ELE": FieldText(
        "HPR heating to electricity price ratio", None,
        (
            "Price of external heating in HPR costing, as a multiple of the HPR "
            "electricity price."
        ),
    ),
    "COSTING_HPR_PRICE_RATIO_COOLING_WATER_TO_ELE": FieldText(
        "HPR cooling water to electricity price ratio", None,
        (
            "Price of cooling water in HPR costing, as a multiple of the HPR "
            "electricity price."
        ),
    ),
    "COSTING_HPR_COOLING_WATER_TEMPERATURE": FieldText(
        "HPR cooling water temperature", "degC",
        "Supply temperature of the default cooling water in HPR costing.",
    ),
    "COSTING_HPR_COOLING_WATER_DT_MIN": FieldText(
        "HPR cooling water approach", "delta_degC",
        (
            "Minimum approach to the cooling water; heat is rejected at its temperature"
            " plus this difference."
        ),
    ),
    "COSTING_HPR_REFRIGERATION_ETA_II": FieldText(
        "Default refrigeration Carnot efficiency", None,
        (
            "Fraction of the Carnot COP achieved by the default refrigeration utility "
            "in HPR costing."
        ),
    ),
    "COSTING_HPR_REFRIGERATION_DT": FieldText(
        "Default refrigeration evaporator approach", "delta_degC",
        (
            "The default refrigeration evaporates this far below the problem's lowest "
            "temperature."
        ),
    ),
    "COSTING_HPR_EQUIPMENT_COST": FieldText(
        "HPR equipment cost at 1 MW", "$",
        (
            "Equipment cost of a 1 MW single-stage heat pump, scaled by capacity, "
            "stages and temperature in HPR capital costs."
        ),
    ),
    "COSTING_HPR_INSTALLATION_FACTOR": FieldText(
        "HPR installation factor", None,
        "Multiplies the HPR equipment cost to give the installed capital cost.",
    ),
    "COSTING_HPR_COST_EXP": FieldText(
        "HPR capacity cost exponent", None,
        "Exponent on each machine's capacity (in MW) in HPR capital costs.",
    ),
    "COSTING_HPR_COST_FIXED_SHARE": FieldText(
        "HPR cost fixed share", None,
        "Share of HPR capital that does not depend on the number of stages.",
    ),
    "COSTING_HPR_COST_STAGE_SHARE": FieldText(
        "HPR cost share per stage", None,
        "Share of HPR capital added for each compression stage, including MVR stages.",
    ),
    "COSTING_HPR_COST_TEMP_FACTOR": FieldText(
        "HPR cost temperature factor", None,
        (
            "Fractional capital increase per 100 K that the hottest delivery "
            "temperature exceeds the cost base temperature."
        ),
    ),
    "COSTING_HPR_COST_TEMP_BASE": FieldText(
        "HPR cost base temperature", "degC",
        (
            "Delivery temperature above which HPR capital cost rises with the "
            "temperature factor."
        ),
    ),
    "COSTING_HPR_CAPITAL_RECOVERY_ENABLED": FieldText(
        "Annualise HPR capital", None,
        "Include the annualised heat pump capital in the HPR total annual cost.",
    ),
    "COSTING_HPR_UTILITY_CAPITAL_RECOVERY_ENABLED": FieldText(
        "Annualise default utility capital", None,
        (
            "Include the annualised capital of the default hot utility and "
            "refrigeration still needed in the HPR annual cost."
        ),
    ),
    "COSTING_HPR_HOT_UTILITY_CAPITAL_COST": FieldText(
        "Default hot utility capital cost", "$/kW",
        "Installed capital per kW of default hot utility capacity in HPR costing.",
    ),
    "COSTING_HPR_REFRIGERATION_CAPITAL_COST": FieldText(
        "Default refrigeration capital cost", "$/kW",
        "Installed capital per kW of default refrigeration capacity in HPR costing.",
    ),
    "COSTING_ORC_ELE_PRICE": FieldText(
        "ORC electricity price", "$/MWh",
        "Value of the net power an ORC generates.",
    ),
    "COSTING_ORC_COOLING_PRICE": FieldText(
        "ORC cooling price", "$/MWh",
        (
            "Price of cooling in ORC costing, for both the process cooling saved and "
            "the condenser cooling added."
        ),
    ),
    "COSTING_ORC_EQUIPMENT_COST": FieldText(
        "ORC equipment cost at 1 MW", "$",
        (
            "Equipment cost of a 1 MW ORC unit, scaled by each unit's net power with "
            "the cost exponent."
        ),
    ),
    "COSTING_ORC_INSTALLATION_FACTOR": FieldText(
        "ORC installation factor", None,
        "Multiplies the ORC equipment cost to give the installed capital cost.",
    ),
    "COSTING_ORC_COST_EXP": FieldText(
        "ORC power cost exponent", None,
        "Exponent on each unit's net power (in MW) in ORC capital costs.",
    ),
    "COSTING_ORC_CAPITAL_RECOVERY_ENABLED": FieldText(
        "Annualise ORC capital", None,
        "Include the annualised ORC capital in its annual cost change.",
    ),
    "HENS_APPROACH_TEMPERATURES": FieldText(
        "HEN approach temperatures", "delta_degC",
        (
            "Minimum approach temperatures (dTmin) to synthesise networks for; one set "
            "of design tasks per value."
        ),
    ),
    "HENS_DT_CONT_MULTIPLIERS": FieldText(
        "HEN dTmin multipliers", None,
        (
            "Multipliers on the first approach temperature for extra pinch design "
            "candidates at quality tier 2 and above. None uses the tier's defaults."
        ),
    ),
    "HENS_DERIVATIVE_THRESHOLDS": FieldText(
        "Thermal derivative thresholds", None,
        (
            "Minimum dQ/dA each match must meet in thermal derivative method runs; one "
            "run per value."
        ),
    ),
    "HENS_SYNTHESIS_QUALITY_TIER": FieldText(
        "HEN synthesis quality tier", None,
        (
            "Search breadth from 0 to 5: 0 skips the thermal derivative method, 1 is "
            "standard OpenHENS, 2 to 5 search more widely."
        ),
    ),
    "HENS_PDM_STAGE_PAIR_LIMIT": FieldText(
        "Pinch design stage pair candidates", None,
        (
            "Number of extra pinch design candidates with set stage counts above and "
            "below the pinch, up to 12. None adds none."
        ),
    ),
    "HENS_TDM_PARENT_LIMIT": FieldText(
        "Thermal derivative parent limit", None,
        "Recorded in the synthesis manifest; no current synthesis step reads it.",
    ),
    "HENS_STAGE_PACKING": FieldText(
        "Stage packing", None,
        (
            "Methods whose integer models keep active stages contiguous: none, pdm, tdm"
            " or all. auto currently means none."
        ),
    ),
    "HENS_STAGE_SELECTION": FieldText(
        "HEN stage counts", None,
        (
            "Stage counts for direct network evolution runs. Exactly two values also "
            "fix the pinch design stages above and below the pinch."
        ),
    ),
    "HENS_SOLVER_PDM": FieldText(
        "Pinch design solver", None,
        "Solver for the pinch design method models.",
    ),
    "HENS_SOLVER_TDM": FieldText(
        "Thermal derivative solver", None,
        "Solver for the thermal derivative method models.",
    ),
    "HENS_SOLVER_EVM": FieldText(
        "Network evolution solver", None,
        "Solver for the network evolution method models.",
    ),
    "HENS_SOLVER_OPTIONS_PDM": FieldText(
        "Pinch design solver options", None,
        "Options passed to the pinch design solver.",
    ),
    "HENS_SOLVER_OPTIONS_TDM": FieldText(
        "Thermal derivative solver options", None,
        "Options passed to the thermal derivative solver.",
    ),
    "HENS_SOLVER_OPTIONS_EVM": FieldText(
        "Network evolution solver options", None,
        "Options passed to the network evolution solver.",
    ),
    "HENS_SOLVE_TOLERANCE": FieldText(
        "HEN solve tolerance", None,
        "Numerical tolerance passed to the HEN synthesis models and duty optimisation.",
    ),
    "HENS_MAX_PARALLEL": FieldText(
        "HEN parallel workers", None,
        "Maximum number of network evolution candidates solved at once.",
    ),
    "HENS_EVM_N_AD_BRANCHES": FieldText(
        "Network evolution add branches", None,
        (
            "Match-adding candidates explored per network evolution step. None uses the"
            " quality tier's default."
        ),
    ),
    "HENS_EVM_N_RM_BRANCHES": FieldText(
        "Network evolution remove branches", None,
        (
            "Match-removing candidates explored per network evolution step. None uses "
            "the quality tier's default."
        ),
    ),
    "HENS_EVM_BEAM_WIDTH": FieldText(
        "Network evolution beam width", None,
        (
            "Number of best topologies kept between network evolution steps when more "
            "than one branch is explored."
        ),
    ),
    "HENS_LOG_LEVEL": FieldText(
        "HEN log level", None,
        (
            "Logging level name, such as INFO or WARNING. Recorded with the synthesis "
            "settings; no current code applies it."
        ),
    ),
    "HENS_OUTPUT_FOLDER": FieldText(
        "HEN output folder", None,
        (
            "Folder for synthesis output files. Recorded with the synthesis settings; "
            "exports take their own folder."
        ),
    ),
    "HENS_OUTPUT_FORMATS": FieldText(
        "HEN export formats", None,
        "Export formats (json, csv, xlsx) recorded in the synthesis manifest.",
    ),
    "HENS_RUN_ID": FieldText(
        "HEN run ID", None,
        (
            "Identifier recorded on synthesis tasks, results and exports. Letters, "
            "digits, underscores, hyphens and periods."
        ),
    ),
    "HENS_BEST_SOLUTIONS_TO_SAVE": FieldText(
        "HEN solutions to save", None,
        "Recorded in the synthesis manifest; no current synthesis step reads it.",
    ),
    "HPR_LOAD_MODE": FieldText(
        "HPR duty setting mode", None,
        (
            "How the HPR duty is set: fraction of the available duty, a fixed duty, or "
            "a duty per period (period_values)."
        ),
    ),
    "HPR_LOAD_FRACTION": FieldText(
        "HPR share of available duty", None,
        (
            "Fraction of the available heating (or cooling) duty the heat pump or "
            "refrigeration serves, in fraction mode."
        ),
    ),
    "HPR_LOAD_DUTY": FieldText(
        "HPR target duty", "kW",
        "Heat pump or refrigeration duty in duty mode, capped at the available duty.",
    ),
    "HPR_LOAD_PERIOD_VALUES": FieldText(
        "HPR duty per period", "kW",
        (
            "Duty for each period ID in period_values mode, capped at that period's "
            "available duty."
        ),
    ),
    "HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": FieldText(
        "Shared multiperiod HPR design", None,
        (
            "Optimise one heat pump or refrigeration design across all periods, "
            "weighted by period, instead of one period."
        ),
    ),
    "HPR_REFRIGERANTS": FieldText(
        "Vapour compression refrigerants", None,
        "Refrigerants for the vapour compression stages, one per stage.",
    ),
    "HPR_REFRIGERANT_SORT_ENABLED": FieldText(
        "Sort refrigerants by critical temperature", None,
        (
            "Assign refrigerants to stages from the highest critical temperature down "
            "instead of in the given order."
        ),
    ),
    "HPR_MVR_FLUIDS": FieldText(
        "MVR fluids", None,
        "Working fluids of the MVR stages in vapour compression with MVR cycles.",
    ),
    "HPR_MVR_COUNT": FieldText(
        "MVR stage count", None,
        "Number of MVR stages in vapour compression with MVR cycles.",
    ),
    "HPR_MVR_ETA_COMP": FieldText(
        "MVR compressor efficiency", None,
        "Isentropic efficiency of the MVR compressors in HPR cycles.",
    ),
    "HPR_MVR_ETA_MOTOR": FieldText(
        "MVR motor efficiency", None,
        "Motor efficiency of the MVR compressors in HPR cycles.",
    ),
    "HPR_N_COND": FieldText(
        "HPR condenser levels", None,
        (
            "Number of condensing temperature levels in the heat pump or refrigeration "
            "cycle."
        ),
    ),
    "HPR_N_EVAP": FieldText(
        "HPR evaporator levels", None,
        (
            "Number of evaporating temperature levels in the heat pump or refrigeration"
            " cycle."
        ),
    ),
    "HPR_ETA_COMP": FieldText(
        "HPR compressor efficiency", None,
        (
            "Isentropic efficiency of the compressors in simulated vapour compression "
            "and Brayton cycles."
        ),
    ),
    "HPR_ETA_EXP": FieldText(
        "HPR expander efficiency", None,
        "Isentropic efficiency of the expander in Brayton cycles.",
    ),
    "HPR_ETA_II_CARNOT": FieldText(
        "HPR Carnot efficiency", None,
        (
            "Second-law efficiency of Carnot-model heat pumps and refrigeration: "
            "the fraction of the cooling Carnot COP, T_l/(T_h - T_l), they achieve. "
            "The heating COP follows from the energy balance, "
            "1 + eta x T_l/(T_h - T_l)."
        ),
    ),
    "HPR_HE_ETA_II_CARNOT": FieldText(
        "Integrated heat engine Carnot efficiency", None,
        (
            "Fraction of the Carnot efficiency achieved by the integrated expander "
            "(heat engine), when it is enabled."
        ),
    ),
    "HPR_INTEGRATED_EXPANDER_ENABLED": FieldText(
        "Integrated expander", None,
        (
            "Let Carnot-model cycles generate work where heat flows from a hotter "
            "evaporator level to a colder condenser level."
        ),
    ),
    "HPR_DT_CONT": FieldText(
        "HPR contribution temperature difference", "delta_degC",
        (
            "dt_cont of the heat pump or refrigeration working fluid, used to shift the"
            " process profiles it exchanges heat with."
        ),
    ),
    "HPR_DT_IHX": FieldText(
        "HPR internal heat exchanger approach", "delta_degC",
        (
            "Only checked by TESPy targeting, which rejects a nonzero value; no other "
            "HPR model reads it."
        ),
    ),
    "HPR_DT_CASCADE_HX": FieldText(
        "HPR cascade heat exchanger approach", "delta_degC",
        (
            "Temperature difference between a lower cycle's condenser and the upper "
            "cycle's evaporator in cascades."
        ),
    ),
    "HPR_DT_ENV_CONT": FieldText(
        "Ambient air contribution temperature difference", "delta_degC",
        "dt_cont of ambient air when it acts as a heat pump source or sink.",
    ),
    "HPR_MAX_MULTISTART": FieldText(
        "HPR optimiser restarts", None,
        "Number of HPR optimiser runs from different starting points.",
    ),
    "HPR_ETA_PENALTY": FieldText(
        "HPR efficiency penalty weight", None,
        "Passed to the HPR targeting inputs, but no current HPR calculation reads it.",
    ),
    "HPR_RHO_PENALTY": FieldText(
        "HPR feasibility penalty weight", None,
        (
            "Weight of the penalty on constraint violations, relative to the targeted "
            "duty, in HPR optimisation."
        ),
    ),
    "HPR_BB_MINIMISER": FieldText(
        "HPR optimiser", None,
        (
            "Black-box optimiser for HPR targeting: dual_annealing, cmaes, bo or "
            "rbf_surrogate."
        ),
    ),
    "HPR_INITIALISE_SIMULATED_CYCLE": FieldText(
        "Seed simulated cycles from Carnot", None,
        "Start simulated vapour compression searches from a Carnot-model placement.",
    ),
    "ORC_N_STAGES": FieldText(
        "ORC units", None,
        "Number of parallel ORC units, each evaporating at its own temperature.",
    ),
    "ORC_ETA_II_CARNOT": FieldText(
        "ORC Carnot efficiency", None,
        "Fraction of the Carnot efficiency achieved by the Carnot-model ORC.",
    ),
    "ORC_DT_CONT": FieldText(
        "ORC contribution temperature difference", "delta_degC",
        "dt_cont of the ORC working fluid against the process heat source.",
    ),
    "ORC_T_COND": FieldText(
        "ORC condensing temperature", "degC",
        "Condensing temperature of the ORC.",
    ),
    "ORC_MIN_LIFT": FieldText(
        "ORC minimum lift", "delta_degC",
        "Smallest difference between the ORC evaporating and condensing temperatures.",
    ),
    "ORC_LOAD_FRACTION": FieldText(
        "ORC share of heat surplus", None,
        "Largest fraction of the below-pinch heat surplus the ORC may take.",
    ),
    "ORC_MAX_MULTISTART": FieldText(
        "ORC optimiser restarts", None,
        "Number of ORC optimiser runs from different starting points.",
    ),
    "ORC_BB_MINIMISER": FieldText(
        "ORC optimiser", None,
        (
            "Black-box optimiser for ORC targeting: dual_annealing, cmaes, bo or "
            "rbf_surrogate."
        ),
    ),
    "ORC_FLUIDS": FieldText(
        "ORC working fluids", None,
        "Candidate working fluids for the simulated ORC.",
    ),
    "ORC_ETA_TURBINE": FieldText(
        "ORC turbine efficiency", None,
        "Isentropic efficiency of the simulated ORC turbine.",
    ),
    "ORC_ETA_PUMP": FieldText(
        "ORC pump efficiency", None,
        "Isentropic efficiency of the simulated ORC feed pump.",
    ),
    "ORC_MAX_SUPERHEAT": FieldText(
        "ORC maximum superheat", "delta_degC",
        "Largest superheat the simulated ORC search may use.",
    ),
    "ORC_RECUPERATOR_ENABLED": FieldText(
        "ORC recuperator", None,
        "Add a recuperator to the simulated ORC.",
    ),
    "ORC_DT_RECUPERATOR": FieldText(
        "ORC recuperator approach", "delta_degC",
        "Minimum approach temperature of the ORC recuperator, when it is enabled.",
    ),
    "PROCESS_MVR_ETA_COMP": FieldText(
        "Process MVR compressor efficiency", None,
        (
            "Isentropic efficiency of direct process MVR compressors, unless the call "
            "gives its own."
        ),
    ),
    "PROCESS_MVR_ETA_MOTOR": FieldText(
        "Process MVR motor efficiency", None,
        (
            "Motor efficiency of direct process MVR compressors, unless the call gives "
            "its own."
        ),
    ),
    "POWER_TURBINE_WORK_ENABLED": FieldText(
        "Turbine work", None,
        (
            "When on, a turbine inlet pressure above 220 bar is reset to 200 bar on "
            "loading. No other code reads it."
        ),
    ),
    "POWER_TURB_T_IN": FieldText(
        "Turbine inlet temperature", "degC",
        "Steam temperature at the turbine inlet for above-pinch cogeneration.",
    ),
    "POWER_TURB_P_IN": FieldText(
        "Turbine inlet pressure", "bar",
        "Steam pressure at the turbine inlet for above-pinch cogeneration.",
    ),
    "POWER_MIN_EFF": FieldText(
        "Turbine minimum efficiency", None,
        (
            "Lowest isentropic efficiency a turbine stage is given when its model "
            "predicts less."
        ),
    ),
    "POWER_LOAD_FRACTION": FieldText(
        "Turbine load fraction", None,
        (
            "Operating flow as a fraction of turbine capacity, used by the efficiency "
            "models for part load."
        ),
    ),
    "POWER_ETA_MECH": FieldText(
        "Turbine mechanical efficiency", None,
        "Mechanical efficiency applied to turbine work.",
    ),
    "POWER_HIGH_P_COND_FLASH_ENABLED": FieldText(
        "Flash high-pressure condensate", None,
        (
            "Let condensate from higher-pressure steam levels flash at lower levels, "
            "reducing the steam they need."
        ),
    ),
}
# fmt: on
