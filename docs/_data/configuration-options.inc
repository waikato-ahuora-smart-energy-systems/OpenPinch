Problem
~~~~~~~

.. csv-table::
   :header: "Option", "Type", "Default", "Unit", "Limits", "Description"
   :widths: 22, 8, 12, 8, 10, 40

   "``PROBLEM_TOP_ZONE_NAME``","string","``Site``","","","Name given to the top zone of the zone tree."
   "``PROBLEM_TOP_ZONE_IDENTIFIER``","enum","``Site``","","Region, Community, Site, Process Zone, Utility Zone, Unit Operation","Zone type of the top zone: Region, Community, Site, Process Zone, Utility Zone or Unit Operation."
   "``PROBLEM_PERIOD_IDS``","string_list","``[""0""]``","","","Identifiers of the operating periods, in order. Multiperiod values give one value per period."
   "``PROBLEM_PERIOD_WEIGHTS``","number_list","``[1.0]``","",">= 0","Weight of each operating period in multiperiod totals and costs. A single value weights every period equally."

Input Units
~~~~~~~~~~~

.. csv-table::
   :header: "Option", "Type", "Default", "Unit", "Limits", "Description"
   :widths: 22, 8, 12, 8, 10, 40

   "``INPUT_UNIT_TEMPERATURE``","string","``degC``","","","Unit assumed for input temperatures given without a unit."
   "``INPUT_UNIT_PRESSURE``","string","``kPa``","","","Unit assumed for input pressures given without a unit."
   "``INPUT_UNIT_ENTHALPY``","string","``kJ/kg``","","","Unit assumed for input specific enthalpies given without a unit."
   "``INPUT_UNIT_HEAT_FLOW``","string","``kW``","","","Unit assumed for input heat flows given without a unit."
   "``INPUT_UNIT_DELTA_T``","string","``delta_degC``","","","Unit assumed for input temperature differences, such as dt_cont, given without a unit."
   "``INPUT_UNIT_HTC``","string","``kW/m^2/delta_degC``","","","Unit assumed for input heat transfer coefficients given without a unit."
   "``INPUT_UNIT_PRICE``","string","``$/MWh``","","","Unit assumed for input utility prices given without a unit."

Output Units
~~~~~~~~~~~~

.. csv-table::
   :header: "Option", "Type", "Default", "Unit", "Limits", "Description"
   :widths: 22, 8, 12, 8, 10, 40

   "``OUTPUT_UNIT_HEAT_FLOW``","string","``kW``","","","Unit used to report heat flows."
   "``OUTPUT_UNIT_TEMPERATURE``","string","``degC``","","","Unit used to report temperatures."
   "``OUTPUT_UNIT_PERCENT``","string","``%``","","","Unit used to report percentages and ratios."
   "``OUTPUT_UNIT_UTILITY_COST``","string","``$/h``","","","Unit used to report utility cost rates. A per-year unit counts a year as COSTING_ANNUAL_OP_TIME operating hours."
   "``OUTPUT_UNIT_WORK``","string","``kW``","","","Unit used to report work and power."
   "``OUTPUT_UNIT_AREA``","string","``m^2``","","","Unit used to report heat transfer area."
   "``OUTPUT_UNIT_CAPITAL_COST``","string","``$``","","","Unit used to report installed capital costs."
   "``OUTPUT_UNIT_ANNUAL_COST``","string","``$/y``","","","Unit used to report annual and annualised costs. A year is COSTING_ANNUAL_OP_TIME operating hours."
   "``OUTPUT_UNIT_EXERGY``","string","``kW``","","","Unit used to report exergy flows."
   "``OUTPUT_UNIT_HPR_COP``","string","``dimensionless``","","","Unit used to report heat pump and refrigeration COPs."

Reporting
~~~~~~~~~

.. csv-table::
   :header: "Option", "Type", "Default", "Unit", "Limits", "Description"
   :widths: 22, 8, 12, 8, 10, 40

   "``REPORTING_DECIMAL_PLACES``","number","``2``","",">= 0","Number of decimal places for reported results. No current code reads this setting."
   "``REPORTING_DEBUG_ENABLED``","boolean","``False``","","","Report extra validation warnings on loading, such as streams renamed because they share a name."

Environment
~~~~~~~~~~~

.. csv-table::
   :header: "Option", "Type", "Default", "Unit", "Limits", "Description"
   :widths: 22, 8, 12, 8, 10, 40

   "``ENV_TEMPERATURE``","number","``15.0``","degC","","Dead-state temperature for exergy, the ambient air level for heat pumps, and the sink for below-pinch turbines."
   "``ENV_PRESSURE``","number","``101.0``","kPa",">= 0","Dead-state pressure. No current calculation reads this setting."

Thermal
~~~~~~~

.. csv-table::
   :header: "Option", "Type", "Default", "Unit", "Limits", "Description"
   :widths: 22, 8, 12, 8, 10, 40

   "``THERMAL_DT_CONT``","number","``5.0``","delta_degC",">= 0","The dt_cont of any stream or utility that does not give its own."
   "``THERMAL_DT_PHASE_CHANGE``","number","``0.01``","delta_degC",">= 0","Smallest temperature span of a stream. Narrower streams, such as phase changes, are widened to it with the same duty."
   "``THERMAL_HTC``","number","``1.0``","kW/m^2/delta_degC","> 0","The heat transfer coefficient of any stream or utility that does not give its own."

Direct Integration
~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Option", "Type", "Default", "Unit", "Limits", "Description"
   :widths: 22, 8, 12, 8, 10, 40

   "``DIRECT_BALANCED_CC_ENABLED``","boolean","``True``","","","Build the balanced composite curves in direct integration. They are always built when area or capital cost is targeted."
   "``DIRECT_VERTICAL_GCC_ENABLED``","boolean","``False``","","","Also build the grand composite curve assuming vertical heat transfer between the composite curves."
   "``DIRECT_ASSISTED_HT_ENABLED``","boolean","``False``","","","Build an assisted-integration grand composite curve by partly cutting pockets; it then sets the utility targets."
   "``DIRECT_ASSISTED_HT_DT``","number","``10.0``","delta_degC",">= 0","Pockets spanning at least this temperature range are partly cut for assisted heat transfer; narrower pockets are kept."

Costing
~~~~~~~

.. csv-table::
   :header: "Option", "Type", "Default", "Unit", "Limits", "Description"
   :widths: 22, 8, 12, 8, 10, 40

   "``COSTING_UTILITY_PRICE``","number","``100.0``","$/MWh",">= 0","Price of the default utilities and of any utility that does not give its own price."
   "``COSTING_ANNUAL_OP_TIME``","number","``8300.0``","h/y","> 0","Operating hours per year, used to turn duties and prices into annual costs."
   "``COSTING_HX_UNIT_COST``","number","``0.0``","$",">= 0","Fixed installed cost per heat exchanger in area-based capital costs."
   "``COSTING_HX_AREA_COEFF``","number","``10000.0``","$",">= 0","Multiplies (area per exchanger in m^2) raised to the area exponent, per exchanger, in capital costs."
   "``COSTING_HX_AREA_EXP``","number","``0.6``","",">= 0","Exponent on the area per exchanger in heat exchanger capital costs."
   "``COSTING_DISCOUNT_RATE``","number","``0.07``","",">= 0","Annual discount rate as a fraction (0.07 is 7 %), used with the service life to annualise capital."
   "``COSTING_SERVICE_LIFE``","number","``20.0``","year",">= 1","Equipment life in years, used with the discount rate to annualise capital."
   "``COSTING_HPR_ELE_PRICE``","number","``100.0``","$/MWh",">= 0","Electricity price for heat pump and refrigeration costing. HPR heating and cooling prices are set relative to it."
   "``COSTING_HPR_PRICE_RATIO_HEAT_TO_ELE``","number","``1.0``","",">= 0","Price of external heating in HPR costing, as a multiple of the HPR electricity price."
   "``COSTING_HPR_PRICE_RATIO_COOLING_WATER_TO_ELE``","number","``0.025``","",">= 0","Price of cooling water in HPR costing, as a multiple of the HPR electricity price."
   "``COSTING_HPR_COOLING_WATER_TEMPERATURE``","number","``25.0``","degC","","Supply temperature of the default cooling water in HPR costing."
   "``COSTING_HPR_COOLING_WATER_DT_MIN``","number","``5.0``","delta_degC",">= 0","Minimum approach to the cooling water; heat is rejected at its temperature plus this difference."
   "``COSTING_HPR_REFRIGERATION_ETA_II``","number","``0.4``","","> 0","Fraction of the Carnot COP achieved by the default refrigeration utility in HPR costing."
   "``COSTING_HPR_REFRIGERATION_DT``","number","``5.0``","delta_degC",">= 0","The default refrigeration evaporates this far below the problem's lowest temperature."
   "``COSTING_HPR_EQUIPMENT_COST``","number","``485000.0``","$",">= 0","Equipment cost of a 1 MW single-stage heat pump, scaled by capacity, stages and temperature in HPR capital costs."
   "``COSTING_HPR_INSTALLATION_FACTOR``","number","``2.3``","",">= 0","Multiplies the HPR equipment cost to give the installed capital cost."
   "``COSTING_HPR_COST_EXP``","number","``0.7``","",">= 0","Exponent on each machine's capacity (in MW) in HPR capital costs."
   "``COSTING_HPR_COST_FIXED_SHARE``","number","``0.7``","",">= 0","Share of HPR capital that does not depend on the number of stages."
   "``COSTING_HPR_COST_STAGE_SHARE``","number","``0.3``","",">= 0","Share of HPR capital added for each compression stage, including MVR stages."
   "``COSTING_HPR_COST_TEMP_FACTOR``","number","``0.4``","",">= 0","Fractional capital increase per 100 K that the hottest delivery temperature exceeds the cost base temperature."
   "``COSTING_HPR_COST_TEMP_BASE``","number","``75.0``","degC","","Delivery temperature above which HPR capital cost rises with the temperature factor."
   "``COSTING_HPR_CAPITAL_RECOVERY_ENABLED``","boolean","``True``","","","Include the annualised heat pump capital in the HPR total annual cost."
   "``COSTING_HPR_UTILITY_CAPITAL_RECOVERY_ENABLED``","boolean","``True``","","","Include the annualised capital of the default hot utility and refrigeration still needed in the HPR annual cost."
   "``COSTING_HPR_HOT_UTILITY_CAPITAL_COST``","number","``750.0``","$/kW",">= 0","Installed capital per kW of default hot utility capacity in HPR costing."
   "``COSTING_HPR_REFRIGERATION_CAPITAL_COST``","number","``1500.0``","$/kW",">= 0","Installed capital per kW of default refrigeration capacity in HPR costing."
   "``COSTING_ORC_ELE_PRICE``","number","``100.0``","$/MWh",">= 0","Value of the net power an ORC generates."
   "``COSTING_ORC_COOLING_PRICE``","number","``2.5``","$/MWh",">= 0","Price of cooling in ORC costing, for both the process cooling saved and the condenser cooling added."
   "``COSTING_ORC_EQUIPMENT_COST``","number","``2300000.0``","$",">= 0","Equipment cost of a 1 MW ORC unit, scaled by each unit's net power with the cost exponent."
   "``COSTING_ORC_INSTALLATION_FACTOR``","number","``1.3``","",">= 0","Multiplies the ORC equipment cost to give the installed capital cost."
   "``COSTING_ORC_COST_EXP``","number","``0.75``","",">= 0","Exponent on each unit's net power (in MW) in ORC capital costs."
   "``COSTING_ORC_CAPITAL_RECOVERY_ENABLED``","boolean","``True``","","","Include the annualised ORC capital in its annual cost change."

Heat Exchanger Network Synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Option", "Type", "Default", "Unit", "Limits", "Description"
   :widths: 22, 8, 12, 8, 10, 40

   "``HENS_APPROACH_TEMPERATURES``","number_list","``[14.0]``","delta_degC","","Minimum approach temperatures (dTmin) to synthesise networks for; one set of design tasks per value."
   "``HENS_DT_CONT_MULTIPLIERS``","number_list","``None``","","","Multipliers on the first approach temperature for extra pinch design candidates at quality tier 2 and above. None uses the tier's defaults."
   "``HENS_DERIVATIVE_THRESHOLDS``","number_list","``[0.5]``","","","Minimum dQ/dA each match must meet in thermal derivative method runs; one run per value."
   "``HENS_MIN_EXCHANGER_APPROACH_SHARE``","number","``0.0``","","","Smallest exchanger approach the thermal derivative and network evolution steps may choose, as a share of the requested approach temperature, from 0 up to but not including 1. Those steps choose approaches by cost, down to 0.1 K by default (0); at 0.5 a 10 K design keeps every exchanger at 5 K or more."
   "``HENS_SYNTHESIS_QUALITY_TIER``","number","``1``","","0 to 5","Search breadth from 0 to 5: 0 skips the thermal derivative method, 1 is standard OpenHENS, 2 to 5 search more widely."
   "``HENS_PDM_STAGE_PAIR_LIMIT``","number","``None``","","0 to 12","Number of extra pinch design candidates with set stage counts above and below the pinch, up to 12. None adds none."
   "``HENS_TDM_PARENT_LIMIT``","number","``None``","",">= 1","Recorded in the synthesis manifest; no current synthesis step reads it."
   "``HENS_STAGE_PACKING``","string","``auto``","","","Methods whose integer models keep active stages contiguous: none, pdm, tdm or all. auto currently means none."
   "``HENS_STAGE_SELECTION``","number_list","``[1, 2, 3]``","","","Stage counts for direct network evolution runs. Exactly two values also fix the pinch design stages above and below the pinch."
   "``HENS_SOLVER_PDM``","string","``couenne``","","","Solver for the pinch design method models."
   "``HENS_SOLVER_TDM``","string","``couenne``","","","Solver for the thermal derivative method models."
   "``HENS_SOLVER_EVM``","string","``ipopt-pyomo``","","","Solver for the network evolution method models."
   "``HENS_SOLVER_OPTIONS_PDM``","object","``{}``","","","Options passed to the pinch design solver: known tolerances, iteration limits and numeric settings, each a number, a boolean or one word."
   "``HENS_SOLVER_OPTIONS_TDM``","object","``{}``","","","Options passed to the thermal derivative solver: known tolerances, iteration limits and numeric settings, each a number, a boolean or one word."
   "``HENS_SOLVER_OPTIONS_EVM``","object","``{}``","","","Options passed to the network evolution solver: known tolerances, iteration limits and numeric settings, each a number, a boolean or one word."
   "``HENS_SOLVE_TOLERANCE``","number","``0.001``","",">= 0","Numerical tolerance passed to the HEN synthesis models and duty optimisation."
   "``HENS_MAX_PARALLEL``","number","``1``","",">= 1","Maximum number of network evolution candidates solved at once."
   "``HENS_EVM_N_AD_BRANCHES``","number","``None``","",">= 1","Match-adding candidates explored per network evolution step on every route, including the standard OpenHENS route. None uses the quality tier's default, and 1 on tier 0 and 1 routes."
   "``HENS_EVM_N_RM_BRANCHES``","number","``None``","",">= 1","Match-removing candidates explored per network evolution step on every route, including the standard OpenHENS route. None uses the quality tier's default, and 1 on tier 0 and 1 routes."
   "``HENS_EVM_BEAM_WIDTH``","number","``4``","",">= 1","Number of best topologies kept between network evolution steps when more than one branch is explored."
   "``HENS_LOG_LEVEL``","string","``INFO``","","","Logging level name, such as INFO or WARNING. Recorded with the synthesis settings; no current code applies it."
   "``HENS_OUTPUT_FOLDER``","string","``""""``","","","Folder for synthesis output files. Recorded with the synthesis settings; exports take their own folder."
   "``HENS_OUTPUT_FORMATS``","string_list","``[]``","","","Export formats (json, csv, xlsx) recorded in the synthesis manifest."
   "``HENS_RUN_ID``","string","``default``","","","Identifier recorded on synthesis tasks, results and exports. Letters, digits, underscores, hyphens and periods."
   "``HENS_BEST_SOLUTIONS_TO_SAVE``","number","``1``","",">= 1","Recorded in the synthesis manifest; no current synthesis step reads it."

Heat Pump and Refrigeration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Option", "Type", "Default", "Unit", "Limits", "Description"
   :widths: 22, 8, 12, 8, 10, 40

   "``HPR_LOAD_MODE``","string","``fraction``","","","How the HPR duty is set: fraction of the available duty, a fixed duty, or a duty per period (period_values)."
   "``HPR_LOAD_FRACTION``","number","``1.0``","","0 to 1","Fraction of the available heating (or cooling) duty the heat pump or refrigeration serves, in fraction mode."
   "``HPR_LOAD_DUTY``","number","``None``","kW",">= 0","Heat pump or refrigeration duty in duty mode, capped at the available duty."
   "``HPR_LOAD_PERIOD_VALUES``","object","``{}``","kW","","Duty for each period ID in period_values mode, capped at that period's available duty."
   "``HPR_MULTIPERIOD_OPTIMIZATION_ENABLED``","boolean","``False``","","","Optimise one heat pump or refrigeration design across all periods, weighted by period, instead of one period."
   "``HPR_REFRIGERANTS``","string_list","``[""water"", ""ammonia""]``","","","Refrigerants for the vapour compression stages, one per stage."
   "``HPR_REFRIGERANT_SORT_ENABLED``","boolean","``True``","","","Assign refrigerants to stages from the highest critical temperature down instead of in the given order."
   "``HPR_MVR_FLUIDS``","string_list","``[""Water""]``","","","Working fluids of the MVR stages in vapour compression with MVR cycles."
   "``HPR_MVR_COUNT``","number","``1``","",">= 1","Number of MVR stages in vapour compression with MVR cycles."
   "``HPR_MVR_ETA_COMP``","number","``0.7``","","> 0 to 1","Isentropic efficiency of the MVR compressors in HPR cycles."
   "``HPR_MVR_COMPRESSOR_MODEL``","string","``constant``","","","Isentropic efficiency model of the MVR compressors: constant (HPR_MVR_ETA_COMP), pareto_front, wang_cleland or brendel, as for HPR_COMPRESSOR_MODEL."
   "``HPR_MVR_ETA_MOTOR``","number","``0.95``","","> 0 to 1","Motor efficiency of the MVR compressors in HPR cycles."
   "``HPR_N_COND``","number","``3``","",">= 0","Number of condensing temperature levels in the heat pump or refrigeration cycle."
   "``HPR_N_EVAP``","number","``2``","",">= 0","Number of evaporating temperature levels in the heat pump or refrigeration cycle."
   "``HPR_ETA_COMP``","number","``0.7``","","> 0 to 1","Isentropic efficiency of the compressors in simulated vapour compression and Brayton cycles."
   "``HPR_ETA_MOTOR``","number","``0.95``","","> 0 to 1","Motor efficiency of the vapour compression compressors. The electrical work is the shaft work divided by it; the losses leave the cycle."
   "``HPR_COMPRESSOR_MODEL``","string","``constant``","","","Isentropic efficiency model of the vapour compression compressors: constant (HPR_ETA_COMP), pareto_front (HeatPumpLab's best published efficiency against pressure ratio), wang_cleland (a polynomial in pressure ratio) or brendel (pressure ratio and suction pressure). The Python API also takes a function of the compressor conditions."
   "``HPR_ETA_EXP``","number","``0.7``","","0 to 1","Isentropic efficiency of the expander in Brayton cycles."
   "``HPR_ETA_II_CARNOT``","number","``0.5``","","> 0 to 1","Second-law efficiency of Carnot-model heat pumps and refrigeration: the fraction of the cooling Carnot COP, T_l/(T_h - T_l), they achieve. The heating COP follows from the energy balance, 1 + eta x T_l/(T_h - T_l)."
   "``HPR_HE_ETA_II_CARNOT``","number","``0.5``","","0 to 1","Fraction of the Carnot efficiency achieved by the integrated expander (heat engine), when it is enabled."
   "``HPR_INTEGRATED_EXPANDER_ENABLED``","boolean","``False``","","","Let Carnot-model cycles generate work where heat flows from a hotter evaporator level to a colder condenser level. Heat only moves downhill, one condenser-evaporator pair at a time; with this off, such a pair is plain heat recovery with no work credit."
   "``HPR_DT_CONT``","number","``0.0``","delta_degC",">= 0","dt_cont of the heat pump or refrigeration working fluid, used to shift the process profiles it exchanges heat with."
   "``HPR_DT_IHX``","number","``0.0``","delta_degC",">= 0","Only checked by TESPy targeting, which rejects a nonzero value; no other HPR model reads it."
   "``HPR_DT_CASCADE_HX``","number","``0.0``","delta_degC",">= 0","Temperature difference between a lower cycle's condenser and the upper cycle's evaporator in cascades."
   "``HPR_AMBIENT_AIR_ENABLED``","boolean","``True``","","","Whether ambient air may act as a free heat source or sink in HPR targeting. When off, no air is placed and no AIR streams are added."
   "``HPR_DT_ENV_CONT``","number","``10.0``","delta_degC",">= 0","dt_cont of ambient air when it acts as a heat pump source or sink."
   "``HPR_MAX_PRESSURE_RATIO``","number","``15.0``","",">= 1","Largest ratio of condensing to evaporating pressure across one vapour-compression compressor."
   "``HPR_SCREEN_MIN_SUCTION_PRESSURE``","number","``0.5``","bar",">= 0","Refrigerant screening flags a fluid whose evaporating saturation pressure is below this as deep vacuum suction."
   "``HPR_SCREEN_CRITICAL_MARGIN``","number","``5.0``","delta_degC",">= 0","Refrigerant screening flags a fluid that condenses within this of its critical temperature as near critical."
   "``HPR_SCREEN_MAX_STAGE_LIFT``","number","``70.0``","delta_degC",">= 0","Refrigerant screening also screens a cascade for a single-stage or cascade layout with a stage lifting more than this, splitting the stage into equal lifts within it. Must exceed HPR_DT_CASCADE_HX."
   "``HPR_MAX_MULTISTART``","number","``10``","",">= 0","Number of HPR optimiser runs from different starting points."
   "``HPR_ETA_PENALTY``","number","``0.001``","",">= 0","Passed to the HPR targeting inputs, but no current HPR calculation reads it."
   "``HPR_RHO_PENALTY``","number","``10.0``","",">= 0","Weight of the penalty on constraint violations, relative to the targeted duty, in HPR optimisation."
   "``HPR_BB_MINIMISER``","enum","``cmaes``","","dual_annealing, cmaes, bo, rbf_surrogate","Black-box optimiser for HPR targeting: dual_annealing, cmaes, bo or rbf_surrogate."
   "``HPR_INITIALISE_SIMULATED_CYCLE``","boolean","``True``","","","Start simulated vapour compression searches from a Carnot-model placement."

Organic Rankine Cycle
~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Option", "Type", "Default", "Unit", "Limits", "Description"
   :widths: 22, 8, 12, 8, 10, 40

   "``ORC_N_STAGES``","number","``1``","",">= 1","Number of parallel ORC units, each evaporating at its own temperature."
   "``ORC_ETA_II_CARNOT``","number","``0.5``","","> 0 to 1","Fraction of the Carnot efficiency achieved by the Carnot-model ORC."
   "``ORC_DT_CONT``","number","``5.0``","delta_degC",">= 0","dt_cont of the ORC working fluid against the process heat source."
   "``ORC_T_COND``","number","``30.0``","degC","","Condensing temperature of the ORC."
   "``ORC_MIN_LIFT``","number","``10.0``","delta_degC",">= 0","Smallest difference between the ORC evaporating and condensing temperatures."
   "``ORC_LOAD_FRACTION``","number","``1.0``","","0 to 1","Largest fraction of the below-pinch heat surplus the ORC may take."
   "``ORC_MAX_MULTISTART``","number","``5``","",">= 1","Number of ORC optimiser runs from different starting points."
   "``ORC_BB_MINIMISER``","enum","``dual_annealing``","","dual_annealing, cmaes, bo, rbf_surrogate","Black-box optimiser for ORC targeting: dual_annealing, cmaes, bo or rbf_surrogate."
   "``ORC_FLUIDS``","string_list","``[""R1233zd(E)"", ""R1234ze(E)"", ""Isopentane"", ""n-Pentane"", ""Toluene""]``","","","Candidate working fluids for the simulated ORC."
   "``ORC_ETA_TURBINE``","number","``0.8``","","> 0 to 1","Isentropic efficiency of the simulated ORC turbine."
   "``ORC_ETA_PUMP``","number","``0.7``","","> 0 to 1","Isentropic efficiency of the simulated ORC feed pump."
   "``ORC_MAX_SUPERHEAT``","number","``20.0``","delta_degC",">= 0","Largest superheat the simulated ORC search may use."
   "``ORC_RECUPERATOR_ENABLED``","boolean","``False``","","","Add a recuperator to the simulated ORC."
   "``ORC_DT_RECUPERATOR``","number","``10.0``","delta_degC",">= 0","Minimum approach temperature of the ORC recuperator, when it is enabled."

Process MVR
~~~~~~~~~~~

.. csv-table::
   :header: "Option", "Type", "Default", "Unit", "Limits", "Description"
   :widths: 22, 8, 12, 8, 10, 40

   "``PROCESS_MVR_ETA_COMP``","number","``0.7``","","> 0 to 1","Isentropic efficiency of direct process MVR compressors, unless the call gives its own."
   "``PROCESS_MVR_ETA_MOTOR``","number","``0.95``","","> 0 to 1","Motor efficiency of direct process MVR compressors, unless the call gives its own."

Power and Cogeneration
~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Option", "Type", "Default", "Unit", "Limits", "Description"
   :widths: 22, 8, 12, 8, 10, 40

   "``POWER_TURBINE_WORK_ENABLED``","boolean","``False``","","","When on, a turbine inlet pressure above 220 bar is reset to 200 bar on loading. No other code reads it."
   "``POWER_TURB_T_IN``","number","``450.0``","degC","","Steam temperature at the turbine inlet for above-pinch cogeneration."
   "``POWER_TURB_P_IN``","number","``90.0``","bar",">= 0","Steam pressure at the turbine inlet for above-pinch cogeneration."
   "``POWER_MIN_EFF``","number","``0.1``","",">= 0","Lowest isentropic efficiency a turbine stage is given when its model predicts less."
   "``POWER_LOAD_FRACTION``","number","``1.0``","",">= 0","Operating flow as a fraction of turbine capacity, used by the efficiency models for part load."
   "``POWER_ETA_MECH``","number","``1.0``","","> 0","Mechanical efficiency applied to turbine work."
   "``POWER_HIGH_P_COND_FLASH_ENABLED``","boolean","``False``","","","Let condensate from higher-pressure steam levels flash at lower levels, reducing the steam they need."

