# Contracts

## Overview
Pydantic input and output contracts at the OpenPinch application boundary: the
case input, targeting results and reports, workspace bundles, heat pump and
refrigeration (HPR) records, HPR performance maps, utility placement, heat
recovery dt_min inversion, controllability and turbine results. It also holds
the unit rules that convert values on entry and exit, and the JSON Schema
generator for typed clients. The package depends on `OpenPinch/domain`; the
application layer builds on these models.

## Interface
- `ValueWithUnit`, `PeriodValueWithUnit`, `PeriodValueWithUnitAndWeights`, `PeriodValueWithUnitAndIds`, `ScalarOrVU`, `MaybeVU`, `HPRMetric` (`common.py`): value-with-unit containers and aliases.
- `TargetInput`, `StreamSchema`, `UtilitySchema`, `StreamSegmentSchema`, `TemperatureHeatProfileSchema`, `ZoneTreeSchema`, `HeatExchangerSchema`, `HeatExchangerNetworkSchema`, `PlantProfileSchema`, `ProcessComponentSchema`, `ProcessComponentSourceSchema`, `NonLinearStream` and related (`input.py`): validated case input.
- `TargetOutput` (`output.py`): cached targeting output with targets, graphs and optional design result.
- `TargetResults`, `HeatUtility`, `PinchTemp`, `ReportMetric`, `GraphAvailability`, `ProblemReport` (`reporting.py`): report-facing results.
- `report_field`, `metric_specifications`, `AggregationPolicy`, `MetricSpecification` (`report_metrics.py`): per-field period aggregation policy metadata.
- `INPUT_UNIT_RULES`, `OUTPUT_UNIT_RULES`, `standardise_input_value`, `coerce_output_value`, `heat_flow_from_heat_capacity_flowrate` (`units.py`): canonical and display unit rules and coercion.
- `split_report_value` (`report_units.py`): splits scalar, multiperiod or array report data into value and unit.
- `GraphSet`, `Graph`, `Segment`, `DataPoint` (`graphs.py`): plot data.
- `ValidationReport`, `ValidationIssue`, `ConfigurationFieldMetadata`, `PinchWorkspaceBundle`, `WorkspaceCaseBundleEntry`, `validate_workspace_case_name` (`workspace.py`): validation reports and persisted workspaces.
- HPR (`hpr.py`): search budget, failure diagnostics and `HPRTargetingError`, `HprTargetSimulationRecord` with loop and stage records, `HeatPumpTargetInputs`, `HeatPumpTargetOutputs`, `MultiPeriodHPRTargetInputs`, `HPRBackendResult`, `ZERO_USEFUL_DUTY_REASON`, and internal state types.
- `HprPerformanceMap`, `HprPerformanceMapRequest`, `HprPerformanceMapUnits`, `HprPerformancePoint`, `JsonValue` (`hpr_performance_map.py`): versioned physical performance map.
- Utility placement (`utility_placement.py`): `UtilityPlacementRequest`, `UtilityPlacementOptions`, `UtilityPlacementResult`, `UtilityPlacementModel`, templates, envelopes, candidates and enums.
- `HeatRecoveryDtMinResult`, `HeatRecoveryDtMinStatus`, `HeatRecoveryQuantity` (`heat_recovery_dt_min.py`): inverse heat-recovery result.
- `HprRefrigerantScreenResult`, `RefrigerantScreenRow`, `RefrigerantScreenQuantity`, `RefrigerantScreenVerdict` (`hpr_refrigerant_screen.py`): refrigerant screening result, one row per fluid and level pair, with verdicts in the order they are judged; `practical` returns the practical rows.
- `HprLoadKinksResult`, `HprLoadKink`, `HprLoadKinkKind`, `LOGICAL_LOAD_KINDS` (`hpr_load_kinks.py`): vertices of a heat pump sink or refrigeration source net load profile and the logical HPR load fractions; quantities reuse `HeatRecoveryQuantity`.
- `HeatExchangerNetworkControllabilityResult` and its endpoint, actuator, pairing and component models (`controllability.py`).
- `TurbineSolveResult`, `TurbineStageResult` (`turbine.py`): turbine solve output, raw floats in fixed units (kW, degC, bar, kJ/kg, kg/s); reported per period as `TargetResults.turbine_result`.
- `CONTRACTS`, `VALUE_JSON_SCHEMA`, `contract_json_schemas` (`schema.py`, in `__all__`) and `STREAM_COLLECTION_JSON_SCHEMA` (module-level, not in `__all__`): JSON Schema for the public contracts.
- Subdirectories: [synthesis](./synthesis/README.md) - HEN synthesis task, result and design-view models; [resources](./resources/README.md) - packaged JSON data, with the HPR performance-map schema and fixtures under it.

## Invariants
- Case input, synthesis, workspace bundle, value-with-unit, HPR output and record, and specialist result models set `extra="forbid"`, so unknown keys are rejected. The HPR solver bundles (`HeatPumpTargetInputs`, `HPRPeriodCase`, `MultiPeriodHPRTargetInputs`, `HPRBackendResult`, `HPRParsedState`, `HPRThermoArtifacts`) and `TurbineStageResult` ignore unknown keys, so a misspelled optional keyword is dropped silently. The top-level report models (`TargetOutput`, `TargetResults`, graphs, `ValidationReport`) ignore unknown keys, but records nested in them (`design`, `provenance`, `hpr_target_simulation_record`) still reject them [input.py:TargetInput] [common.py:ValueWithUnit] [reporting.py:TargetResults] [hpr.py:HeatPumpTargetInputs] [hpr.py:HeatPumpTargetOutputs] [turbine.py:TurbineStageResult].
- `HeatPumpTargetInputs.eta_comp` and `eta_mvr_comp` are a constant isentropic efficiency or a function of the compressor conditions; `eta_motor` (`HPR_ETA_MOTOR`) is the vapour compression motor efficiency and `eta_mvr_motor` (`HPR_MVR_ETA_MOTOR`) the MVR one [hpr.py:HeatPumpTargetInputs].
- `PeriodValueWithUnitAndIds` requires non-empty, unique period ids aligned with `values` [common.py:PeriodValueWithUnitAndIds].
- Every `TargetResults` field carries an aggregation policy; `metric_specifications` raises for a field without one [report_metrics.py:metric_specifications].
- `coerce_output_value` converts a report metric to the configured display unit (default unit without a config) and rescales per-year units by `COSTING_ANNUAL_OP_TIME` hours when source and target year exponents differ [units.py:coerce_output_value]. Report models keep any `Value` and any mapping or object with a unit as given, and coerce plain numbers and unit-less mappings to default display units; configured conversion happens in the callers [reporting.py:_report_value_data].
- Input values are converted to canonical units (degC, kPa, kJ/kg, kW, kW/delta_degC, delta_degC, kW/m^2/delta_degC, $/MWh); `dt_cont` given in degC or K is read as a temperature difference [units.py:INPUT_UNIT_RULES, standardise_input_value].
- A stream has either `segments`, a `profile`, or ordinary thermal data (`t_supply`, `t_target` and `heat_flow` or `heat_capacity_flowrate`); `segments` and `profile` are mutually exclusive and `heat_capacity_flowrate` cannot accompany either. Parent `t_supply`/`t_target`/`heat_flow` may accompany nested data; `StreamSchema` does not compare them, the match is checked when the application prepares the problem [application/_problem/input/segments.py:_validate_supplied_parent_aggregates] [input.py:StreamSchema].
- Sibling zone names are unique [input.py:ZoneTreeSchema].
- `TargetInput.process_components` is left out of dumps when empty, so input without components dumps as it did before the field existed [input.py:TargetInput].
- A residual-basis input cannot contain streams, networks, plant profiles or process components, must use its single frozen period, cannot enable shared-period HPR, and allows only a childless Process Zone with no temperature-shift multiplier other than 1.0 [input.py:TargetInput._validate_residual_basis].
- Exchanger period states, direction semantics and segment area totals are checked; `area` must match the largest period total of segment areas [input.py:HeatExchangerSchema].
- Workspace bundles accept only `schema_version` `"3"`; case names are portable (non-empty, no path characters, control characters, leading or trailing whitespace, trailing period or Windows device names, and not `.` or `..`) [workspace.py:PinchWorkspaceBundle, validate_workspace_case_name].
- A performance map must be in canonical point order, satisfy `q_sink = q_source + motor_efficiency * electric_power` within tolerance (`motor_efficiency` defaults to 1.0, shaft power), have useful duty equal to `load_fraction * reference_capacity`, and use basis `q_sink` for heat pumps and `q_source` for refrigeration [hpr_performance_map.py:HprPerformanceMap].
- `HprTargetSimulationRecord.topology_id` must equal `cycle_id`; every topology other than single-stage vapour compression needs loop evidence; the condensing temperature exceeds the evaporating temperature [hpr.py:HprTargetSimulationRecord].
- `RefrigerantScreenRow` requires a positive `rank` on practical rows and none on others, and requires the saturation pressures, pressure ratio and suction capacity unless the verdict is above critical or below triple point, when it forbids them [hpr_refrigerant_screen.py:RefrigerantScreenRow].
- `HeatRecoveryDtMinResult` checks that requested and achieved recovery do not exceed the thermodynamic limit, that the residual equals achieved minus requested, and that `status` agrees with the values [heat_recovery_dt_min.py:HeatRecoveryDtMinResult].
- `HprLoadKinksResult` requires `steepening > 1`, a positive `maximum_duty`, vertex load fractions strictly increasing inside (0, 1), and `logical_load_fractions` equal to the step and steepens vertices followed by 1.0; a vertex has no `slope_after` exactly when it is a step [hpr_load_kinks.py:HprLoadKinksResult] [hpr_load_kinks.py:HprLoadKink].
- Output metrics `heat_recovery_dt_min` and `hpr_load_kink_slope` are temperature differences, so a temperature display override resolves to its delta unit [units.py:coerce_output_value].
- Controllability scores and interaction entries lie in 0..1 [controllability.py:_validate_score].
- Utility-placement values are frozen and finite, and period ids are unique and aligned with weights [utility_placement.py:_FrozenContract, UtilityPlacementResult].
- Inputs use the validation JSON schema and results the serialisation schema [schema.py:CONTRACTS].

## Side Effects
- `TargetInput` rewrites `options` for a residual-basis input, adding `PROBLEM_PERIOD_IDS` and `PROBLEM_PERIOD_WEIGHTS` defaults [input.py:TargetInput._validate_residual_basis].
- `HeatExchangerSchema` sets `area` from segment contributions with `object.__setattr__` [input.py:HeatExchangerSchema._validate_area_contributions].
- `HprPerformanceMap` returns a deep copy of `provenance` on every access [hpr_performance_map.py:HprPerformanceMap.__getattribute__].
- `HPRTargetingError` defines `__reduce__` so it pickles with its diagnostics [hpr.py:HPRTargetingError].
- `PlacementUnitSystem` imports `Value` lazily and checks by trial conversion that the temperature units convert to `degC` and `delta_degC`; unit strings are kept as given [utility_placement.py:_require_convertible].
- Report models serialise `Value` and `StreamCollection` fields to dictionaries in JSON mode [reporting.py:_ReportModel].
- No file or network IO in this directory.

## Verification
- Run: `uv run --no-sync pytest tests/contracts`
