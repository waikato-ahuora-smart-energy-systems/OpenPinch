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
- `TargetInput`, `StreamSchema`, `UtilitySchema`, `StreamSegmentSchema`, `TemperatureHeatProfileSchema`, `ZoneTreeSchema`, `HeatExchangerSchema`, `HeatExchangerNetworkSchema`, `PlantProfileSchema`, `NonLinearStream` and related (`input.py`): validated case input.
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
- `HeatExchangerNetworkControllabilityResult` and its endpoint, actuator, pairing and component models (`controllability.py`).
- `TurbineSolveResult`, `TurbineStageResult` (`turbine.py`): turbine solve output.
- `CONTRACTS`, `VALUE_JSON_SCHEMA`, `STREAM_COLLECTION_JSON_SCHEMA`, `contract_json_schemas` (`schema.py`): JSON Schema for the public contracts.
- Subdirectories: [synthesis](./synthesis/README.md) - HEN synthesis task, result and design-view models; [resources](./resources/README.md) - packaged JSON data, with the HPR performance-map schema and fixtures under it.

## Invariants
- Most models set `extra="forbid"`, so unknown keys are rejected [input.py:TargetInput, common.py:ValueWithUnit].
- `PeriodValueWithUnitAndIds` requires non-empty, unique period ids aligned with `values` [common.py:PeriodValueWithUnitAndIds].
- Every `TargetResults` field carries an aggregation policy; `metric_specifications` raises for a field without one [report_metrics.py:metric_specifications].
- Report values are converted to configured display units; annual-cost units are rescaled by `COSTING_ANNUAL_OP_TIME` rather than the calendar year [units.py:coerce_output_value].
- Input values are converted to canonical units (degC, kPa, kJ/kg, kW, kW/delta_degC, delta_degC, kW/m^2/delta_degC, $/MWh); `dt_cont` given in degC or K is read as a temperature difference [units.py:INPUT_UNIT_RULES, standardise_input_value].
- A stream has either `segments`, a `profile`, or ordinary thermal data (`t_supply`, `t_target` and `heat_flow` or `heat_capacity_flowrate`); `segments` and `profile` are mutually exclusive and `heat_capacity_flowrate` cannot accompany either. Parent `t_supply`/`t_target`/`heat_flow` may accompany nested data and are checked against it at input validation [input.py:StreamSchema].
- Sibling zone names are unique [input.py:ZoneTreeSchema].
- A residual-basis input cannot contain streams, networks or plant profiles and must use its single frozen period [input.py:TargetInput].
- Exchanger period states, direction semantics and segment area totals are checked; `area` must match the largest period total of segment areas [input.py:HeatExchangerSchema].
- Workspace bundles accept only `schema_version` `"3"`; case names are portable (no path characters, control characters, trailing period or Windows device names) [workspace.py:PinchWorkspaceBundle, validate_workspace_case_name].
- A performance map must be in canonical point order, satisfy `q_sink = q_source + electric_power` within tolerance, have useful duty equal to `load_fraction * reference_capacity`, and use basis `q_sink` for heat pumps and `q_source` for refrigeration [hpr_performance_map.py:HprPerformanceMap].
- `HprTargetSimulationRecord.topology_id` must equal `cycle_id`; multi-loop topologies need loop evidence; the condensing temperature exceeds the evaporating temperature [hpr.py:HprTargetSimulationRecord].
- `HeatRecoveryDtMinResult` checks that requested and achieved recovery do not exceed the thermodynamic limit, that the residual equals achieved minus requested, and that `status` agrees with the values [heat_recovery_dt_min.py:HeatRecoveryDtMinResult].
- Controllability scores and interaction entries lie in 0..1 [controllability.py:_validate_score].
- Utility-placement values are frozen and finite, and period ids are unique and aligned with weights [utility_placement.py:_FrozenContract, UtilityPlacementResult].
- Inputs use the validation JSON schema and results the serialisation schema [schema.py:CONTRACTS].

## Side Effects
- `TargetInput` rewrites `options` for a residual-basis input, adding `PROBLEM_PERIOD_IDS` and `PROBLEM_PERIOD_WEIGHTS` defaults [input.py:TargetInput._validate_residual_basis].
- `HeatExchangerSchema` sets `area` from segment contributions with `object.__setattr__` [input.py:HeatExchangerSchema._validate_area_contributions].
- `HprPerformanceMap` returns a deep copy of `provenance` on every access [hpr_performance_map.py:HprPerformanceMap.__getattribute__].
- `HPRTargetingError` defines `__reduce__` so it pickles with its diagnostics [hpr.py:HPRTargetingError].
- `PlacementUnitSystem` imports `Value` lazily and converts units while validating [utility_placement.py:_require_convertible].
- Report models serialise `Value` and `StreamCollection` fields to dictionaries in JSON mode [reporting.py:_ReportModel].
- No file or network IO in this directory.

## Verification
- Run: `uv run --no-sync pytest tests/contracts`
