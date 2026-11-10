# Domain

## Overview
Core in-memory state of OpenPinch: unit-aware values, streams and stream collections, the problem table, zones and their solved targets, heat exchanger records, configuration and the shared enumerations. Modules here import only from each other and from third-party libraries; no module imports the analysis, application, contracts, adapters or presentation packages. Those packages build on this one, and `contracts/input.py` reuses the exchanger validation rules from `_heat_exchanger`. See `docs/api/domain-model.rst` and `docs/developer/architecture.rst`.

## Interface
- `Value` (`value.py`): scalar or multiperiod quantity on a Pint registry with unit conversion, arithmetic and `to_dict` / `from_dict`; `ureg` and `Q_` are the registry and quantity factory.
- `Stream` (`stream.py`): hot, cold or utility stream with supply and target states, derived thermodynamic values, period context and optional ordered segments.
- `StreamCollection` (`stream_collection.py`): ordered, keyed set of streams with lazy sorting, filtered subset getters, numeric views and `to_dict`.
- `ProblemTable` (`problem_table.py`): NumPy-backed table with label and enum column access, pinch row search, interval insertion and in-place `update`.
- `Zone` (`zone.py`): hierarchical analysis boundary owning stream collections, subzones, targets and graphs.
- Target models (`targets.py`): `BaseTargetModel`, `UtilitySummaryTarget`, `DirectIntegrationTarget`, `ResidualUtilityTarget`, `SubzoneAggregateTarget`, `IndirectIntegrationTarget`, `EnergyTransferTarget`, `HeatPumpTargetBase` with direct and indirect heat pump and refrigeration subclasses, `DirectOrcTarget`, and the `AnyTargetModel` union.
- `AnalysisProvenance` (`analysis.py`): frozen record of an analysis's owner, method, zone address, periods, input fingerprint, prerequisite ids and canonical settings JSON.
- `HeatExchanger` (`heat_exchanger.py`) and `HeatExchangerNetwork` (`heat_exchanger_network.py`): validated exchanger and network records, with period-aware totals and labelled lookups; `HeatExchangerNetwork.from_structure` builds a zero-duty structure.
- `Configuration` (`configuration.py`): flat options turned into typed frozen groups (`problem`, `input_units`, `output_units`, `thermal`, `costing`, `hens`, `hpr`, `orc`, `power` and others); `tol`, `C_to_K`, `DUTY_RTOL`, `T_CRIT`, `duty_tolerance`, `heat_resolution` are the numerical constants and helpers.
- `CustomCompressorModel` (`compressor_model.py`): a user compressor efficiency function with its name; `to_dict` gives `{"custom_compressor_model": name}`, the form provenance records.
- `CONFIG_FIELD_SPECS`, `USER_CONFIG_FIELD_SPECS`, `ConfigurationFieldSpec`, `validate_configuration_options`, `validate_internal_configuration_options`, `validate_configuration_option_value`, `input_unit_options_to_map`, `output_unit_options_to_map` (`configuration_fields.py`): declarative option table and validators.
- Enumerations (`enums.py`): `ZoneType`, `TargetType`, `IntegrationType`, `TargetMethod`, `StreamType`, `StreamID`, `StreamLoc`, `FluidPhase`, `HeatExchangerKind`, `HeatExchangerNetworkLabel`, `ProblemTableLabel`, `GraphType`, `TurbineModel`, `BB_Minimiser` and others, plus the `CogenerationTarget` protocol.
- `HPRLoadSummary`, `HPRResidualProfile`, `HPRThermalSlice`, `HPRResidualData`, `HPRResidualSnapshot` (`hpr.py`): frozen records handing HPR targeting results to residual utility studies.
- `validate_solver_option`, `validate_solver_options`, `ALLOWED_SOLVER_OPTION_NAMES` (`solver_options.py`): allow-listed HEN solver options.
- `build_coolprop_abstract_state`, `validate_coolprop_fluid_name` (`fluids.py`): parse single and mixture CoolProp fluid specs; `critical_point` is a cached helper, and `coolprop_fluid_spelling` returns a spec upper-cased, or as given when CoolProp rejects the upper-cased form (`HprConfig.normalised_refrigerants` uses it).
- `Progress`, `reporting`, `scope`, `emit` (`progress.py`): progress reports from long analyses through a context-variable reporter.
- `non_empty_str`, `finite_float` (`_validation.py`): reusable Pydantic validator factories.
- Subdirectories:
  - [_heat_exchanger](./_heat_exchanger/README.md): exchanger period state, area slices and shared validation rules.
  - [_heat_exchanger_network](./_heat_exchanger_network/README.md): a label key type alias, currently unused.
  - [_problem_table](./_problem_table/README.md): equality, interval insertion and update typing for `ProblemTable`.
  - [_stream](./_stream/README.md): stream value state, thermodynamics, segments and linearisation.
  - [_stream_collection](./_stream_collection/README.md): subset, numeric view, serialisation and sort helpers.
  - [_value](./_value/README.md): value coercion, resolution and unit-text handling.

## Invariants
- Domain code does not import from the analysis, application, contracts, adapters or presentation packages. [enums.py:CogenerationTarget]
- Every quantity carries a unit; `Stream` stores its values in fixed units (degC, kPa, kJ/kg, kW, `delta_degC`, `$/MW/h`) and converts on assignment. [stream.py:Stream]
- `deepcopy` of a `Stream` or `StreamCollection` copies the stored state and re-derives nothing. Copied values share their read-only magnitudes; a copied collection drops its numeric views and sorts again on first use. [stream.py:Stream.__deepcopy__] [stream_collection.py:StreamCollection.__deepcopy__]
- Values owned by a stream are read-only; changes go through the stream setters, and for a segmented stream the aggregate temperatures, heat flow and HTC are derived, so edit segments instead. [stream.py:Stream._read_only_value]
- Retired short attribute names on `Stream` (for example `t_supply`, `htc`) raise `AttributeError` on assignment. [stream.py:Stream.__setattr__]
- Configuration group dataclasses must match `CONFIG_FIELD_SPECS` field for field, checked at import. [configuration.py:_check_group_types]
- Configuration option keys are validated against `CONFIG_FIELD_SPECS`; unknown keys are rejected, and `HPR_TYPE`, `ORC_MODEL` and `POWER_TURB_MODEL` are accepted only by the internal validator. [configuration_fields.py:validate_configuration_options] [configuration_fields.py:validate_internal_configuration_options]
- `HPR_COMPRESSOR_MODEL` and `HPR_MVR_COMPRESSOR_MODEL` take a model name; only the internal validator accepts a function, wrapped in `CustomCompressorModel`, and the user validator rejects one with a pointer to the per-call argument. [configuration_fields.py:_validate_configuration_options] [configuration_fields.py:_compressor_model]
- Period ids are unique and non-empty, and the weight list is empty (equal weights), has one value, or has one per period. [configuration_fields.py:_validate_period_options]
- `tol` is 1e-6 and the temperature rounding in the problem table derives from it. [configuration.py:tol] [_problem_table/intervals.py:_TEMPERATURE_DECIMAL_PLACES]
- Adding a target to a zone removes stored targets that depend on it, directly or transitively, unless `invalidate_dependents` is false. [zone.py:Zone.add_target]
- Target models derive `integration_type` and `target_method` from their class or target type and forbid extra fields. [targets.py:BaseTargetModel]
- A collection and its streams share one period context; adding a stream with an incompatible period count raises. [stream_collection.py:StreamCollection._validate_stream_period_context]
- Exchanger and network records are validated Pydantic models; metadata fields `solver_metadata`, `source_metadata` and `solver_axis_metadata` are excluded from serialisation. [heat_exchanger.py:HeatExchanger] [heat_exchanger_network.py:HeatExchangerNetwork]
- HEN solver option names must be on the allow-list, with number, boolean or single-word values of at most 32 characters. [solver_options.py:validate_solver_option]
- `AnalysisProvenance` is frozen and stores canonical sorted-key settings JSON without NaN. [analysis.py:AnalysisProvenance]
- HPR residual profiles require aligned grids of at least two points with strictly decreasing temperatures and `net`, `heating` and `cooling` values no lower than -1e-7. [hpr.py:HPRResidualProfile]
- Progress reporting is a no-op outside `reporting`, and an exception from the callback propagates out of the analysis; reports of one stage closer together than `min_interval` (0.25 s) are dropped except the first and the finishing one. [progress.py:emit] [progress.py:reporting]

## Side Effects
- Importing `value.py` creates a Pint `UnitRegistry`, defines `USD` and `NZD`, installs the unit-text grammar check as a preprocessor and sets it as the application registry. [value.py:ureg]
- `fluids.py` imports `CoolProp` lazily, and `critical_point` and `coolprop_fluid_spelling` cache results with `functools.cache`. [fluids.py:build_coolprop_abstract_state] [fluids.py:critical_point] [fluids.py:coolprop_fluid_spelling]
- `Stream` validates `fluid_name` by constructing a CoolProp state. [stream.py:Stream._normalise_fluid_name]
- `StreamCollection.add` appends `_n` to the stream's name and key on a key collision (unless `prevent_overwrite=False`), applies the collection's period context to the stream and may widen the collection's context to match a multiperiod stream, and `remove`, `get_stream_by_name` and similar methods emit `warnings.warn` for missing names. [stream_collection.py:StreamCollection.add]
- `StreamCollection` caches numeric views and, for value-attribute sort keys, the sort order, keyed on stream revision counters; callable or metadata sort keys are re-sorted each time. [stream_collection.py:StreamCollection.numeric_view]
- `Zone.dt_cont_multiplier` assignment cascades to every subzone not locked with `lock_dt_cont_multiplier`, sets the multiplier on their streams, and clears the targets of each zone it reaches. [zone.py:Zone.dt_cont_multiplier]
- `Configuration.update_values` mutates the flat values and rebuilds the groups; it raises `KeyError` for unknown keys and does not revalidate values, so callers validate first (for example `validate_internal_configuration_options`). [configuration.py:Configuration.update_values]
- `progress.py` holds context variables for the reporter and scope, and checks `os.getpid()` so forked workers do not report. [progress.py:_REPORTER] [progress.py:_Reporter]
- `ProblemTable` methods such as `insert`, `delete_row`, `round`, `update` and `sort_by_column` mutate the table in place; `update` on a different temperature grid inserts rows, so the row count can change. [problem_table.py:ProblemTable]
- Module-level flags `ACTIVATE_TIMING` and `LOG_TIMING` are defined in `configuration.py`. [configuration.py:ACTIVATE_TIMING]

## Verification
- Run: `uv run --no-sync pytest tests/domain`
