# HPR Performance Maps and Targeting Capability

## Overview
Engine-neutral machinery for two jobs. First, generate a complete heat pump or refrigeration performance map (source, sink and load grid) for a single-stage vapour-compression target by driving a point simulator, validating each point and recording provenance. Second, support HPR targeting: CoolProp and TESPy capability preflights, a validated candidate-evaluator lifecycle with a cache, and detached simulation records attached to winning targets. Concrete engines live in [`adapters`](./adapters/README.md). `../targeting` and `../_multiperiod` import the preflight and record functions.

## Interface
Names re-exported by `__init__.py`:
- `generate_hpr_performance_map` (`generation.py`): produce one `HprPerformanceMap` or raise `HprMapGenerationError` with ordered diagnostics.
- `build_hpr_map_generation_context` (`context.py`): validate a basis and request into an `HprMapGenerationContext`.
- `build_hpr_target_map_basis`, `HprPerformanceMapCompatibilityError` (`target_basis.py`): detach a map basis from a successful target or reject it with a code.
- `get_hpr_point_simulator` (`factory.py`): create a fresh `coolprop` or `tespy` simulator.
- `iter_hpr_operating_points` (`points.py`): lazily expand the source, sink and load product.
- `parse_hpr_working_fluid`, `resolve_hpr_working_fluid`, `to_tespy_fluid_token` (`fluids.py`): fluid string parsing and capability proof.
- `HprPointSimulator`, `HprPointSimulatorFactory` (`protocols.py`); value classes `HprTargetMapBasis`, `HprWorkingFluidSpec`, `HprMapGenerationContext`, `HprOperatingPoint`, `HprPointSimulation`, `HprSimulatorMetadata`, `HprSimulationDiagnostic` (`models.py`); `HprMapGenerationError` (`errors.py`).

Used by sibling packages but not re-exported:
- `coolprop_preflight.py`: `preflight_coolprop_fluid_names`, `preflight_coolprop_hpr_targeting`, `representative_hpr_point`, `PreparedCoolPropTargeting`.
- `targeting.py`: `preflight_tespy_hpr_targeting`, `normalize_hpr_simulation_backend`, `get_hpr_target_evaluator`, `HprTargetEvaluatorCoordinator`, `validate_hpr_target_result`, `PreparedTespyHprTargeting`, `HprTargetCompatibilityError`, `HprTargetDependencyError` (also an `ImportError`, for a missing `tespy`), `HprTargetValidationTolerances`, `HprTargetCacheStats`.
- `targeting_models.py`: request, result, metadata, failure, error and protocol types for target evaluation, and `HprThermalProfilePoint`.
- `target_records.py`: `build_coolprop_target_simulation_record`, `build_tespy_target_simulation_record`. With a compressor efficiency model the CoolProp record gives each stage the efficiency its cycle applied, the work-weighted mean as `compressor_isentropic_efficiency`, and the model's name as the `compressor_model` assumption; with constant efficiencies the record is unchanged. Each CoolProp loop stage carries its motor efficiency: `eta_mvr_motor` for MVR stages and `eta_motor` for vapour compression stages; its `compressor_work` is electrical.
- `provenance.py`: `build_hpr_map_provenance`. `settings.py`: `TESPY_CONVERGENCE_SETTINGS`. `resources.py`: compressor characteristic loaders. `errors.py`: `HprSimulatorFailure`, `diagnostic_from_failure`, `sanitize_details`.
- Subdirectories: [`adapters`](./adapters/README.md) holds the CoolProp and TESPy simulators and the TESPy evaluator; [`characteristics`](./characteristics/README.md) holds the packaged compressor characteristic JSON.

## Invariants
- Map generation is atomic: after context construction any diagnostic raises `HprMapGenerationError` and no partial map is returned; invalid basis or request values raise plain `ValueError` from `build_hpr_map_generation_context` first [generation.py:generate_hpr_performance_map] [context.py:build_hpr_map_generation_context].
- Only the `single_stage_vapour_compression` cycle and the `coolprop` and `tespy` backends are supported; REFPROP is rejected [context.py:build_hpr_map_generation_context, fluids.py:resolve_hpr_working_fluid].
- Nominal temperatures must exceed absolute zero with positive lift; approaches, superheat, subcooling and internal-heat-exchanger change must be non-negative; efficiency lies in (0, 1] [context.py:build_hpr_map_generation_context].
- Map temperatures are degC and duties and power kW; points close the energy balance `q_sink = q_source + power` on compressor shaft power and match the requested useful duty; point `electric_power` is that power divided by the request's `motor_efficiency` (default 1.0) [generation.py:_normalize_point, generation.py:generate_hpr_performance_map].
- The reference capacity basis is `q_sink` for heat pumps and `q_source` for refrigeration [context.py:build_hpr_map_generation_context].
- Operating point order is source, then sink, then load, with a running ordinal; evaporating temperature is source minus approach and condensing is sink plus approach [points.py:iter_hpr_operating_points].
- After a session-fatal failure remaining points are reported as `session_unavailable` [generation.py:generate_hpr_performance_map].
- A target can define a map only if it is a successful single-period heat pump or refrigeration target with a one-evaporator, one-condenser single-stage record that agrees with its scalar details, and whose record names no `compressor_model` assumption (`compressor_model`), since a map holds the compressor efficiency constant [target_basis.py:build_hpr_target_map_basis].
- Diagnostics and error details are bounded: messages 240 characters, at most 20 entries, strings 160 characters [errors.py:sanitize_details, errors.py:HprSimulatorFailure].
- Value dataclasses are frozen [models.py, targeting_models.py].
- Candidate cache holds at most 512 entries, evicting the least recently used entry (the cache stores results and candidate-local failures), and is cleared on close; the coordinator moves through created, ready, evaluating, closing, closed, fatal and cleanup_failed states [targeting.py:HprTargetEvaluatorCoordinator].
- Candidate acceptance checks backend, model, fluid, convergence, finiteness, energy balance, useful duty, COP and profile ordering with the named tolerances [targeting.py:validate_hpr_target_result, targeting.py:HprTargetValidationTolerances].
- TESPy targeting is limited to one evaporator and one condenser, no integrated expander, no internal heat exchanger, a constant compressor efficiency (`unsupported_compressor_model` for a model), rejects multiperiod inputs, rejects any `hpr_type` other than `cascade_vapour_comp` (`unsupported_cycle`), and requires `tespy` to be importable [targeting.py:preflight_tespy_hpr_targeting].
- An unsupported seed state in the CoolProp preflight marks the stage unsupported but does not fail the search; an unsupported fluid identity does [coolprop_preflight.py:preflight_coolprop_hpr_targeting, coolprop_preflight.py:preflight_coolprop_fluid_names].

## Side Effects
- CoolProp property and abstract-state calls; the predefined-mixture list is cached with `lru_cache` [fluids.py:_registered_mixture_names].
- `find_spec("tespy")` checks the optional extra; `adapters.tespy` is imported lazily and a missing extra becomes a typed failure [targeting.py:preflight_tespy_hpr_targeting, factory.py:get_hpr_point_simulator, targeting.py:get_hpr_target_evaluator].
- Package versions are read through `importlib.metadata` [provenance.py:_package_version, target_records.py:_distribution_version].
- The compressor characteristic is read from package data [resources.py:read_tespy_compressor_characteristic_bytes].
- With the TESPy backend each point simulator writes and reads a private temporary directory, removed on close [adapters/tespy.py:TespyHprPointSimulator.prepare].
- Simulator sessions are opened and closed by the generator and the coordinator [generation.py:generate_hpr_performance_map, targeting.py:HprTargetEvaluatorCoordinator].

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_pumps/test_hpr_map_generation.py tests/analysis/heat_pumps/test_hpr_target_basis.py tests/analysis/heat_pumps/test_hpr_targeting_preflight.py tests/analysis/heat_pumps/test_hpr_target_evaluator_contracts.py`
