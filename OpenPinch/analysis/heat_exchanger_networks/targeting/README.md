# HEN Targeting

## Overview
Orchestrates the HEN synthesis methods: it builds the task graph for pinch design (PDM), thermal derivative (TDM) and network evolution (ESM), runs each stage through an executor, and assembles the workflow result. `execute_open_hens_method` chains the stages; the other modules also support standalone seeded runs. Canonical topology helpers here are shared with `../reporting`.

## Interface
- `execute_open_hens_method` (`open_hens_method.py`): runs PDM, then TDM and ESM as the tier requires, and returns a `SynthesisWorkflowResult`.
- `build_pinch_design_method_tasks`, `execute_pinch_design_method_stage` (`pinch_design_method.py`): root PDM tasks from tier pathways and quality candidates.
- `build_thermal_derivative_method_tasks`, `build_seeded_thermal_derivative_method_tasks`, `execute_thermal_derivative_method_stage`, `execute_seeded_thermal_derivative_method_stage` (`thermal_derivative_method.py`): TDM tasks from PDM outcomes or seed networks.
- `build_network_evolution_method_tasks`, `build_seeded_network_evolution_method_tasks`, `build_network_evolution_method_tasks_from_pinch_design_method`, `build_direct_network_evolution_method_tasks` and the matching `execute_*_stage` functions (`network_evolution_method.py`): ESM tasks from TDM outcomes, PDM outcomes, seeds, or none.
- `canonical_topology_restrictions`, `canonical_stage_count`, `topology_restriction_signature` (`topology.py`): canonical grid-stage form and duty-independent signature of recovery matches.
- `run_stage`, `run_single_method_workflow`, `build_seeded_quality_tasks` (`_stage.py`): shared stage runner, single-method workflow and seeded-task builder.
- Underscore-prefixed `_execute_*_method_workflow` functions are the standalone entry points used by `../service.py`.

## Invariants
- Only successful tasks of the expected upstream method spawn downstream tasks, and utility-only outcomes are skipped as final candidates [thermal_derivative_method.py:build_thermal_derivative_method_tasks, network_evolution_method.py:build_network_evolution_method_tasks].
- Tier 0 skips TDM; tiers above 1 also run ESM directly from PDM in addition to the TDM route [open_hens_method.py:execute_open_hens_method].
- When Couenne is missing for every PDM (or TDM) task, ESM runs without them and a warning is emitted [open_hens_method.py:execute_open_hens_method].
- Tier 0 pathways run every configured approach temperature; expanded tiers run multiples of the first one [pinch_design_method.py:_pathway_approach_temperatures].
- Stage-pair PDM candidates come from a fixed ordered sweep truncated to the requested count [pinch_design_method.py:_PDM_STAGE_PAIR_SWEEP].
- Canonical restrictions drop empty stage gaps, keep matches sharing a hot or cold stream in one stage, and split independent groups into consecutive stages [topology.py:canonical_topology_restrictions].
- Seeded quality tasks are de-duplicated on approach temperature and canonical topology, plus derivative threshold when thresholds are distinct [_stage.py:build_seeded_quality_tasks].
- Single-method workflows replace `method_sequence` and `design_method` in the settings [_stage.py:run_single_method_workflow].

## Side Effects
- Constructs a default `LocalSynthesisExecutor` (solver processes) when no executor is supplied [_stage.py:run_stage].
- Emits a `RuntimeWarning` for the Couenne fallback [open_hens_method.py:execute_open_hens_method].
- Times workflows with `perf_counter`; no file IO.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_topology.py tests/analysis/heat_exchanger_networks/test_pinch_design_method.py tests/analysis/heat_exchanger_networks/test_thermal_derivative_method.py tests/analysis/heat_exchanger_networks/test_network_evolution_method.py`
