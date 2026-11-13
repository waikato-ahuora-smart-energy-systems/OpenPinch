# HEN Targeting

## Overview
Orchestrates the HEN synthesis methods: it builds the task graph for pinch design (PDM), thermal derivative (TDM) and network evolution (ESM), runs each stage through an executor, and assembles the workflow result. `execute_open_hens_method` chains the stages; the other modules also support standalone seeded runs. Canonical topology helpers here are shared with `../reporting`.

## Interface
- `execute_open_hens_method` (`open_hens_method.py`): runs the pinch design stages, then TDM and ESM as the search depth requires, and returns a `SynthesisWorkflowResult`.
- `build_pinch_design_method_tasks`, `execute_pinch_design_method_stage` (`pinch_design_method.py`): one MILP PDM task per dt_cont multiplier, plus the `HENS_PDM_STAGE_PAIR_LIMIT` stage-pair candidates at the first multiplier.
- `build_heuristic_pinch_design_method_tasks`, `execute_heuristic_pinch_design_method_stage` (`heuristic_pinch_design_method.py`): one heuristic pinch design task per configured multiplier, carrying the network cap.
- `build_thermal_derivative_method_tasks`, `build_seeded_thermal_derivative_method_tasks`, `execute_thermal_derivative_method_stage`, `execute_seeded_thermal_derivative_method_stage` (`thermal_derivative_method.py`): TDM tasks from PDM outcomes or seed networks.
- `build_network_evolution_method_tasks`, `build_seeded_network_evolution_method_tasks`, `build_network_evolution_method_tasks_from_pinch_design_method`, `build_direct_network_evolution_method_tasks` and the matching `execute_*_stage` functions (`network_evolution_method.py`): ESM tasks from TDM outcomes, PDM outcomes, seeds, or none.
- `canonical_topology_restrictions`, `canonical_stage_count`, `topology_restriction_signature` (`topology.py`): canonical grid-stage form and duty-independent signature of recovery matches.
- `run_stage`, `run_single_method_workflow`, `build_seeded_quality_tasks` (`_stage.py`): shared stage runner, single-method workflow and seeded-task builder.
- Underscore-prefixed `_execute_*_method_workflow` functions are the standalone entry points used by `../service.py`.

## Invariants
- Only successful tasks of the expected upstream method spawn downstream tasks, and utility-only outcomes are skipped as final candidates [thermal_derivative_method.py:build_thermal_derivative_method_tasks, network_evolution_method.py:build_network_evolution_method_tasks].
- Search depth 0 runs no ESM; depth 1 evolves the pinch design networks directly and runs no TDM; from depth 2 the main MILP networks go through TDM, and from depth 3 every MILP network also seeds ESM directly [open_hens_method.py:execute_open_hens_method].
- The extra minimum-unit networks of a PDM task (outcomes whose task settings carry `pdm_variant`) never go to TDM; they seed ESM directly [thermal_derivative_method.py:build_thermal_derivative_method_tasks] [open_hens_method.py:execute_open_hens_method].
- `execute_pinch_design_stages` (`pinch_design_method.py`) is the pinch design step of both the standalone PDM workflow and `execute_open_hens_method`: the MILP stage when `runs_milp_pinch_design` (`pdm_milp`, or no heuristic networks), then the heuristic stage when `heuristic_pdm_networks` is above 0. Heuristic outcomes join the parent outcomes, seed ESM directly from depth 1 and in every Couenne fallback, and never go to TDM. Each task carries `heuristic_networks`, the configured cap [pinch_design_method.py:execute_pinch_design_stages] [open_hens_method.py:execute_open_hens_method] [network_evolution_method.py:build_network_evolution_method_tasks_from_pinch_design_method] [heuristic_pinch_design_method.py:build_heuristic_pinch_design_method_tasks].
- `run_stage` returns the stage's tasks followed by the tasks the executor derived for extra outcomes, so every outcome's task is in the returned tasks [_stage.py:run_stage].
- When Couenne is missing for every PDM task, ESM runs directly; when it is missing for every TDM task, ESM runs from the MILP outcomes not already seeded; a warning is emitted in both cases [open_hens_method.py:execute_open_hens_method].
- Evolution tasks built from outcomes or directly carry the branch breadth of the run, and two idle rounds of patience when it is above 1, so they agree with the manifest; seeded tasks leave breadth to the configuration [network_evolution_method.py:_run_breadth_settings].
- Tasks carry a dt_cont multiplier, never an approach temperature. The multipliers are the search width's unless `HENS_DT_CONT_MULTIPLIERS` is set [pinch_design_method.py:build_pinch_design_method_tasks].
- Stage-pair PDM candidates come from a fixed ordered sweep truncated to the requested count [pinch_design_method.py:_PDM_STAGE_PAIR_SWEEP].
- Canonical restrictions drop empty stage gaps, keep matches sharing a hot or cold stream in one stage, and split independent groups into consecutive stages [topology.py:canonical_topology_restrictions].
- Seeded TDM and ESM tasks are de-duplicated on dt_cont multiplier and canonical topology, plus derivative threshold when thresholds are distinct [_stage.py:build_seeded_quality_tasks].
- Single-method workflows replace `method_sequence` and `design_method` in the settings [_stage.py:run_single_method_workflow].

## Side Effects
- Constructs the executor from the `default_executor` factory its caller passes (`LocalSynthesisExecutor`, solver processes, for the OpenHENS methods) when none is supplied [_stage.py:run_stage].
- Emits a `RuntimeWarning` for the Couenne fallback [open_hens_method.py:execute_open_hens_method].
- Times workflows with `perf_counter`; no file IO.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_topology.py tests/analysis/heat_exchanger_networks/test_pinch_design_method.py tests/analysis/heat_exchanger_networks/test_thermal_derivative_method.py tests/analysis/heat_exchanger_networks/test_network_evolution_method.py tests/analysis/heat_exchanger_networks/test_heuristic_pinch_design_method.py`
