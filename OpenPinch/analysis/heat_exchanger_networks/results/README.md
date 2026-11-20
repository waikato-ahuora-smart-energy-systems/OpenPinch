# HEN Results

## Overview
Turns executed task outcomes into the canonical synthesis result, resolves seed networks for standalone methods, and selects a ranked network into a result. It sits after `../execution` and `../targeting` in the workflow and uses `../reporting` for feasibility checks and ranking keys.

## Interface
- `SynthesisWorkflowResult` (`assembly.py`): frozen dataclass of tasks, outcomes, accepted result and run time.
- `build_synthesis_result` (`assembly.py`): ranks the run record's networks, accepts the first and builds the result with its manifest.
- `HeatExchangerNetworkRecord` (`record.py`, public): thread-safe record of one run's verified networks; `version`, `status`, `outcomes()`, `snapshot()`, `changed_since(version)`.
- `admit(outcome)` (`record.py`): offers one solved outcome to the records installed in the current context; returns whether any changed.
- `recording_active()` (`record.py`): whether any record is installed in the current context.
- `recording(record, *, finalise=None)` and `running(record, *, finalise=None)` (`record.py`): context managers that install a record; `running` also makes it the run's record and settles its status.
- `run_record(outcomes)` (`record.py`): the enclosing run's record, or a record of the admissible `outcomes` outside a run.
- `demote_infeasible_outcomes` (`assembly.py`): marks solver successes that fail `verify_network_feasibility`, or that carry `diagnostics["unconverged_termination"]`, as failed.
- `resolve_seed_networks` (`seeds.py`): returns explicit seed networks, or the cached design's selected network followed by its other ranked topologies, up to a limit.
- `ranked_networks` (`selection.py`): unique ranked outcomes of a result.
- `network_for_rank` (`selection.py`): network at a one-based rank, without mutation.
- `select_network` (`selection.py`): writes the chosen outcome into the result.

## Invariants
- An outcome is admitted only when it succeeded with a network and a total annual cost and `demote_infeasible_outcomes` keeps it a success (converged, feasible); a destination's `finalise` result must pass the same rule. `finalise` runs in a copy of the context with no records installed, so the solves it makes never enter the run [record.py:admit, _admissible, _outside_records].
- A record holds one entry per topology (`network_structure_signature`, keyed by a 16-hex hash), the lower of `(misses its utility levels, total annual cost, task id, key)`, where a site network's cost is its `summary_metrics["site_total_annual_cost"]` (`../reporting/ranking.py:ranking_cost`), so a network that holds its utility levels (`holds_utility_levels`) comes before any that does not; entries are ordered by that key, and each entry's data carries `holds_utility_levels` (and `site_total_annual_cost` for a site network), so content and order do not depend on arrival order. Entries are never removed [record.py:HeatExchangerNetworkRecord._insert].
- `version` rises by one on every insert, replacement and status change; an entry's `version` is the record version at which it last changed. `snapshot()` and `changed_since()` return new JSON-ready data, built under the record's lock [record.py:HeatExchangerNetworkRecord].
- Status runs `pending`, `running`, then `complete` (first entry's key as `selected`) or `failed` (when the run raises); a record that is not pending raises `ValueError` when run again [record.py:running, HeatExchangerNetworkRecord._set_status].
- The result's ranked networks are exactly the run record's entries and the accepted outcome is the first; outside a run, the admissible outcomes make up the record. With none admitted, the failure summary of the outcomes is raised as `WorkflowContractError`, or "admitted no network" when some succeeded; the summary names a failed task by its method and dt_cont multiplier, or a fixed-structure task (`settings["fixed_structure"]`, stamped with the evolution method) as a fixed-structure duty optimisation with its objective [assembly.py:build_synthesis_result, _task_label] [record.py:run_record].
- Infeasible and unconverged outcomes are demoted, keeping their network and a reason, for the failure summary [assembly.py:demote_infeasible_outcomes, _accepted_outcome].
- The accepted network's run id, task id, period id, method are overwritten from settings and task, and the stage count is kept from the network when it has one [assembly.py:build_synthesis_result].
- Manifest dt_cont multipliers are those the tasks actually ran, falling back to the settings values [assembly.py:build_synthesis_result].
- Manifest `method_sequence` lists each method once, in outcome order (MILP pinch design, heuristic pinch design, thermal derivative, then network evolution), without outcomes that failed because Couenne is missing; `settings.method_sequence` does not affect it [assembly.py:_executed_method_sequence].
- `solution_rank` is one-based; ranks below 1 or beyond the available count raise `IndexError` [selection.py:network_for_rank].
- `select_network` keeps whatever `ranked_networks` the result holds; for a finalised design that is the unique feasible ranking built by `finalise_design_result`, not every outcome, and `solution_rank` indexes that list [selection.py:select_network] [context.py:finalise_design_result].
- A cached seed may come from any HEN method; for problems with segmented streams every cached seed must carry the current `SEGMENT_PROFILE_VERSION` [seeds.py:resolve_seed_networks].
- Seed networks must be non-empty `HeatExchangerNetwork` instances or mappings that validate as one [seeds.py:_normalise_seed_networks].

## Side Effects
- `admit` and `running` mutate the installed records; `admit` runs each destination's `finalise`, which may solve (fit and rate a network) [record.py:admit].
- `select_network` mutates its `result` argument (network, task id, solver status, method, stage count, objective values) [selection.py:select_network].
- `ranking` is imported inside the function rather than at module level [assembly.py:_accepted_outcome, selection.py:ranked_networks].
- Only demoted outcomes and the accepted network are copied; feasible outcomes, ranked outcomes and seed networks are returned as stored; no IO. [seeds.py:resolve_seed_networks] [selection.py:network_for_rank]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_reporting.py tests/analysis/heat_exchanger_networks/test_hen_audit.py tests/analysis/heat_exchanger_networks/test_network_record.py`
- There is no results-specific test file; `seeds.py` is covered through the seeded method tests in the same directory.
