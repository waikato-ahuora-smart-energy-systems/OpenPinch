# HEN Results

## Overview
Turns executed task outcomes into the canonical synthesis result, resolves seed networks for standalone methods, and selects a ranked network into a result. It sits after `../execution` and `../targeting` in the workflow and uses `../reporting` for feasibility checks and ranking keys.

## Interface
- `SynthesisWorkflowResult` (`assembly.py`): frozen dataclass of tasks, outcomes, accepted result and run time.
- `build_synthesis_result` (`assembly.py`): demotes infeasible outcomes, accepts the best one and builds the result with its manifest.
- `demote_infeasible_outcomes` (`assembly.py`): marks solver successes that fail `verify_network_feasibility`, or that carry `diagnostics["unconverged_termination"]`, as failed.
- `resolve_seed_networks` (`seeds.py`): returns explicit seed networks, or the cached design's selected network followed by its other ranked topologies, up to a limit.
- `ranked_networks` (`selection.py`): unique ranked outcomes of a result.
- `network_for_rank` (`selection.py`): network at a one-based rank, without mutation.
- `select_network` (`selection.py`): writes the chosen outcome into the result.

## Invariants
- The accepted outcome is the successful outcome with the lowest `outcome_ranking_key`, whichever method produced it; ties are broken by task id [assembly.py:_accepted_outcome].
- Infeasible and unconverged outcomes are demoted before acceptance, keeping their network and a reason; `WorkflowContractError` is raised only when no success remains [assembly.py:demote_infeasible_outcomes, _accepted_outcome].
- The accepted network must be present, and its run id, task id, period id, method are overwritten from settings and task, and the stage count is kept from the network when it has one [assembly.py:build_synthesis_result].
- Manifest dt_cont multipliers are those the tasks actually ran, falling back to the settings values [assembly.py:build_synthesis_result].
- Manifest `method_sequence` lists each method once, in outcome order (pinch design, then EVM tasks seeded directly, then TDM and its EVM tasks), without outcomes that failed because Couenne is missing; `settings.method_sequence` does not affect it [assembly.py:_executed_method_sequence].
- `solution_rank` is one-based; ranks below 1 or beyond the available count raise `IndexError` [selection.py:network_for_rank].
- `select_network` keeps whatever `ranked_networks` the result holds; for a finalised design that is the unique feasible ranking built by `finalise_design_result`, not every outcome, and `solution_rank` indexes that list [selection.py:select_network] [context.py:finalise_design_result].
- A cached seed may come from any HEN method; for problems with segmented streams every cached seed must carry the current `SEGMENT_PROFILE_VERSION` [seeds.py:resolve_seed_networks].
- Seed networks must be non-empty `HeatExchangerNetwork` instances or mappings that validate as one [seeds.py:_normalise_seed_networks].

## Side Effects
- `select_network` mutates its `result` argument (network, task id, solver status, method, stage count, objective values) [selection.py:select_network].
- `ranking` is imported inside the function rather than at module level [assembly.py:_accepted_outcome, selection.py:ranked_networks].
- Only demoted outcomes and the accepted network are copied; feasible outcomes, ranked outcomes and seed networks are returned as stored; no IO. [seeds.py:resolve_seed_networks] [selection.py:network_for_rank]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_reporting.py tests/analysis/heat_exchanger_networks/test_hen_audit.py`
- There is no results-specific test file; `seeds.py` is covered through the seeded method tests in the same directory.
