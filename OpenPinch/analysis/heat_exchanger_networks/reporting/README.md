# HEN Reporting

## Overview
Checks, ranks and exports synthesis designs. Feasibility checks work on service-boundary network data (exchanger states plus `source_metadata` and `solver_axis_metadata`), not on the private solver model. `../results` uses these checks to demote and rank outcomes, and `../context.py` uses them to verify the accepted design.

## Interface
- `verify_network_feasibility` (`verification.py`): feasibility failure strings for one network (summary totals, cost consistency, temperature order, minimum approach, duty and stream heat balances, areas).
- `is_network_feasible` (`verification.py`): true when there are no failures.
- `verify_synthesis_result` (`verification.py`): contract failures for an accepted design result and its ranked outcomes.
- `rank_unique_network_outcomes` (`ranking.py`): ranked feasible successful outcomes with duplicate structures removed.
- `network_structure_signature` (`ranking.py`): sorted topology signature of a network.
- `outcome_ranking_key` (`ranking.py`): `(objective, task_id)` key used for both acceptance and ranking.
- `export_heat_exchanger_network_synthesis_results` (`exports.py`): writes JSON and CSV views and a manifest from a problem's cached design.

## Invariants
- The same key ranks and accepts outcomes: lowest objective first, task id as tie-break, missing objective last [ranking.py:outcome_ranking_key].
- Only successful outcomes with a network, an objective and a passing feasibility check are ranked [ranking.py:_ranked_candidate_outcomes].
- Structure signatures ignore duty and count only active matches [ranking.py:network_structure_signature].
- Summary totals are checked only for single-period networks; stream balances are checked per period otherwise [verification.py:verify_network_feasibility].
- Networks whose `source_metadata["solver_non_isothermal_model"]` is explicitly falsy and whose `solver_framework` is PDM or TDM (isothermal stage boundaries) skip per-exchanger duty and LMTD checks and the area-consistency check; stream-level balances and segment contribution checks still apply [verification.py:_uses_isothermal_stage_boundaries].
- Each match is checked against its own required approach when the network records it, not only the global dTmin [verification.py:_check_minimum_approach].
- Tolerances are module constants (duty 1.0 absolute and 1e-4 relative, cost 1.0 and 1e-4, temperature 1e-3, area 1e-3 and 1e-4) [verification.py:_DUTY_ABS_TOL].
- Export requires a live problem with a cached design, else `TypeError` or `RuntimeError` [exports.py:export_heat_exchanger_network_synthesis_results].
- Export paths in the manifest are relative to the output directory in POSIX form [exports.py:_record].

## Side Effects
- Export creates `results/` and `metrics/` under the output directory and writes `results/<task_id>.json`, `metrics/solution_metrics.csv`, `metrics/run_summary.csv` and `manifest.json`; tasks without an id share `results/unknown-task.json` and existing files are overwritten [exports.py:export_heat_exchanger_network_synthesis_results].
- Verification and ranking are pure.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_reporting.py`
