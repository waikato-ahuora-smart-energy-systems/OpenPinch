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
- The same key ranks and accepts outcomes: networks that hold their utility levels first, then lowest objective (a site network's `summary_metrics["site_total_annual_cost"]` in place of its objective, `ranking_cost`), task id as tie-break, missing objective last [ranking.py:outcome_ranking_key].
- A network holds its utility levels when each utility's duty in every period is within `LEVEL_TOLERANCE` (1e-3, relative to the target or of 1 kW) of `source_metadata["utility_level_targets"]`, summing its heaters or coolers by utility name; a network held at least at its levels (`source_metadata["utility_level_holds"] == "at_least"`) also holds them above the target; a network without targets holds them [ranking.py:holds_utility_levels].
- Only successful outcomes with a network, an objective and a passing feasibility check are ranked [ranking.py:_ranked_candidate_outcomes].
- Structure signatures ignore duty and count only active matches; a heater or cooler keeps its place among the recovery stages (`None` at the stream end), so the same unit on another side of a recovery stage is another structure. A staged heater sits on its cold stream at the boundary before its stage and a staged cooler on its hot stream at the boundary after it; recovery matches and staged units are then numbered by each stream's order of exchangers, so shifting, gapping or renumbering the stages of exchangers that share no stream does not change the signature [ranking.py:network_structure_signature] [../targeting/topology.py:canonical_stage_indices].
- Verification reads a utility slice's approach and film coefficient under its parent's name (`solver_axis_metadata["utility_parents"]`), as its exchangers carry the parent's name [verification.py:_stream_order].
- Costs are checked for consistency, not recomputed: total annual cost equals utility plus capital, and, when every exchanger records its capital cost, the network's capital equals their sum. The cost law and utility prices are not on the network, so capital is not recomputed from areas nor utility cost from duties [verification.py:_check_cost_consistency, _check_exchanger_capital].
- Summary totals are checked only for single-period networks; stream balances are checked per period otherwise, from the `*_by_period` stream metadata (a network without it has no per-period balance check) [verification.py:verify_network_feasibility].
- The area check takes each period's heat transfer coefficients from `*_heat_transfer_coefficients_by_period` when the metadata has them, else the scalar ones [verification.py:_check_area_consistency].
- Networks whose `source_metadata["solver_non_isothermal_model"]` is explicitly falsy and whose `solver_framework` is PDM or TDM (isothermal stage boundaries) skip per-exchanger duty and LMTD checks and the area-consistency check; stream-level balances and segment contribution checks still apply [verification.py:_uses_isothermal_stage_boundaries].
- Each match is checked against its own required approach when the network records it; a recovery match without one falls back to the network's `minimum_recovery_approach` source metadata (a duty-optimised network records it) [verification.py:_check_minimum_approach].
- An exchanger's area fails only when below what its duty needs: the required area for a single-segment exchanger, the maximum period-total segment area for a segmented one; spare area (an installed unit running below its area) passes; a rated network's area, single-segment or segmented, is widened by its `source_metadata["rating_area_allowances"]` entry (exact over the rating's calibrated Chen sizing) before the check. Each segment contribution's own area must still match its duty/U/LMTD [verification.py:_check_area_consistency, _check_segment_contribution_areas].
- Tolerances are module constants (duty 1.0 absolute and 1e-4 relative, cost 1.0 and 1e-4, temperature 1e-3, area 1e-3 and 1e-4) [verification.py:_DUTY_ABS_TOL].
- Export requires a live problem with a cached design, else `TypeError` or `RuntimeError` [exports.py:export_heat_exchanger_network_synthesis_results].
- Export paths in the manifest are relative to the output directory in POSIX form [exports.py:_record].

## Side Effects
- Export creates `results/` and `metrics/` under the output directory and writes `results/<task_id>.json`, `metrics/solution_metrics.csv`, `metrics/run_summary.csv` and `manifest.json`; tasks without an id share `results/unknown-task.json` and existing files are overwritten [exports.py:export_heat_exchanger_network_synthesis_results].
- Verification and ranking are pure.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_reporting.py`
