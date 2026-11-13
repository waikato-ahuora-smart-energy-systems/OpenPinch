# HEN Heuristic Pinch Design

## Overview
A rules-based (Linnhoff) pinch design method. It proposes network topologies for each pinch side; the pinch design side model in `../models` realises each proposal with only its matches allowed. This package is pure Python decision logic: it reads a set-up, unsolved `PinchDecompModel` and builds no model of its own.

## Interface
- `pinch_side`, `PinchSide`, `SideStream`, `PINCH_TOLERANCE` (`side_streams.py`): one side's decision data. Each active stream carries per-period loads, the CP at its pinch end, whether it reaches the pinch, and its temperature against the heat counted from its pinch end; the side carries pair approaches, the recovery pairs the side model can place, and the utilities each stream can use.

- `pinch_streams`, `number_rule_met`, `cp_rule_met`, `split_counts`, `cp_feasible`, `CP_TOLERANCE` (`rules.py`): the Linnhoff pinch rules. The "must" streams are those with no utility on the side (hot above the pinch, cold below) and their partners the other kind; `split_counts` gives the branches per stream after the splitting rules, and `cp_feasible` whether split ratios exist that meet the CP rule on a set of pinch matches.
- `propose_side_topologies`, `pinch_assignments`, `SideProposal`, `ProposedMatch`, `LOAD_FRACTION_TOLERANCE` (`enumeration.py`): up to a cap of distinct side topologies. A `SideProposal` lists its recovery matches (with tick-off duties) in a stage grid, the allowed heaters and coolers, its stage count and whether every must-stream load was placed; `z_restriction` gives the `[z, z_hu, z_cu]` restrictions the side model takes.

## Invariants
- Only streams active on the side in some period are listed. A stream absent in a period has zero load, zero CP and no pinch contact there [side_streams.py:_side_stream].
- The pinch end is a hot stream's outlet above the pinch and its inlet below; a cold stream's inlet above and its outlet below. A stream reaches the pinch in a period when its pinch-end temperature is within `PINCH_TOLERANCE` K of the pinch temperature shifted by its own contribution [side_streams.py:_side_stream].
- A segmented stream's pinch CP is that of the clipped segment at its pinch end, and its temperature follows the clipped piecewise profile [side_streams.py:_profile_points].
- `load` and `pinch_cp` are period-weighted means; `pinch_cp` averages only over periods where the stream is active [side_streams.py:SideStream].

- Rules apply only to streams that reach the pinch. The CP rule for a pinch match holds when the must stream's CP is at most its partner's, within `CP_TOLERANCE`, in every period both reach the pinch [rules.py:cp_rule_met].
- Splitting: a must stream that no partner can take is split in two, then while must branches outnumber partner branches the partner with the largest CP per branch gains a branch [rules.py:split_counts].
- `cp_feasible` checks each period separately as a transport problem (max flow): every must stream's CP shared over its pinch matches with no partner over its own CP [rules.py:cp_feasible].

- Pinch matches: each must stream takes as many partners as `split_counts` gives it branches and a partner hosts at most its branch count, with split ratios meeting the CP rule (`cp_feasible`); sets with fewer splits come first. If none exists, each partner may host one more branch, then the CP rule is dropped [enumeration.py:pinch_assignments].
- Tick-off: a pinch match takes the smaller of its two loads; a split stream shares its load over its pinch matches in proportion to the other streams' CPs [enumeration.py:_tick_off].
- Away from the pinch, the must stream with the largest remaining load is matched next, branching over partners (tick-off first, then larger duty). A pair is matched at most once. A match's duty is the largest, up to both remaining loads, whose two ends keep the pair's minimum approach in every period both streams run, each stream's slice taken next from its pinch end and scaled by its period load [enumeration.py:_extend, _feasible_duty].
- Stages: pinch matches sit next to the pinch (stage S - 1 above, stage 0 below) and every further match one stage further out than the last match on either of its streams. Partners with load left get every utility they may use [enumeration.py:_place, _proposal].
- Loads within `LOAD_FRACTION_TOLERANCE` of a stream's load count as met. Proposals are de-duplicated by `signature`; each pinch-match set has its own depth-first search and the searches take turns; complete proposals come first and incomplete ones only fill the cap. The search is bounded by a node budget per proposal requested [enumeration.py:propose_side_topologies].

## Side Effects
- None: reads model attributes and solver arrays only.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_heuristic_side_streams.py tests/analysis/heat_exchanger_networks/test_heuristic_rules.py tests/analysis/heat_exchanger_networks/test_heuristic_enumeration.py`
