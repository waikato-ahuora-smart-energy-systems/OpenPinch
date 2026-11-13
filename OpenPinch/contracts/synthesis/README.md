# Heat Exchanger Network Synthesis Contracts

## Overview
Pydantic models for heat exchanger network (HEN) synthesis: manifests and export
records, per-method task inputs and outputs, topology restrictions, the
problem-owned result, and a frozen design view for clients that draw it. They use
`HeatExchangerNetwork` and the design-method enum from `OpenPinch/domain`.
`OpenPinch/contracts/output.py` embeds the result in `TargetOutput`, and
`OpenPinch/contracts/schema.py` exports the result and view as JSON Schema.

## Interface
- `HeatExchangerNetworkSynthesisManifest`, `HeatExchangerNetworkSynthesisExportRecord`, `SynthesisTaskStatus`, `SynthesisOutputFormat` (`common.py`): run manifest (grids, stage selection, method sequence, limits) and export-record types.
- `HeatExchangerNetworkSynthesisMethodInput`, `HeatExchangerNetworkSynthesisMethodOutput` (`method.py`): one PDM, TDM or evolution run; the input derives a deterministic `task_id` via `generate_task_id`.
- `HeatExchangerNetworkSynthesisTask`, `HeatExchangerNetworkSynthesisTaskOutcome` (`task.py`): task record, and an outcome whose `task` is required.
- `HeatExchangerNetworkTopologyRestriction` (`topology.py`): source/sink stream, stage and duty inherited by downstream tasks.
- `HeatExchangerNetworkSynthesisResult` (`result.py`): network, identities, solver status, objective values and ranked outcomes.
- `HeatExchangerNetworkView` and its rows (`view.py`): ranking, utility duties and grid layout (`HeatExchangerNetworkGridLayout`, `...GridMatch`, `...GridStream`, `...GridBranchCount`, `...RankingRow`, `...UtilityDuty`); `GridMatchKind`.
- Private helpers in `common.py` (`_validate_run_id` and similar) are imported by the sibling modules.
- Subdirectories: none.

## Invariants
- `run_id` matches `^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$` [common.py:_validate_run_id].
- Identity strings are stripped and must be non-empty [common.py:_validate_optional_identity].
- dt_cont multiplier and derivative-threshold grids are non-empty, finite and positive; `stage_selection` is non-empty, positive and unique [common.py:HeatExchangerNetworkSynthesisManifest].
- `search_width` lies in 0..3 and `search_depth` in 0..4; `pdm_stage_pair_limit` in 0..12; `stage_packing` is one of auto, none, pdm, tdm, all [common.py:HeatExchangerNetworkSynthesisManifest].
- `task_id` is `hens-task-` plus the first 16 hex characters of a SHA-256 of the sorted-key JSON of the task's identifying fields (`task_id`, `seed_network` and `metadata` are excluded; `settings` must be JSON-serialisable), so equal definitions give equal ids; a supplied `task_id` is kept (stripped of surrounding whitespace) [method.py:HeatExchangerNetworkSynthesisMethodInput.generate_task_id].
- Method output `objective_value` and result `objective_values` are finite and non-negative; `stage_count` is positive on the method input and the result [method.py:HeatExchangerNetworkSynthesisMethodOutput, result.py:HeatExchangerNetworkSynthesisResult].
- Topology restriction `stage` is positive and `duty` non-negative and finite [topology.py:HeatExchangerNetworkTopologyRestriction].
- Models forbid extra fields and validate on assignment, except the view models, which are frozen [view.py:_ViewModel].
- By convention, layouts built by `presentation/network_grid/layout.py` list recovery, then heater, then cooler matches; the model does not enforce the order, and drawing positions are left to the client [view.py:HeatExchangerNetworkGridLayout].

## Side Effects
- None. Method input fills `task_id` and output fills `method` from `task` during validation [method.py:_ensure_task_id, _fill_method_from_task].

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_contracts.py` (the nearest existing test; there is no dedicated synthesis model test under tests/contracts).
