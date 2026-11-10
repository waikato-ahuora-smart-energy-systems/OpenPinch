# Application Problem Targeting

## Overview
Execution support for the `problem.target` accessor. `catalog.py` declares the named methods and
their capabilities, `dispatch.py` orders zone traversal, `execution.py` runs services on a prepared
zone tree and caches `TargetOutput`, `state.py` isolates each call on a copy and commits it on
success, and `provenance.py` fingerprints inputs and stamps targets so later calls can validate a
`base_target`. The accessors in `OpenPinch/application/_problem/accessors`, `problem.py`,
`workspace.py`, `case_derivation.py`, `hpr_selection.py`, `heat_recovery_dt_min.py` and
`hpr_load_kinks.py` use it.

## Interface
- `METHOD_CATALOG`, `AnalysisMethodSpec` (`catalog.py`): read-only map of `target.*` and `design.*` method names to scopes, prerequisites, period behaviour, dependencies and result adapter.
- `require_available`, `catalog_method_names`, `install_catalog_forwarders` (`catalog.py`): reject unavailable methods; list a surface's names; add `**kwargs` forwarders to a class.
- `run_targeting_for_zone_and_subzones` (`dispatch.py`): children first, then direct and indirect services by zone type.
- `execute_targeting`, `execute_cogeneration_targeting`, `execute_exergy_targeting` (`execution.py`): run one target family and return the resulting target; `execute_targeting` returns `None` when a heat pump, refrigeration or ORC service produces no target.
- `run_problem_targeting` (`execution.py`): run a whole tree and cache `problem._results`; `run_exergy_targeting_for_zone_and_subzones` walks a tree in post-order calling the exergy service and caches nothing.
- `resolve_target_zone`, `walk_zone_tree`, `resolve_runtime_period_options`, `build_execution_master_zone`, `scratch_child_targets` (`execution.py`): zone and period selection helpers.
- `process_component_work_for_zone`, `attach_process_component_work_targets` (`execution.py`): add attached component work to targets.
- `AnalysisExecutionContext`, `PreparedPeriodState` (`state.py`): isolated invocation context and the per-period retained runtime graph.
- `snapshot_problem`, `prepared_period_state`, `commit_problem`, `invalidate_analysis`, `target_transaction` (`state.py`): copy, publish and clear problem state; decorator for accessor methods.
- `canonical_json`, `input_fingerprint`, `make_provenance`, `stamp_targets` (`provenance.py`): fingerprint inputs and stamp `AnalysisProvenance` on targets.
- `resolve_target_selection`, `resolve_hpr_residual_case`, `validate_base_target` (`provenance.py`): check `base_target` references.
- Subdirectories: none.

## Invariants
- `METHOD_CATALOG` is a `MappingProxyType`; Brayton heat pump and refrigeration are unavailable and raise `NotImplementedError`. [catalog.py:METHOD_CATALOG], [catalog.py:require_available]
- Child zones are targeted before their parent; `base_target_type` is removed from child arguments. [dispatch.py:run_targeting_for_zone_and_subzones]
- A zone type outside the known set raises `ValueError`. [dispatch.py:run_targeting_for_zone_and_subzones]
- A target zone selector must match exactly one zone by address or name. [execution.py:resolve_target_zone]
- Indirect (total site) heat integration targets (`SA`, `II`) need subzones; indirect heat pump and refrigeration targets do not. [execution.py:_require_subzones_for_indirect_target]
- Ancestor targets and graphs are cleared before a non-energy-transfer run. [execution.py:execute_targeting]
- Without `include_subzones`, child targets are computed on scratch copies and restored. [execution.py:scratch_child_targets]
- Every `target_transaction` call runs on `snapshot_problem` and publishes only on success; published results keep only rows and graphs in the invocation's scope. [state.py:target_transaction]
- A frozen HPR residual basis only allows `direct_heat_integration`. [state.py:target_transaction]
- A `base_target` must be a current local target of the same owner, input fingerprint, zone and period, with a single period. [provenance.py:validate_base_target], [provenance.py:resolve_target_selection]
- Fingerprints are SHA-256 of canonical JSON with sorted keys and no NaN; unserialisable types raise `TypeError`. [provenance.py:canonical_json], [provenance.py:input_fingerprint]
- Targets are stamped children and prerequisites first. [provenance.py:stamp_targets]
- Committed components keep their identity: existing handles are updated in place and re-pointed at the problem. [state.py:commit_problem]

## Side Effects
- Targeting mutates the zone tree's `targets` and `graphs` and sets `problem._results`. [execution.py:run_problem_targeting], [execution.py:execute_targeting]
- `build_execution_master_zone` may call `problem.load(...)`. [execution.py:build_execution_master_zone]
- `target_transaction` assigns the problem's master zone, components, results and last run spec, stores `_period_states[period_id]`, and imports `retain_hpr_target` lazily. [state.py:target_transaction]
- `invalidate_analysis` clears every zone's targets and graphs and the problem's cached result fields. [state.py:invalidate_analysis]
- `install_catalog_forwarders` sets attributes on a class. [catalog.py:install_catalog_forwarders]

## Verification
- Run: `uv run --no-sync pytest tests/application/test_analysis_lifecycle.py tests/application/test_analysis_contracts.py`
- There is no `tests/application/_problem` directory; these are the nearest focused tests. Broader: `uv run --no-sync pytest tests/application`.
