# Workspace helpers

## Overview
Private helpers owned by `OpenPinch/application/workspace.py`. They normalise case inputs into JSON-style dictionaries and manage which case is active, which are cached as live `PinchProblem` objects, and when stored input is synchronised from a live case. `PinchWorkspace` imports `state` as a module and imports named functions from `case_inputs`.

## Interface
- `normalise_case_input` (`case_inputs.py`): returns a defensive copy of a `TargetInput` (as a Python-mode dump) or `dict`; other types raise `TypeError`.
- `project_name_from_case_input` (`case_inputs.py`): root `zone_tree.name`, or `None`.
- `merge_case_inputs` (`case_inputs.py`): deep merge of two case input dictionaries; the overlay wins and nested dictionaries merge.
- `canonical_case_input_from_source` (`case_inputs.py`): converts a `PinchProblem`, `TargetInput`, dict, path or CSV tuple into `(case_input, project_name)`.
- `JsonDict`, `PathLike` (`case_inputs.py`): type aliases re-exported for the workspace.
- `case_for_name` (`state.py`): returns the cached live problem for a case or builds and caches one.
- `default_case_name` (`state.py`): active case, else the baseline, else the first case, else `None`.
- `resolve_case_name` (`state.py`): resolves an optional name or raises `KeyError`.
- `invalidate_case_state` (`state.py`): drops the cached problem for one case.
- `sync_case_input` (`state.py`): writes a live problem's canonical input back into workspace storage.
- Subdirectories: none.

## Invariants
- The helpers operate on the workspace's private `_case_inputs`, `_case_cache`, `_active_case_name`, `baseline_name` and `project_name` attributes; renaming them breaks this directory [state.py].
- Stored inputs are deep copies, so callers cannot alias workspace storage [case_inputs.py:normalise_case_input] [state.py:case_for_name].
- A cached problem is returned without reassigning its name, because that would invalidate its cached results [state.py:case_for_name].
- A source that fails `PinchProblem` construction with `ValueError` is stored as a plain dictionary so it can be corrected; a non-dict source re-raises [case_inputs.py:canonical_case_input_from_source].
- The explicit `project_name` argument always wins. A `PinchProblem` source then uses its own `project_name`, then the zone tree name, then the workspace name; a `TargetInput` uses the zone tree name, then the workspace name; a dict or file source uses the zone tree name, then the built problem's name, then the workspace name (an invalid dict that cannot be built falls back to the argument, the workspace name, then the tree name). The final default is `"Site"` [case_inputs.py:canonical_case_input_from_source].
- `case_inputs.py` imports `PinchProblem` at module level, while `state.py` imports it inside functions [case_inputs.py] [state.py:case_for_name].

## Side Effects
- `default_case_name` sets `workspace._active_case_name` as a side effect of resolving a default [state.py:default_case_name].
- `case_for_name` constructs a `PinchProblem` (validation and zone-tree build) and stores it in the cache [state.py:case_for_name].
- `sync_case_input` replaces the stored input for a case when it differs from the live canonical input [state.py:sync_case_input].
- No file or network IO in this directory.

## Verification
- Run: `uv run --no-sync pytest tests/application/test_pinch_workspace.py`
- There is no `tests/application/_workspace` directory; the helpers are exercised through `PinchWorkspace`.
