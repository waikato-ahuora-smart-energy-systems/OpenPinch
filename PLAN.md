# OpenPinch Package Sweep Plan

Last sweep: 2026-05-21

This plan replaces the previous frontend-oriented roadmap with a package-wide
maintenance and hardening backlog based on the current repository state.

## Verified Baseline

- `uv run pytest -q` passes: `840 passed in 30.73s`.
- `uv run pytest -q tests/test_release_artifacts.py tests/test_docs_build.py`
  passes.
- `uv run python scripts/build_dist.py` passes.
- `uv run ruff check .` passes.

## Current Debt Snapshot

- There is no active Ruff backlog at the repository level; the current lint
  baseline is clean.
- A small number of targeted Ruff `per-file-ignores` now exist for intentional
  wildcard-import test patterns and a handful of data-heavy or placeholder
  files. Those exceptions should be revisited if those files are refactored.

The package is currently green for the restored verification gates, full-repo
lint, and build flow, but the sweep still shows substantial structural debt in
the public orchestration layer and several partial subsystems.

## High-Risk Hotspots

- `OpenPinch/classes/pinch_problem.py` (`1028` lines)
- `OpenPinch/classes/pinch_workspace.py` (`734` lines)
- `OpenPinch/classes/_workspace_support.py` (`759` lines)
- `OpenPinch/services/input_data_processing/data_preparation.py` (`755` lines)
- `OpenPinch/services/common/graph_data.py` (`868` lines)
- `OpenPinch/utils/wkbook_to_json.py` (`395` lines)

These files are carrying too many responsibilities at once. Refactoring them
should be treated as maintainability work, not cosmetic cleanup.

## Now

### 1. Refactor `PinchProblem` into smaller components

Why this matters:

- `PinchProblem` currently mixes loading, validation, source normalization,
  zone-tree preparation, targeting orchestration, comparison helpers, Excel
  export, dashboard launch, and error formatting.
- The class still owns too many responsibilities and too much mutable state.

What to do:

- Split loader logic away from orchestration.
- Split schema/semantic validation and message formatting into dedicated
  helpers.
- Split export and dashboard integration away from core solve orchestration.
- Keep state ownership explicit as the class is broken into smaller units.

Done when:

- `PinchProblem` reads like a coordinator rather than a 1000-line utility
  bucket.
- Smaller units can be tested independently.

### 2. Refactor `PinchWorkspace` and `_workspace_support`

Why this matters:

- `pinch_workspace.py` and `_workspace_support.py` currently combine storage,
  payload normalization, validation, workflow dispatch, comparison logic,
  serialization, and frontend-view shaping.
- Broad exception-to-view translation hides real failure categories.
- This layer is close to being a second product surface and deserves clearer
  boundaries.

What to do:

- Split persistence/storage concerns from workflow execution.
- Split comparison and view-model generation from case lifecycle management.
- Narrow broad exception handling into clearer error/status taxonomies.
- Document which methods are stable user-facing API and which are adapter code.

Done when:

- The workspace layer has clean internal boundaries.
- Failure handling is predictable instead of broadly catch-all.

### 3. Simplify input preparation and workbook ingestion

Why this matters:

- `data_preparation.py` and `wkbook_to_json.py` are both large and highly
  procedural.
- `OpenPinch/lib/config.py` still carries an explicit TODO that many
  workbook-style options exist without a well-defined runtime role.
- The path from external payloads to canonical internal state is more complex
  than it needs to be.

What to do:

- Centralize canonical payload normalization.
- Separate zone-tree validation, stream/utility normalization, and default
  utility completion into smaller units.
- Audit configuration fields as one of: supported, legacy alias, workbook-only,
  experimental, or dead.
- Keep workbook parsing but make its contract and failure modes cleaner.

Done when:

- Input preparation is understandable in layers.
- Config options have a documented runtime status instead of mixed implicit
  behavior.

### 4. Quarantine, finish, or remove partial subsystems

Why this matters:

- `OpenPinch/services/heat_pump_integration/cycles/brayton.py` raises
  `NotImplementedError` immediately, but still contains dormant code below it.
- `OpenPinch/services/exergy_analysis/exergy_targeting_entry.py` is mostly a
  small helper plus commented-out restoration stubs.
- `OpenPinch/services/energy_transfer_analysis/energy_transfer_analysis.py` is
  effectively a large commented-out placeholder.
- `OpenPinch/services/common/graph_data.py` still contains a large commented
  ETD block.
- The API docs still present exergy and energy-transfer modules as if they are
  normal reference surfaces.

What to do:

- For each partial subsystem, choose one status: restore, explicitly mark
  experimental, move under a quarantine namespace, or remove.
- Delete large dead/commented blocks once the decision is made.
- Stop documenting intentionally partial modules like standard production
  workflows unless the page clearly says otherwise.

Done when:

- There is no ambiguity about what is supported versus merely present in the
  repository.

## Later

### 5. Performance and memory profiling for larger studies

Do this after the structural refactors above. Profiling a monolith before it is
cleanly separated tends to produce noisy results and hard-to-apply findings.

### 6. Bundle evolution and migration policy

`PinchWorkspace` persistence is already functional. Once the workspace layer is
refactored, add explicit versioning, migration rules, and cache-compatibility
tests rather than letting bundle behavior drift implicitly.

## Guiding Rule

Do not spend the next cycle adding new feature surface area before items 1
through 4 are under control. The package is currently functional, but the sweep
shows that maintenance debt is concentrated in public orchestration code and
partially restored subsystems. That is the real constraint on the next round of
progress.
