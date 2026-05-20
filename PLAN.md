# OpenPinch Package Sweep Plan

Last sweep: 2026-05-21

This plan replaces the previous frontend-oriented roadmap with a package-wide
maintenance and hardening backlog based on the current repository state.

## Verified Baseline

- `uv run pytest -q` passes: `833 passed in 32.21s`.
- `uv run python -m build --wheel --sdist --no-isolation` passes.
- `uv run ruff check OpenPinch --select F401,F403,F405,E711,F541,W291,I001`
  passes.
- Docs build smoke and release artifact boundary checks are now covered by the
  default test suite.

The package is currently green for the restored verification gates, staged
package lint, and build flow, but the sweep still shows substantial structural
debt in the public orchestration layer and several partial subsystems.

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
- It is declared as a `@dataclass` but also has a custom `__init__`, mutable
  internal state, and fields shadowed by properties.
- Ruff already flags field/property collisions in this file.

What to do:

- Split loader logic away from orchestration.
- Split schema/semantic validation and message formatting into dedicated
  helpers.
- Split export and dashboard integration away from core solve orchestration.
- Remove the current partial-dataclass pattern and make state ownership
  explicit.

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

## Next

### 5. Fix documentation contradictions and widen docs drift checks

Why this matters:

- `docs/overview/support-and-stability.rst` still claims stable CLI commands
  for `run`, `graph`, `validate`, and `sample`, but the actual CLI currently
  exposes only `notebook`.
- Existing docs consistency tests do not cover that page, so the contradiction
  slipped through.

What to do:

- Align support/stability pages with the real CLI and public API.
- Extend docs consistency coverage to overview/support pages, not just the
  quickstart and guide set.
- Review API reference pages for modules that should not be elevated as normal
  user-facing workflows.

Done when:

- The docs tell one coherent story about what users can actually call today.

### 6. Add clean-install testing for optional dependency boundaries

Why this matters:

- The repository advertises `dashboard`, `notebook`, and `brayton_cycle`
  extras.
- The current test environment installs the dev surface, so base-install
  behavior is not being proven separately.
- Import guards exist in several places, but that is not the same as having a
  tested extra matrix.

What to do:

- Add smoke tests for a core install with no extras.
- Add separate smoke tests for `dashboard`, `notebook`, and `brayton_cycle`
  installs.
- Verify both import behavior and small entrypoint calls under the right extra.

Done when:

- The claimed optional dependency model is verified instead of assumed.

### 7. Clean up packaging and toolchain metadata

Why this matters:

- The build is healthy right now, but some metadata still needs attention.
- `pyproject.toml` has small cleanup debt such as duplicate `ruff` entries in
  the dev dependency group.
- `requires-python = ">=3.14"` may be correct, but it should be an intentional
  policy backed by code and CI, not a floor that simply stayed in place.

What to do:

- Deduplicate dependency declarations.
- Confirm the minimum supported Python version and test it intentionally.
- Keep build commands centralized so docs, CI, and local workflows use the same
  path.

Done when:

- Packaging metadata is minimal, intentional, and verified.

## Later

### 8. Performance and memory profiling for larger studies

Do this after the structural refactors above. Profiling a monolith before it is
cleanly separated tends to produce noisy results and hard-to-apply findings.

### 9. Bundle evolution and migration policy

`PinchWorkspace` persistence is already functional. Once the workspace layer is
refactored, add explicit versioning, migration rules, and cache-compatibility
tests rather than letting bundle behavior drift implicitly.

## Guiding Rule

Do not spend the next cycle adding new feature surface area before items 1
through 4 are under control. The package is currently functional, but the sweep
shows that maintenance debt is concentrated in public orchestration code and
partially restored subsystems. That is the real constraint on the next round of
progress.
