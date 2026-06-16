# HENS-01 Dependency and Runtime Viability Spike

## PRD Summary

Prove that OpenHENS solver dependencies can coexist with OpenPinch packaging
without making the core package heavy or fragile. This task decides the optional
dependency shape before any equation models are moved.

## User Outcome

OpenPinch users who do not use HEN synthesis can keep installing and importing
the core package without GEKKO, Pyomo, solver binaries, plotting stacks, or
workbook dependencies. HEN users get a documented optional install path.

## Scope

- Packaging metadata, dependency resolution, import behavior, CI marker policy,
  and smoke tests.
- No synthesis equation move.
- No public HEN workflow implementation.

## Plan Context

Read these sections before implementation:

- [Design Principles](../../../OPENHENS_MIGRATION_PLAN.md#design-principles)
- [Phase 1: Dependency and Runtime Viability Spike](../../../OPENHENS_MIGRATION_PLAN.md#phase-1-dependency-and-runtime-viability-spike)
- [Validation Strategy](../../../OPENHENS_MIGRATION_PLAN.md#validation-strategy)
- [Risks and Mitigations](../../../OPENHENS_MIGRATION_PLAN.md#risks-and-mitigations)
- [Non-Goals for the First Migration](../../../OPENHENS_MIGRATION_PLAN.md#non-goals-for-the-first-migration)

Settled decisions for this task:

- Solver packages are optional synthesis dependencies, not core dependencies.
- Importing `OpenPinch` must not import GEKKO, Pyomo, solver factories,
  plotting stacks, workbook libraries, or wake-management packages.
- The Python version mismatch must be resolved or documented before dependency
  additions proceed.
- This task decides dependency viability only; it does not move equations or
  create the public workflow.
- The required workflow in `README.md` is mandatory; dependency scaffolding must
  not create alternate imports or execution paths around it.

## Requirements Checklist

- [x] Record the Python version policy decision, including the current OpenPinch
      `>=3.14` target and the source OpenHENS `>=3.12` target.
- [x] Test `pyomo`, `gekko`, `matplotlib`, `plotly`, `kaleido`, `openpyxl`,
      and `wakepy` under the selected OpenPinch Python target.
- [x] Decide whether a `synthesis` optional extra can be added now or must wait
      for dependency compatibility work.
- [x] If the extra is added, keep synthesis dependencies out of core and out of
      unrelated extras.
- [x] Decide whether `synthesis` belongs in the `full` extra and document why.
- [x] Update packaging metadata tests for the new extra, or add an explicit
      blocker note if the extra cannot be added yet.
- [x] Define CI marker policy for fast tests, optional synthesis tests, and
      solver-binary tests.
- [x] Add import smoke tests proving `import OpenPinch` does not import GEKKO,
      Pyomo, solver factories, plotting stacks, workbook libraries, or wake
      management packages.
- [x] Keep GEKKO/Pyomo imports inside backend modules or functions that are
      reached only by synthesis execution.
- [x] Add actionable missing-dependency error messages for the future synthesis
      path, even if the path is not fully implemented in this task.
- [x] Update developer documentation with the optional install and test marker
      policy.

## General Standards That Apply

- [x] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [x] Core OpenPinch install remains lightweight.
- [x] Optional synthesis dependencies are lazy and isolated.
- [x] No public `OpenHENS` facade, import shim, or runtime compatibility layer
      is introduced to support dependency loading.
- [x] Solver binaries are never assumed to be available in default CI.

## Verification Checklist

- [x] `rtk uv run pytest tests/test_packaging_metadata.py -q` passes or the
      packaging test equivalent passes.
- [x] `rtk uv run pytest tests/test_package_api_surface.py -q` passes or the API
      surface equivalent passes.
- [x] A core import smoke test demonstrates that synthesis-only packages are not
      imported by `import OpenPinch`.
- [x] `rtk uv sync` or the repo's core install command succeeds without
      synthesis dependencies.
- [x] `rtk uv sync --extra synthesis` succeeds if the extra is added; otherwise
      the dependency/version blocker is documented with exact package names and
      constraints.

## Definition of Done

- [ ] The project has a reviewed decision on Python and optional synthesis
      dependency policy.
- [x] Core import and install behavior is protected by tests.
- [x] Solver and synthesis dependencies are optional, lazy, and documented.
- [x] CI knows which tests are fast default tests and which require solver
      binaries.
- [x] No solver model code moved as part of this task.

## Out of Scope

- Moving OpenHENS equation models.
- Building the HEN design workflow.
- Converting examples.
- Running solver benchmark parity.

## Implementation Notes

- 2026-06-16: Python policy decision is to keep OpenPinch at
  `requires-python = ">=3.14"` / `.python-version` `3.14` while treating source
  OpenHENS' `>=3.12` as migration source context only. `rtk uv run python
  --version` reported `Python 3.14.2`.
- 2026-06-16: Added `openpinch[synthesis]` with `pyomo>=6.10.0`,
  `gekko>=1.3.2`, `matplotlib>=3.10.9`, `plotly>=6.8.0`,
  `kaleido>=1.3.0`, `openpyxl>=3.1.5`, and `wakepy>=1.0.0`. `full`
  intentionally excludes solver-only synthesis packages so existing full
  optional installs do not pull HEN solver stacks.
- 2026-06-16: `rtk uv lock` passed and resolved 141 packages, adding
  `choreographer==1.3.0`, `gekko==1.3.2`, `jeepney==0.9.0`,
  `kaleido==1.3.0`, `logistro==2.0.1`, `orjson==3.11.9`,
  `pyomo==6.10.1`, `simplejson==4.1.1`, and `wakepy==1.0.0`.
- 2026-06-16: `rtk uv run python -c "import importlib.metadata as m; ..."`
  under the synthesis environment reported `pyomo==6.10.1`, `gekko==1.3.2`,
  `matplotlib==3.10.9`, `plotly==6.8.0`, `kaleido==1.3.0`,
  `openpyxl==3.1.5`, and `wakepy==1.0.0`.
- 2026-06-16: CI marker policy is in `pytest.ini`: unmarked tests are fast
  default tests, `@pytest.mark.synthesis` requires `openpinch[synthesis]` but no
  external solver binary, and `@pytest.mark.solver` requires external binaries
  such as Couenne or IPOPT. Developer docs record rerun commands.
- 2026-06-16: Added internal
  `OpenPinch.services.heat_exchanger_network_synthesis._dependencies` guards.
  Missing Python packages raise `MissingSynthesisDependencyError` with
  `python -m pip install "openpinch[synthesis]"`; missing binaries raise
  `MissingSynthesisSolverError` with solver installation and `PATH` guidance.
- 2026-06-16: Follow-up review fix removed repository-only `rtk uv` rerun
  guidance from runtime `MissingSynthesisDependencyError` and
  `MissingSynthesisSolverError` text. Runtime errors now point general users to
  normal optional package installation, solver installation on `PATH`, and the
  synthesis dependency policy docs; repository-specific `rtk uv ...` commands
  remain only in developer evidence notes.
- 2026-06-16: `rtk uv run pytest tests/test_packaging_metadata.py -q` passed:
  `10 passed in 0.05s`.
- 2026-06-16: `rtk uv run pytest tests/test_package_api_surface.py -q` passed:
  `2 passed in 7.26s`.
- 2026-06-16: `rtk uv run pytest tests/test_synthesis_dependency_boundaries.py
  -q` passed: `3 passed in 6.68s`. This includes a subprocess smoke test that
  `import OpenPinch` does not load `gekko`, `pyomo`, `pyomo.environ`,
  `pyomo.opt`, `matplotlib`, `matplotlib.pyplot`, `plotly`,
  `plotly.graph_objects`, `kaleido`, `openpyxl`, or `wakepy`.
- 2026-06-16: After the runtime error-message follow-up,
  `rtk uv run pytest tests/test_synthesis_dependency_boundaries.py -q` passed:
  `3 passed in 3.23s`.
- 2026-06-16: `rtk uv run pytest
  tests/test_streamlit_webviewer/test_web_graphing.py -q` passed:
  `7 passed in 2.69s`, covering the existing graph/dashboard helper touched to
  make Plotly and Streamlit imports lazy.
- 2026-06-16: `rtk uv sync` passed for the core sync gate: `Resolved 141
  packages in 3ms`; `Audited 127 packages in 3ms`.
- 2026-06-16: `rtk uv sync --extra synthesis` passed, installing the synthesis
  dependency delta including `gekko==1.3.2`, `kaleido==1.3.0`,
  `pyomo==6.10.1`, and `wakepy==1.0.0`.
- 2026-06-16: `rtk uv run python scripts/optional_install_smoke.py synthesis`
  passed and imported the declared synthesis dependency set.
- 2026-06-16: `rtk uv run pytest tests/test_docs_consistency.py -q` passed:
  `12 passed in 0.12s`, covering the added developer docs page in the docs
  index.
- 2026-06-16: `rtk uv run ruff check
  OpenPinch/streamlit_webviewer/web_graphing.py
  OpenPinch/services/heat_exchanger_network_synthesis
  tests/test_packaging_metadata.py tests/test_package_api_surface.py
  tests/test_synthesis_dependency_boundaries.py scripts/optional_install_smoke.py`
  passed.
- 2026-06-16: After the runtime error-message follow-up,
  `rtk uv run ruff check
  OpenPinch/services/heat_exchanger_network_synthesis/_dependencies.py
  tests/test_synthesis_dependency_boundaries.py` passed.
- 2026-06-16: No synthesis equations, public HEN workflow, OpenHENS facade,
  import shim, runtime CSV loader, case/study root, or standalone synthesis
  runner was added. The reviewed-decision Definition of Done item remains open
  pending adversarial review.
- 2026-06-16: HENS-01 review follow-up confirmed root `.DS_Store` is an
  unrelated pre-existing/user-owned worktree modification. `rtk git diff
  --cached --name-status` produced no staged entries, so `.DS_Store` remains
  unstaged, was not reverted, and is excluded from the HENS-01 task slice and
  final commit/PR scope.
