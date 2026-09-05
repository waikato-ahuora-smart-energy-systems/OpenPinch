# RTD and Comprehensive HPR Notebook Code Generation Plan

## Unit Context

- Unit: generated comprehensive HPR tutorial and synchronized RTD coverage.
- Requirements: FR-NB-01 through FR-NB-08 and NFR-NB-01 through NFR-NB-06.
- Existing owners: tutorial generator, packaged notebook 09, tutorial manifest,
  packaging tests, HPR RTD guide, notebook series, and coverage page.
- Dependencies: completed HPR target backend selector, winning simulation record,
  target-owned map accessor, Unit 1 map request/contract, and optional TESPy
  adapter.
- Database or infrastructure entities: none.
- Single source of truth: this plan for the focused generation stage.

## Public Contract Used by the Tutorial

- Root workflow: `from OpenPinch import PinchProblem`.
- Specialist published request:
  `from OpenPinch.contracts.hpr_performance_map import HprPerformanceMapRequest`.
- Target calls: current vapour-compression heat-pump/refrigeration methods with
  omitted/default CoolProp and explicit TESPy selection.
- Map call: `problem.target.hpr_performance_map(target=..., request=...)`.
- Consumer boundary: `performance_map.model_dump(mode="json")`.

## Execution Steps

### Step 1: Add RED Tutorial and RTD Contracts

- [x] Extend notebook packaging tests to require notebook 09 to import the
  specialist HPR request contract, inspect a winning record, call
  `hpr_performance_map`, serialize plain JSON, show explicit TESPy selection,
  and include an explicit molar mixture.
- [x] Narrow the notebook import rule so the documented
  `OpenPinch.contracts.hpr_performance_map` module is permitted while other
  internal-owner imports remain prohibited.
- [x] Extend RTD consistency tests to require notebook 09 to be named as the
  comprehensive target-to-map tutorial and require the same workflow concepts.
- [x] Change the map manifest row expectation back to `mapped and executable`
  and run RED to observe only the intended missing source/docs failures.
- [x] Mark Step 1 complete immediately.

### Step 2: Expand the Canonical Notebook 09 Generator

- [x] Modify only the existing notebook 09 definition in
  `scripts/generate_tutorial_notebooks.py`; do not create a duplicate notebook.
- [x] Keep a bounded guarded screening helper and demonstrate one reliable
  default CoolProp target, its detached winning record, design-derived map
  coordinates, a small multi-load map, and plain JSON serialization.
- [x] Demonstrate explicit TESPy targeting as a separate result with the
  supported single-stage one-by-one topology and no implied fallback.
- [x] Include explicit N-component molar-mixture syntax and explain that actual
  feasibility is determined by installed property support and operating state.
- [x] Update notebook 09 question, interpretation, next-step, and presentation
  metadata so the cells form one coherent engineering study.
- [x] Mark Step 2 complete immediately.

### Step 3: Regenerate and Validate the Packaged Notebook

- [x] Run the canonical generator to update
  `OpenPinch/data/notebooks/09_vapour_compression_and_brayton.ipynb`.
- [x] Prove a second generator run is byte-idempotent and changes no unrelated
  notebook.
- [x] Compile every notebook 09 code cell and inspect the generated source for
  the complete target-record-map-export sequence.
- [x] Keep generated notebook metadata/profile consistent with the optional HPR
  tutorial convention.
- [x] Mark Step 3 complete immediately.

### Step 4: Synchronize Tutorial Inventory and RTD

- [x] Restore `problem.target.hpr_performance_map` in
  `docs/_data/tutorial-coverage.csv` to `mapped and executable` with notebook 09
  as its primary tutorial.
- [x] Update `docs/guides/heat-pump-workflows.rst` to point readers to notebook
  09 as the comprehensive executable example and describe its exact scope.
- [x] Update `docs/examples/notebook-series.rst` and
  `docs/examples/tutorial-coverage-map.rst` so the notebook description and
  197-operation coverage statement are accurate.
- [x] Update other HPR reference/capability pages only if a consistency test
  identifies a missing or contradictory claim.
- [x] Mark Step 4 complete immediately.

### Step 5: Run Focused Example, PBT, and Integration Gates

- [x] Run generator idempotence, notebook compilation/source/import,
  public-operation manifest, tutorial coverage, documentation consistency, and
  resource tests with Hypothesis seed `20260715` where applicable.
- [x] Execute notebook 09 under its declared optional profile if the bounded
  real-engine workflow remains within the supported local budget; retain the
  existing isolated real TESPy public smoke as the oracle.
- [x] Run focused HPR target, record, basis, accessor, map-generation, mixture,
  and serialization tests.
- [x] Run Sphinx with warnings as errors, repository Ruff lint, changed-surface
  formatting, Python compilation, and `git diff --check`.
- [x] Fix only scoped regressions and mark Step 5 complete immediately.

### Step 6: Summarize Generated Changes

- [x] Create
  `aidlc-docs/construction/rtd-comprehensive-hpr-notebook/code/code-summary.md`
  with modified files, tutorial narrative, RTD changes, executable evidence,
  PBT compliance, limitations, and OpenUtility boundary.
- [x] Confirm FR-NB-01 through FR-NB-08 and NFR-NB-01 through NFR-NB-06 are
  traceable to implementation and tests.
- [x] Validate all generated Markdown/notebook/manifest content and confirm no
  duplicate brownfield file or enabled-extension finding.
- [x] Update state and audit, present the standardized code review prompt, and
  pause for approval before Build and Test.
- [x] Mark Step 6 complete immediately after the prompt is logged.

## Property-Based Testing Compliance Plan

- PBT-01: generator reproducibility, manifest completeness, serialization, and
  existing target/map physical invariants are identified.
- PBT-02/PBT-03: existing generated map serialization and physical invariants
  remain part of the focused HPR gate.
- PBT-04: generator idempotence remains explicitly tested.
- PBT-05: existing CoolProp and real TESPy public workflow oracles remain in the
  gate.
- PBT-06: N/A for notebook/docs changes; evaluator lifecycle stateful PBT is
  rerun as an integration dependency.
- PBT-07 through PBT-10: reuse domain strategies, seed `20260715`, shrinking,
  Hypothesis/pytest integration, and complementary explicit notebook examples.

## Approval and Quality Gates

- [x] Requirements approved.
- [x] Workflow plan approved.
- [x] This detailed plan contains no Mermaid/ASCII diagram or embedded JSON/YAML;
  headings, paths, code fences, checkboxes, identifiers, and special characters
  were validated before creation.
- [x] Obtain explicit approval of this complete code-generation plan.
- [x] Obtain explicit approval of generated code before Build and Test.
