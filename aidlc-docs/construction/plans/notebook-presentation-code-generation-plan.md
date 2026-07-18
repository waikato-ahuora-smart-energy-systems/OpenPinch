# Notebook Presentation Code Generation Plan

## Plan Authority

This checklist is the single source of truth for the Notebook Presentation unit.
Each step must be completed in order and marked `[x]` in the same interaction.

## Unit Context

- **Unit**: Notebook Presentation.
- **Stories**: NB-01 through NB-18.
- **Workspace root**: `/Users/timothyw/Github_Local/OpenPinch`.
- **Project type**: Brownfield Python library.
- **Canonical owner**: `scripts/generate_tutorial_notebooks.py`.
- **Generated resources**: `OpenPinch/data/notebooks/*.ipynb`.
- **Verification owner**: `tests/packaging/test_notebooks.py` and related
  packaging tests.
- **Dependencies**: Existing `PinchProblem`, `PinchWorkspace`, plotting,
  reporting, IPython display, optional HPR, solver, and interactive profiles.
- **Interfaces preserved**: Root public API, notebook inventory, profile metadata,
  source-only nbformat contract, coverage manifest, and package dependencies.
- **Database entities**: None.
- **Service boundaries**: None.

## Implementation Approach

Add canonical, notebook-specific presentation definitions to the generator and
integrate one explicit review section into every tutorial. Each review section
will use IPython display on already-calculated public result objects, avoiding
hidden reanalysis. Existing notebook definitions will retain useful summaries,
rankings, comparisons, or status objects where necessary. Generated notebooks
will remain source-only.

## Execution Checklist

### Step 1: Establish the focused baseline

- [x] Run current notebook structure, compilation, coverage, and base-profile
  execution tests.
- [x] Run generator and repository diff checks to confirm the canonical starting
  state.
- **Stories**: NB-01 through NB-18.

### Step 2: Add regression-first presentation contracts

- [x] Extend `tests/packaging/test_notebooks.py` to require exactly one review
  section and a following explicit display cell in every notebook.
- [x] Require review placement before subject-specific interpretation.
- [x] Add generator repeatability and profile-aware presentation checks.
- [x] Add a domain-specific Hypothesis property over valid tutorial names for
  source-only cells, sequential unique IDs, and review-cell invariants.
- [x] Confirm the new presentation regressions fail against the pre-change
  generator while existing controls continue to pass.
- **Traceability**: FR-02 through FR-10; NFR-02, NFR-03, NFR-05, NFR-06;
  AC-02 through AC-07, AC-09, AC-10, AC-12.

### Step 3: Add canonical presentation definitions

- [x] Modify `scripts/generate_tutorial_notebooks.py` in place with one
  subject-specific review description and display program for each notebook.
- [x] Integrate the review Markdown and code cells into `enrich()` without
  changing the existing step-title contract.
- [x] Keep display cells source-only and limited to cached or already-computed
  public results.
- [x] Make in-process repeated generation deterministic by enriching a detached
  notebook value rather than mutating canonical definitions.
- **Stories**: NB-01 through NB-18.

### Step 4: Retain reviewable workflow results

- [x] Update existing generator code cells only where a comparison, summary,
  ranking, or status result must be retained for the review section.
- [x] Avoid duplicate engineering analysis and new private imports.
- [x] Preserve declared runtimes, optional extras, and operation coverage.
- **Stories**: NB-01 through NB-18.

### Step 5: Regenerate the notebook resources

- [x] Run the canonical generator.
- [x] Verify exactly 18 numbered notebooks are produced.
- [x] Verify every generated code cell has `execution_count: null` and empty
  outputs.
- [x] Run the generator again and confirm byte-identical output.
- **Files**: `OpenPinch/data/notebooks/*.ipynb`.

### Step 6: Run focused base and structural verification

- [x] Run notebook generator and packaging tests with the repository Hypothesis
  seed and shrinking enabled.
- [x] Run all ten base-profile notebook execution tests.
- [x] Verify every code cell compiles, uses only public package imports, and
  executes without user input.
- [x] Inspect representative rendered table and plot objects from review cells.
- **Stories**: NB-01 through NB-07 and NB-12 through NB-14, plus structural
  coverage for NB-08 through NB-11 and NB-15 through NB-18.

### Step 7: Run every optional execution profile

- [x] Execute the four slow-HPR notebooks.
- [x] Execute the three external-solver HEN notebooks.
- [x] Execute the interactive notebook with its existing test-safe dashboard
  boundary.
- [x] Resolve all profile-specific execution or rendering failures without
  weakening the approved acceptance contract.
- **Stories**: NB-08 through NB-11 and NB-15 through NB-18.

### Step 8: Run integrated tutorial and documentation verification

- [x] Run tutorial coverage, packaged resource, documentation consistency,
  release-artifact, and relevant entrypoint tests.
- [x] Regenerate or verify the tutorial coverage manifest only if source changes
  require it.
- [x] Confirm all existing public operations remain demonstrated.
- **Traceability**: FR-01, FR-10; AC-01, AC-07, AC-09, AC-11.

### Step 9: Run repository quality and patch review

- [x] Run Ruff lint and format checks for modified Python files.
- [x] Run Markdown and JSON parsing checks, `git diff --check`, and focused stale
  or private-import scans.
- [x] Review the complete patch for accidental outputs, temporary files,
  duplicate files, dependency changes, or unrelated modifications.
- [x] Confirm partial PBT compliance for PBT-03, PBT-07, PBT-08, and PBT-09;
  document PBT-02 as N/A unless an inverse pair is introduced.

### Step 10: Create implementation summary and handoff

- [x] Create
  `aidlc-docs/construction/notebook-presentation/code/code-generation-summary.md`.
- [x] Record modified and generated files, story and requirement coverage,
  verification evidence, extension compliance, and deferred full-suite gates.
- [x] Update AI-DLC state and audit records.

## Story Completion

- [x] NB-01 through NB-07 - base heat-integration and analysis tutorials.
- [x] NB-08 through NB-11 - slow-HPR tutorials.
- [x] NB-12 through NB-14 - base power and transfer tutorials.
- [x] NB-15 through NB-17 - external-solver HEN tutorials.
- [x] NB-18 - interactive reporting and export tutorial.

## PBT Compliance Plan

| Rule | Planned status |
|---|---|
| PBT-02 Round trips | N/A unless implementation introduces an inverse pair. |
| PBT-03 Invariants | Enforce source-only state, unique sequential IDs, review placement, and deterministic generation. |
| PBT-07 Generator quality | Use a domain strategy sampling canonical valid tutorial identifiers. |
| PBT-08 Shrinking and reproducibility | Retain Hypothesis shrinking and fixed repository seed `20260715`. |
| PBT-09 Framework selection | Retain Hypothesis with pytest integration. |

## Approval

The user approved this complete plan and all remaining gates through completion
with the exact response: `Approve & Continue till completion. `
