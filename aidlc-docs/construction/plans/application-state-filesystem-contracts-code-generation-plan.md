# Application State and Filesystem Contracts Code Generation Plan

This document is the single source of truth for Unit 1 Code Generation. Steps
must execute in numerical order, and each checkbox must be marked in the same
interaction in which its work completes.

## Unit Context

- **Unit**: Application State and Filesystem Contracts.
- **Project type**: Brownfield Python package; modify existing owners in place.
- **Workspace root**: `/Users/timothyw/Github_Local/OpenPinch`.
- **Requirements**: FR-1, FR-2, FR-4, FR-5; safety, reliability, state
  consistency, portability, maintainability, dependency, and bounded-overhead
  NFRs.
- **User stories**: N/A; bounded correctness requirements provide traceability.
- **Dependencies**: approved requirements, Application Design, unit artifacts,
  and Unit 1 Functional Design. No other construction unit is required.
- **Downstream consumer**: Unit 3 current documentation and final repository
  gates.
- **Public interfaces retained**: `PinchProblem.problem_data`,
  `PinchProblem.set_dt_cont_multiplier`, `PinchWorkspace` case APIs,
  `CaseBatchResult`, workspace bundle schema version `3`, and Excel export return
  type.
- **Database entities**: none.
- **Frontend components**: none.
- **Deployment artifacts**: none; the existing wheel/source distribution is
  verified after all units.

## Existing Files to Modify

### Application code

- `OpenPinch/contracts/workspace.py`
- `OpenPinch/application/workspace.py`
- `OpenPinch/application/problem.py`
- `OpenPinch/presentation/reporting/workbook.py`

### Tests

- `tests/application/test_pinch_problem.py`
- `tests/application/test_pinch_workspace.py`
- `tests/presentation/test_workbook_reporting.py`

### Unit documentation

- `aidlc-docs/construction/application-state-filesystem-contracts/code/implementation-summary.md`
- `aidlc-docs/aidlc-state.md`
- `aidlc-docs/audit.md`

No duplicate `*_new.py`, `*_modified.py`, compatibility module, migration
script, API layer, repository layer, or deployment file will be created.

## Generation Steps

### Step 1: Confirm the focused baseline and exact ownership

- [x] Re-read the current implementations and focused tests immediately before
  editing.
- [x] Run the focused Unit 1 baseline excluding solver tests.
- [x] Record the baseline result without changing unrelated working-tree files.

**Step 1 evidence**: 128 focused non-solver tests passed in 20.69 seconds. The
pre-existing AI-DLC documentation changes were preserved; no application or test
file changed during the baseline.

### Step 2: Add problem-state regressions first

- [x] Add a mapping-input test proving nested mutation of returned
  `problem_data` cannot change runtime streams or `to_problem_json()`.
- [x] Add a `TargetInput` test proving returned Pydantic/nested state is detached.
- [x] Add an active-workspace delegation test proving the same isolation.
- [x] Add an unloaded multiplier test expecting the canonical `RuntimeError`.
- [x] Add a lazy-rebuild multiplier test preserving successful behavior and
  cache invalidation.
- [x] Run these tests and record their expected pre-fix failures.

**Step 2 evidence**: all five new regressions failed before implementation for
the expected reasons: mutable mapping state, mutable `TargetInput` state, mutable
workspace delegation, unloaded `AttributeError`, and absent lazy rebuilding.

**Traceability**: FR-2, FR-5; state consistency and maintainability NFRs;
acceptance criteria 1, 3, and 6.

### Step 3: Add workspace identity and containment regressions first

- [x] Parameterize rejected runtime identifiers: empty, padded whitespace, dot
  components, both separators, controls, forbidden portable characters,
  trailing period, and Windows device names/extensions.
- [x] Parameterize accepted identifiers including ordinary names, spaces,
  Unicode, internal periods, underscores, and hyphens.
- [x] Verify constructor, `load`, `_set_case_input` through public workflows,
  `scenario`, and batch boundaries share the contract.
- [x] Verify workspace bundles reject invalid `baseline_name` and invalid case
  keys while preserving schema version `3` and valid mappings.
- [x] Add fixed-seed Hypothesis strategies for valid/invalid identifiers and
  retain shrinking under the repository CI seed.
- [x] Add batch tests proving hostile identifiers never reach exporters, valid
  result keys remain unchanged, resolved case directories stay under the root,
  and one case failure remains isolated.
- [x] Run these tests and record their expected pre-fix failures.

**Step 3 evidence**: with Hypothesis seed 20260715, 51 cases failed before the
fix and 7 accepted-name controls passed. Failures cover runtime/bundle rejection,
generated invalid identifiers, corrupted-state export, and symlink containment.

**Traceability**: FR-1; safety, portability, maintainability, and bounded-
overhead NFRs; acceptance criteria 1 and 2.

### Step 4: Add workbook-allocation regressions first

- [x] Freeze time and prove two allocations return distinct paths.
- [x] Run concurrent allocations against one directory and prove all paths are
  unique and reserved.
- [x] Verify filenames retain sanitized project prefix and `.xlsx` suffix.
- [x] Force writer failure and prove the reserved partial path is removed while
  the original exception propagates.
- [x] Verify successful public export still produces a readable workbook with
  existing sheet/content expectations.
- [x] Run these tests and record their expected pre-fix failures.

**Step 4 evidence**: all four selected pre-fix checks failed as expected: old
filename precision, missing reservation helper, concurrent missing helper, and
writer failure replacing/leaving the intended cleanup contract.

**Traceability**: FR-4; reliability, portability, and bounded-overhead NFRs;
acceptance criteria 1 and 5.

### Step 5: Implement the shared canonical case-identifier validator

- [x] Add one concrete validator to `OpenPinch/contracts/workspace.py` without
  exporting it from package-root markers.
- [x] Implement the exact CASE-01 through CASE-05 rules from Functional Design.
- [x] Apply it to bundle `baseline_name` and all `cases` keys with actionable
  Pydantic errors.
- [x] Preserve valid input spelling and reject rather than normalize.
- [x] Run the focused bundle/identifier tests and mark this step complete only
  when they pass.

**Step 5 evidence**: 30 bundle-rejection and accepted-name tests passed. The
validator is concrete-module-only and preserves valid identifier spelling.

### Step 6: Enforce workspace runtime validation and export containment

- [x] Reuse the shared validator at every runtime case-creation/storage boundary.
- [x] Validate `baseline_name` even when the workspace is created without a
  source.
- [x] Replace batch export string interpolation with resolved `Path` composition.
- [x] Enforce `is_relative_to` containment before exporter invocation.
- [x] Preserve original result keys and per-case error isolation.
- [x] Run the focused workspace tests and mark this step complete only when they
  pass.

**Step 6 evidence**: 58 runtime, bundle, fixed-seed generated-name, corrupted-
state, and symlink-containment tests passed.

### Step 7: Isolate problem input and guard multiplier mutation

- [x] Return `deepcopy(self._problem_data)` from `problem_data` and document the
  detached snapshot contract.
- [x] Verify workspace delegation requires no additional copy layer.
- [x] Resolve the prepared root before multiplier zone lookup.
- [x] Preserve numeric warning/default behavior and exact cache invalidation.
- [x] Run focused problem/workspace tests and mark this step complete only when
  they pass.

**Step 7 evidence**: 11 focused snapshot, workspace delegation, unloaded/lazy
multiplier, warning/default, and cache-invalidation tests passed.

### Step 8: Implement exclusive workbook allocation and cleanup

- [x] Replace second-resolution path composition with an exclusive reservation
  helper using standard-library filesystem flags.
- [x] Use readable microsecond timestamp names and deterministic numeric retry
  suffixes while treating `O_EXCL` as the uniqueness authority.
- [x] Close the reservation descriptor before pandas/openpyxl writes.
- [x] Remove the invocation-owned path on every unsuccessful writer exit.
- [x] Preserve successful workbook content, public return type, and normal
  umask-based permissions.
- [x] Run focused reporting tests and mark this step complete only when they
  pass.

**Step 8 evidence**: all 14 workbook reporting tests passed, including frozen-
time collision, 24 concurrent reservations, failure cleanup, and readable
workbook content.

### Step 9: Run integrated Unit 1 verification

- [x] Run all focused problem, workspace, bundle, and workbook tests together.
- [x] Run relevant application usability/closed-contract tests.
- [x] Run Ruff check and format checks on modified Python files.
- [x] Run `git diff --check`.
- [x] Confirm generated property tests use the repository fixed seed in CI and
  retain shrinking.

**Step 9 evidence**: after deterministic Ruff fixes, 203 integrated tests passed
with Hypothesis seed 20260715. Ruff lint, Ruff format check, and patch hygiene
all passed.

### Step 10: Perform structural and regression review

- [x] Confirm only existing brownfield Python modules were modified.
- [x] Confirm no compatibility aliases, sanitizing fallback, schema migration,
  new dependency, root export, duplicate file, database layer, frontend, or
  deployment artifact was introduced.
- [x] Confirm `TargetInput` and serialized HEN payload behavior remains intact.
- [x] Confirm Unit 2 and Unit 3 files remain untouched except shared AI-DLC state
  and audit tracking.

**Step 10 evidence**: source review found only the four planned brownfield owners
and three planned test modules changed. No duplicate/new compatibility files or
dependency/configuration changes exist. Thirty-four HEN serialization, general
round-trip, and root API boundary tests passed.

### Step 11: Generate the implementation summary

- [x] Create the Unit 1 implementation summary under
  `aidlc-docs/construction/application-state-filesystem-contracts/code/`.
- [x] List modified/created files, fulfilled requirements, tests executed, and
  deferred repository-wide gates.
- [x] Mark every completed plan checkbox immediately and update AI-DLC state and
  audit records.
- [x] Present generated code for explicit approval before beginning Unit 2.

**Step 11 evidence**: the implementation summary, state, audit record, and final
approval handoff were generated after all preceding plan checks passed.

## Deferred to Build and Test

The complete fixed-seed non-solver suite, warning-free Sphinx build,
distribution build, isolated wheel smoke, and final stale-symbol gates execute
after all three units. Focused tests and Ruff still run during generation to
prevent later units from building on a broken Unit 1 contract.
