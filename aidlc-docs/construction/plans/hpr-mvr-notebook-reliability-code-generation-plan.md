# HPR and MVR Notebook Reliability Code Generation Plan

This document is the single source of truth for Code Generation. Execute the
steps in order and mark each checkbox complete in the same interaction in which
its work is finished.

## Unit Context

- **Unit**: HPR and MVR Notebook Reliability.
- **Project type**: brownfield Python package.
- **Workspace root**: `/Users/timothyw/Github_Local/OpenPinch`.
- **Stories**: N/A. User Stories were explicitly skipped; traceability is to
  approved functional requirements FR-1 through FR-11 and the NFR identifiers.
- **Primary owner**: `scripts/generate_tutorial_notebooks.py`.
- **Generated artifacts**: packaged notebooks 08, 09, 10, and 11.
- **Public dependencies**: `PinchProblem` target/component accessors,
  `HPRTargetingError`, shared multiperiod `hpr_details`, direct process-MVR
  results, and HPR performance-map contracts.
- **External dependencies**: CoolProp is required by the HPR extra; TESPy stays
  optional and must never become a fallback.
- **Database entities**: none.
- **API changes**: none planned.
- **Infrastructure/deployment artifacts**: none.

## Exact Application and Test Paths

### Modify

- `scripts/generate_tutorial_notebooks.py`
- `OpenPinch/tutorials/notebooks/08_carnot_heat_pump_and_refrigeration.ipynb`
- `OpenPinch/tutorials/notebooks/09_vapour_compression_and_brayton.ipynb`
- `OpenPinch/tutorials/notebooks/10_multiperiod_heat_pumps.ipynb`
- `OpenPinch/tutorials/notebooks/11_process_mvr_and_cascade.ipynb`
- `tests/packaging/test_notebooks.py`
- `tests/strategies/hpr_targeting.py` if a reusable bounded failure-summary
  strategy improves the property tests
- `scripts/generate_tutorial_coverage.py` only where execution-evidence wording
  or Notebook 10 mapping must change
- `docs/_data/tutorial-coverage.csv` through its generator when changed
- `docs/examples/notebook-series.rst`
- `docs/guides/heat-pump-workflows.rst`
- `tests/packaging/test_docs_consistency.py` and
  `tests/packaging/test_tutorial_coverage.py` where synchronized documentation
  assertions require updates

### Create only if needed for focused separation

- `tests/packaging/test_hpr_notebook_properties.py`

No duplicate generator, notebook, public helper, runtime package, or
`*_modified`/`*_new` file is permitted.

## Expected Interfaces and Contracts

- Required targets expose `hpr_success`, `hpr_load`, and `hpr_details`.
- Shared-design results expose finite `design_vector`, aligned `period_ids` and
  `period_weights`, a `period_outputs` mapping, and a finite objective/weighted
  result.
- Typed failures expose `HPRTargetingError.diagnostics`, serialized through
  `model_dump(mode="json")`.
- Direct process MVR exposes per-period stage results and compressor work.
- Generated notebooks remain valid notebook v4 JSON with null execution counts
  and empty output lists.

## Detailed Generation Steps

- [x] **Step 1: Protect and baseline the brownfield state** — record the exact
  working-tree status and Notebook 10 diff before editing; confirm that the
  executed Notebook 10 is the approved artifact to replace during regeneration;
  run the fastest existing generator/static notebook checks; do not alter
  unrelated user changes. Traceability: FR-1, NFR-DET-001, NFR-DET-002.

- [x] **Step 2: Add failing static and property contract tests** — modify
  `tests/packaging/test_notebooks.py` to assert explicit caps and topology,
  required-versus-optional call boundaries, one scalar shared call per Notebook
  10 technology, no shared call through `target.all_periods`, compact review
  variables, and one pytest case per slow-HPR notebook. Add
  `tests/packaging/test_hpr_notebook_properties.py` only if the pure helper
  properties would make the existing file materially less readable; extend
  `tests/strategies/hpr_targeting.py` only for a genuinely reusable bounded
  strategy. Traceability: FR-2 through FR-11; NFR-PERF-001 through
  NFR-TEST-002; PBT-01 through PBT-10.

- [x] **Step 3: Add generator-owned plain helper source** — modify
  `scripts/generate_tutorial_notebooks.py` in place with the smallest reusable
  source fragments needed for compact successful-target summaries, JSON-mode
  typed diagnostics, and distinct optional statuses. Keep helper code
  notebook-local at execution time, catch no broad `Exception`, retain public
  imports only, and introduce no runtime OpenPinch helper. Traceability: FR-7,
  FR-9; NFR-REL-004, NFR-REL-005, NFR-DIAG-001 through NFR-DIAG-003.

- [x] **Step 4: Correct Notebook 08 at the generator** — add explicit one-stage
  topology and 1/20/50 work limits to both Carnot calls, assert successful and
  meaningful target evidence, preserve all residual utility placement,
  derivation, summaries, and plots, and replace any oversized review payload
  with compact evidence. Traceability: FR-2, FR-3, FR-7; NFR-PERF-001,
  NFR-REL-001, NFR-REL-003.

- [x] **Step 5: Correct Notebook 09 at the generator** — call required CoolProp
  water heat pumping and ammonia refrigeration directly with explicit one-stage
  1/20/50 settings; retain and validate the detached winning record and
  performance map; limit the optional adapter to TESPy and Brayton; distinguish
  missing dependency, unavailable method, and typed infeasibility; and display
  only compact plain comparisons. Traceability: FR-2, FR-4, FR-7, FR-9;
  NFR-REL-001, NFR-REL-003 through NFR-REL-005.

- [x] **Step 6: Rebuild Notebook 10 at the generator** — create a fresh problem
  per technology; run bounded scalar shared-design optimizations for Carnot heat
  pumping, Carnot refrigeration, CoolProp VC heat pumping, CoolProp VC
  refrigeration, and VC+MVR with reporting period `base`; prove the exact
  `turndown`/`base`/`peak` set, aligned weights, finite shared vector and
  objective, and successful period outputs; explain that the five results are
  separate optimizations; add one fresh, explicitly complex, more tightly
  bounded optional cascade screen that catches only `HPRTargetingError` and
  exposes bounded JSON diagnostics. Traceability: FR-5 through FR-9;
  NFR-PERF-001 through NFR-PERF-004, NFR-REL-002, NFR-DIAG-001 through
  NFR-DIAG-003.

- [x] **Step 7: Refine Notebook 11 at the generator** — preserve the direct
  process-MVR component, stage/work evidence, serial/parallel equivalence, and
  lifecycle operations; make VC and MVR fluids and the 1-by-1-plus-1-stage
  topology explicit; cap the optimized VC+MVR target at one restart, no more
  than 20 iterations, and no more than 50 evaluations; remove catch-and-continue
  behavior from this required solve; and present compact stage/loop evidence.
  Traceability: FR-2, FR-7, FR-10; NFR-PERF-001, NFR-PERF-002, NFR-REL-001,
  NFR-REL-003.

- [x] **Step 8: Regenerate and validate the four canonical notebooks** — run the
  authoritative generator after Steps 3 through 7, allowing the approved
  replacement of Notebook 10's executed local artifact; verify notebooks 08
  through 11 are source-only, valid nbformat documents, compile cell-by-cell,
  use only allowed public imports, and match a second isolated generation pass
  byte-for-byte. Traceability: FR-1, FR-11; NFR-DET-001, NFR-DET-002,
  NFR-MAINT-001.

- [x] **Step 9: Synchronize tutorial metadata and guidance** — update
  `scripts/generate_tutorial_coverage.py` and regenerate
  `docs/_data/tutorial-coverage.csv` only where actual execution evidence
  changed; update `docs/examples/notebook-series.rst` and
  `docs/guides/heat-pump-workflows.rst` so shared installed-design optimization
  is distinct from independent `target.all_periods` replay; adjust focused docs
  and coverage assertions without weakening catalog completeness. Traceability:
  FR-3 through FR-6, FR-10; NFR-MAINT-002.

- [x] **Step 10: Make slow-HPR execution attributable** — factor the existing
  execution helper in `tests/packaging/test_notebooks.py` and parameterize
  notebook names so 08, 09, 10, and 11 are four distinct pytest cases while
  solver and interactive profiles retain their current opt-in behavior. Include
  notebook name and cell index in failures and do not override notebook
  targeting arguments. Traceability: NFR-TEST-001, NFR-TEST-002.

- [x] **Step 11: Complete property and example oracles** — exercise pure helper
  definitions without invoking a real engine for each generated example;
  verify bounded diagnostic JSON round trips, status vocabulary, finite compact
  values, period-set preservation, and normalization idempotence if applicable;
  retain real example execution for all thermodynamic claims and use seed
  `20260715` where supported. Traceability: NFR-DET-003; PBT-01 through PBT-10.

- [x] **Step 12: Run implementation checkpoints and refine within scope** — run
  fast static/generator/property tests, then execute notebooks 08, 09, 10, and
  11 individually. Record elapsed times as evidence, not hard assertions. If a
  required solve fails, correct tutorial arguments or presentation within the
  approved bounds; if a separate production defect is exposed, stop and report
  it rather than silently expanding into runtime changes. Traceability: all
  functional and NFR requirements.

- [x] **Step 13: Run focused integration and quality checks** — run the combined
  slow-HPR profile, related HPR/MVR tests, notebook packaging, tutorial coverage,
  documentation consistency, Ruff on changed Python, and patch hygiene. Correct
  scoped defects before Code Generation completion; leave the full repository
  and distribution matrix to Build and Test. Traceability: Acceptance Criteria
  1 through 13; NFR-TEST-002.

- [x] **Step 14: Audit and summarize generated code** — inspect the final diff
  for generator/notebook synchronization, compact output, explicit caps,
  absence of duplicate/private/runtime additions, preservation of unrelated
  changes, and enabled-extension compliance; create
  `aidlc-docs/construction/hpr-mvr-notebook-reliability/code/code-generation-summary.md`;
  update state and audit records. Traceability: all approved requirements.

- [x] **Step 15: Update and validate Read the Docs** — extend the HPR/MVR
  fundamentals, public API reference, capability matrix, support boundary, and
  release notes where needed so the bounded required-versus-optional behavior
  and independent-replay-versus-shared-design distinction are discoverable
  outside the notebook guide. Add focused consistency assertions, run the
  warnings-as-errors Sphinx build, and refresh the code-generation summary and
  audit. Traceability: FR-2 through FR-11; NFR-DIAG-001 through NFR-MAINT-002.

## Planned Test Evidence

- Static notebook contract tests for all four sources.
- Generator repeatability, source-only, drift, nbformat, compile, and import
  checks.
- Property-based plain-summary and diagnostic serialization invariants.
- Four independently reported real slow-HPR notebook executions.
- Combined slow-HPR execution profile.
- Focused HPR/MVR, performance-map, and multiperiod regression tests.
- Tutorial coverage and documentation consistency tests.
- Ruff and `git diff --check`.
- Broader repository, documentation build, distribution, and installed-wheel
  evidence deferred to the approved Build and Test stage.

## Completion Boundaries

- No production API or optimizer implementation is changed unless a newly
  discovered defect is reported and separately authorized.
- No user-owned file outside the named scope is modified.
- No commit, push, pull request, publication, dependency installation, or
  external mutation is part of this plan.
- Code Generation completes only when all fourteen checkboxes are marked and
  the focused evidence is green.

## Extension Compliance

- **Property-Based Testing**: enabled and mapped to Steps 2, 3, 8, 11, 12, and
  13 with real example tests retained.
- **Security Baseline**: disabled; N/A.
- **Resiliency Baseline**: disabled; N/A.
