# HENS-00 Baseline Freeze and Acceptance Matrix

## PRD Summary

Create the migration safety baseline before any OpenHENS production code moves
into OpenPinch. This task exists to make later numerical regressions visible and
to define which artifacts, fixtures, metrics, and test commands are authoritative.

## User Outcome

Maintainers can tell whether a later OpenHENS-to-OpenPinch change preserved the
scientific behavior of the current solver and did not accidentally change
benchmark results, accepted fixtures, or test scope.

## Scope

- Documentation, test metadata, fixture policy, and baseline artifact capture.
- No production synthesis code move.
- No public API changes except documentation references or test-only metadata.

## Plan Context

Read these sections before implementation:

- [Purpose](../../../OPENHENS_MIGRATION_PLAN.md#purpose)
- [Phase 0: Baseline Freeze and Acceptance Matrix](../../../OPENHENS_MIGRATION_PLAN.md#phase-0-baseline-freeze-and-acceptance-matrix)
- [Validation Strategy](../../../OPENHENS_MIGRATION_PLAN.md#validation-strategy)
- [Regression Tolerances](../../../OPENHENS_MIGRATION_PLAN.md#regression-tolerances)
- [Recommended Review Slices](../../../OPENHENS_MIGRATION_PLAN.md#recommended-review-slices)

Settled decisions for this task:

- Baseline artifacts and acceptance criteria must exist before production solver
  code moves.
- Converted OpenHENS examples become OpenPinch JSON fixtures; source CSVs are
  not runtime synthesis inputs.
- The Four-stream case is the required baseline gate for migrated solver work.
- The Nine-stream case must still be recreated in the expected migrated formats
  and retained for final verification, but it is not the routine baseline gate.
- The required workflow in `README.md` is mandatory; baseline and acceptance
  criteria must reject any public path that bypasses it.
- The tolerance values in the plan are fixed for migration unless a scientific
  review explicitly changes them.
- Dirty-worktree baselines are invalid unless a patch artifact and hash are
  checked in with the baseline manifest.
- HENS-00 must define adapter snapshot and network snapshot artifact locations
  before adapter or solver model movement begins.
- Missing solver binaries are blockers to record, not reasons to weaken tests.

Baseline source files:

- `openhens_baseline_results/comparison.json`
- `openhens_baseline_results/comparison.csv`
- `openhens_baseline_results/refactor/Four-stream-Yee-and-Grossmann-1990-1/summary.json`
- `openhens_baseline_results/refactor/Four-stream-Yee-and-Grossmann-1990-1/run_summary.csv`
- `openhens_baseline_results/refactor/Four-stream-Yee-and-Grossmann-1990-1/solution_metrics.csv`
- `openhens_baseline_results/refactor/Nine-stream-Linnhoff-and-Ahmad-1999-1/summary.json`

Four-stream expected baseline values:

| Metric | Expected value |
| --- | ---: |
| `best_solution` | `154853.8518602861` |
| `quartile_1` | `154853.8601589499` |
| `quartile_2` | `159038.71986626228` |
| `quartile_3` | `163556.12518498677` |
| `solved_esm_count` | `100` |
| `total_cases_attempted` | `1210` |
| `total_cases_solved` | `1210` |
| `within_2_percent` | `37` |
| `within_5_percent` | `64` |
| `within_10_percent` | `100` |
| `best_dTmin` | `14` |
| `best_min_dQ` | `0.5` |
| `best_stages` | `3` |
| `best_recovery_units` | `3` |
| `best_cu_units` | `2` |
| `best_hu_units` | `1` |

Nine-stream final-verification values:

| Metric | Expected value |
| --- | ---: |
| `best_solution` | `2905807.275299348` |
| `quartile_1` | `2947087.026205118` |
| `quartile_2` | `2954844.749379795` |
| `quartile_3` | `2969362.7969760075` |
| `solved_esm_count` | `71` |
| `total_cases_attempted` | `1155` |
| `total_cases_solved` | `886` |
| `within_2_percent` | `46` |
| `within_5_percent` | `68` |
| `within_10_percent` | `70` |
| `best_dTmin` | `18` |
| `best_min_dQ` | `1.7` |
| `best_stages` | `4` |
| `best_recovery_units` | `11` |
| `best_cu_units` | `3` |
| `best_hu_units` | `3` |

## HENS-00 Baseline Manifest

Checked-in manifest:

- `openhens_baseline_results/baseline_manifest.json`

OpenHENS source provenance is recorded as repository commit metadata. The dirty patch artifacts are intentionally not retained in this OpenPinch PR.

Manifest schema:

- `artifact_schema_version`: `openhens-baseline-manifest/v1`
- Artifact generator: `../OpenHENS/openhens/artifacts.py`, SHA256
  `bf415783e20a86061e026f94a14e284e1ceb949f1c1bd8c14f40ead0b5a2bed2`,
  package version `OpenHENS==0.5.0`.
- Artifact root: `openhens_baseline_results`.

Recorded source repository state:

| Repository | URL | Branch / snapshot | Commit or status |
| --- | --- | --- | --- |
| OpenPinch | `https://github.com/waikato-ahuora-smart-energy-systems/OpenPinch.git` | `codex/openhens-migration-plan` | Manifest captured at `e670a1a8ebdd9fe2c90d8ef445af1d69c67aef2e`; root `.DS_Store` and docs-index changes are pre-existing dirty tracked files excluded from the HENS-00 implementation slice. |
| OpenHENS current checkout | `https://github.com/waikato-ahuora-smart-energy-systems/OpenHENS.git` | `codex/refactor-architecture` | Clean at `2afc14b7779482fc829edb1c3fa187b918d7fb19` (`Restore OpenHENS solver parity`). |
| OpenHENS artifact main snapshot | `https://github.com/waikato-ahuora-smart-energy-systems/OpenHENS.git` | Existing artifact metadata | `fbf237bd3bdfbd8cf32698a137f68c0f0db5c1e6`; existing README records that this run used a temporary Python 3.12 environment because legacy main failed under Python 3.14. |
| OpenHENS artifact refactor snapshot | `https://github.com/waikato-ahuora-smart-energy-systems/OpenHENS.git` | Existing artifact metadata | `92e942fec148d5e6a1e052bb3d207b95a4f85379` with `working_tree=dirty` in `openhens_baseline_results/refactor/branch_summary.json`; the later clean parity commit is `2afc14b7779482fc829edb1c3fa187b918d7fb19`. |

Dependency and solver provenance:

| Environment | Evidence |
| --- | --- |
| OpenPinch Python | `Python 3.14.2` from `rtk uv run python --version`. |
| OpenPinch lock | `uv.lock` SHA256 `dc8ccfa6a2aa445deaeb769aa3cab84692f89d6915c69c584a49cb376cadd828`. |
| OpenPinch package versions | `OpenPinch==0.2.3`, `numpy==2.4.6`, `pydantic==2.13.4`, `scipy==1.17.1`, `matplotlib==3.10.9`, `plotly==6.8.0`. |
| OpenHENS Python | `Python 3.14.2` from `rtk uv run python --version` on the current checkout. |
| OpenHENS lock | `../OpenHENS/uv.lock` SHA256 `0b7fddd39ae202d8889462a4910202e5fd88f073051965623bfbc172883998b0`. |
| OpenHENS package versions | `OpenHENS==0.5.0`, `numpy==2.4.4`, `pandas==3.0.3`, `pyomo==6.10.0`, `gekko==1.3.2`, `matplotlib==3.10.9`, `plotly==6.7.0`, `kaleido==1.3.0`, `openpyxl==3.1.5`, `wakepy==1.0.0`. |
| Solver binaries | `rtk which couenne` and `rtk which ipopt` exited 1 with no path. Solver reruns are blocked locally until Couenne and IPOPT are installed and on PATH. |
| OpenHENS dirty snapshot | `openhens_baseline_results/refactor/branch_summary.json` records the dirty artifact-generation snapshot at `92e942fec148d5e6a1e052bb3d207b95a4f85379`; retained provenance now uses commit metadata only. |

Baseline generation commands:

| Purpose | CWD | Command | Current status |
| --- | --- | --- | --- |
| OpenPinch fast tests | OpenPinch | `rtk uv run pytest` | HENS-00 verification command; result recorded in Implementation Notes. |
| OpenPinch docs | OpenPinch | `rtk uv run scripts/build_docs.py` | HENS-00 verification command; result recorded in Implementation Notes. |
| OpenHENS fast tests | `../OpenHENS` | `rtk uv run pytest -m "not solver"` | Passed locally: 56 passed, 2 deselected in 8.09s. |
| OpenHENS solver collection | `../OpenHENS` | `rtk uv run pytest -m solver --collect-only` | Passed locally: selected Four-stream and Nine-stream solver regressions. |
| OpenHENS solver rerun | `../OpenHENS` | `rtk uv run pytest -m solver` | Blocked locally because Couenne and IPOPT are missing from PATH. |

Source case hashes:

| Case | Source CSV path | SHA256 | Role |
| --- | --- | --- | --- |
| `Four-stream-Yee-and-Grossmann-1990-1` | `../OpenHENS/examples/cases/Four-stream-Yee-and-Grossmann-1990-1.csv` | `0c0a7f9592e7d8e27d35aef523530d70c11826c407dcbe4adba7fff86f4835f2` | Routine solver baseline. |
| `Nine-stream-Linnhoff-and-Ahmad-1999-1` | `../OpenHENS/examples/cases/Nine-stream-Linnhoff-and-Ahmad-1999-1.csv` | `fbf065509f355584623bfdfd5c38478410ddbab2a7c696e49bb3c15df3161b10` | Final-verification solver baseline. |

Generated artifact paths:

| Case | Expected paths |
| --- | --- |
| Four-stream | `openhens_baseline_results/refactor/Four-stream-Yee-and-Grossmann-1990-1/summary.json`, `run_summary.csv`, `solution_metrics.csv`, and the curated `network_snapshots/Four-stream-Yee-and-Grossmann-1990-1/best-esm.json`. |
| Nine-stream | `openhens_baseline_results/refactor/Nine-stream-Linnhoff-and-Ahmad-1999-1/summary.json`, `run_summary.csv`, `solution_metrics.csv`, and the curated `network_snapshots/Nine-stream-Linnhoff-and-Ahmad-1999-1/best-esm.json`. |
| Cross-branch parity | `openhens_baseline_results/comparison.json`, `openhens_baseline_results/comparison.csv`, and `openhens_baseline_results/refactor/branch_summary.json`. |

## Fixture And Artifact Policy

- Converted OpenHENS examples must become OpenPinch-compatible JSON fixtures
  under `tests/fixtures/openhens/<case-id>.json`.
- Source OpenHENS CSV files are source material only. They may be read by a
  one-time conversion script or test-data preparation note, but runtime HEN
  synthesis must not accept OpenHENS CSVs or expose a CSV synthesis API.
- The public fixture execution path must be:
  `TargetInput` / JSON -> `PinchProblem` -> `PinchProblem.load(...)` ->
  `prepare_problem(...)` -> prepared `Zone` and `StreamCollection` -> internal
  adapter/service. Tests must fail if arrays are built directly from converted
  fixture rows, raw `TargetInput`, or a HEN result schema.
- Compact OpenHENS summary CSV/JSON files under
  `openhens_baseline_results/refactor/<case-id>/` are migration reference
  artifacts and test evidence only. Raw generated solver traces, pickles,
  workbooks, plots, and timestamped per-run JSON dumps are excluded from the
  branch and must be regenerated outside version control when needed.
- Optional migrated exports must be generated from `problem.results` /
  `TargetOutput.design`, identify OpenPinch problem or workspace variant
  identity, and stay outside the core in-memory workflow.

## Structural Snapshot Policy

HENS-03 must create structural fixture snapshots for both Tier 0 cases:

- `openhens_baseline_results/fixture_snapshots/Four-stream-Yee-and-Grossmann-1990-1.json`
- `openhens_baseline_results/fixture_snapshots/Nine-stream-Linnhoff-and-Ahmad-1999-1.json`

Each fixture snapshot must include:

- source CSV path and SHA256,
- migrated JSON fixture path and SHA256,
- process stream count,
- utility count,
- economic/costing fields mapped into OpenPinch schemas/configuration,
- design-grid task counts for PDM, TDM, and ESM,
- private adapter array shapes,
- hot/cold/utility axis maps keyed by stable OpenPinch stream identities,
- covered `dTmin` values,
- `PinchProblem.load(...)` and `prepare_problem(...)` success evidence.

Order-invariance fixtures:

- `tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.reordered.json`
- `tests/fixtures/openhens/Nine-stream-Linnhoff-and-Ahmad-1999-1.reordered.json`

Order-invariance tests must prove that reordered stream rows keep the same
OpenPinch stream identities, target values, adapter axis-map semantics, and
future exchanger-link expectations. Private solver axis positions may move, but
labelled network results must not.

## Adapter And Network Snapshot Policy

Adapter snapshot artifact path:

- `openhens_baseline_results/adapter_snapshots/<case-id>/dTmin-<value>.json`

Minimum adapter coverage:

- Routine baseline: Four-stream at `dTmin=14`.
- Full pinch/grid parity coverage: Four-stream over source `dTmin` grid
  `[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]`.
- Nine-stream: required when a task touches grid generation, pinch parity, or
  adapter semantics; final-verification coverage includes `dTmin=18`.

Adapter snapshots must include:

- array shapes,
- axis maps,
- unit conventions,
- source stream identities,
- utility identities,
- active `dTmin`,
- private numeric arrays needed by the moved equation models,
- source fixture and OpenPinch preparation evidence.

Curated best-ESM network snapshot paths:

- `openhens_baseline_results/network_snapshots/Four-stream-Yee-and-Grossmann-1990-1/best-esm.json`
- `openhens_baseline_results/network_snapshots/Nine-stream-Linnhoff-and-Ahmad-1999-1/best-esm.json`

Network snapshots must include:

- source artifact path,
- run id and task id,
- method,
- source/sink stream identities,
- exchanger kind,
- stage,
- duty,
- area,
- utility loads,
- TAC,
- units,
- verification-failure status,
- per-field numeric tolerances.

The default comparison set is the single best ESM result. A task that compares
more than that must explicitly choose either top `best_solutions_to_save` rows
or all solved ESM rows.

## Migration Acceptance Matrix

| Task | Invariant later work must preserve | Required proof |
| --- | --- | --- |
| HENS-01 Dependency and runtime viability | Core `import OpenPinch` remains light; synthesis dependencies and solver binaries are optional/lazy. | OpenPinch fast tests, import smoke tests, packaging metadata tests, and documented missing-solver errors. |
| HENS-02 Synthesis schemas and network domain | No public `OpenHENS`, `CaseStudy`, `SynthesisStudy`, OpenHENS field aliases, or raw-input workflow roots are added. Network links use OpenPinch identities. | Schema/domain tests, negative public API tests, export snapshots, and order-invariance label-access tests. |
| HENS-03 JSON fixture conversion and adapter | Both Tier 0 source CSVs are converted once into OpenPinch JSON; runtime synthesis does not read CSV; adapter arrays derive from a prepared `PinchProblem`. | Fixture validation tests, structural snapshots, adapter snapshots at required `dTmin`, and direct-array-bypass negative tests. |
| HENS-04 Pinch target parity | OpenPinch `ProblemTable` / targeting parity supplies pinch decomposition data without changing source behavior. | Full-grid Four-stream parity and any touched Nine-stream grid coverage for targets, active masks, and structural PDM fields. |
| HENS-05 Workflow, result cache, fake executor | HEN workflow starts from `PinchProblem.design` or `PinchWorkspace`; `TargetOutput.design` is canonical; exports are optional views from `problem.results`. | Fake-executor workflow tests, workspace dispatch tests, result-cache tests, and artifact export tests. |
| HENS-06 Equation kernel base move | Solver defaults, objective formulas, tolerances, variable names, and optional dependency boundaries stay behavior-preserving. | Focused model tests, import guards, Four-stream routine solver gate when binaries are available, and no eager solver imports. |
| HENS-07 Stagewise and PDM model move | PDM/TDM/ESM fan-out and parent/warm-start context remain equivalent to OpenHENS. | Task graph tests, parent-problem propagation tests, Four-stream solver gate, and selected adapter/network snapshots. |
| HENS-08 Stage reduction and topology evolution | Stage reduction and evolution heuristics remain behavior-preserving; network extraction uses labelled OpenPinch links. | Four-stream solver gate, best-ESM network snapshot comparison, heat/area/load tolerances, and row-order invariance tests. |
| HENS-09 Public service and documentation | Public docs show only the OpenPinch-owned workflow and no compatibility/bypass surface. | Public API tests, docs examples, docs build, and negative checks for OpenHENS import/path/alias wording. |
| HENS-10 Duplicate helper replacement | Replacing duplicate LMTD, costing, or pinch helpers causes no TAC, topology, utility, or artifact drift. | Focused helper parity tests plus affected-case fixture, adapter, or network snapshot comparisons. |
| HENS-11 Regression expansion and retirement | Additional cases are explicitly named with source paths, hashes, grids, thresholds, and commands before implementation. | Updated README matrix, regenerated manifest, tiered solver tests, and final Nine-stream verification before retirement. |

## Requirements Checklist

- [x] Record the exact source OpenHENS commit, OpenPinch commit, branch names,
      Python versions, solver binary versions, and dependency lock state used
      for the baseline.
- [x] Add a checked-in baseline manifest with repository URL, clean commit SHA
      or stored dirty patch hash, exact generation commands, Python version,
      dependency lock hash, solver binary versions, source case paths and
      hashes, artifact-generation script/version, and artifact schema version.
- [x] Mark any existing dirty-worktree baseline as non-reproducible until its
      patch artifact is checked in.
- [x] Record baseline commands for OpenPinch fast tests and docs build.
- [x] Record baseline commands for OpenHENS fast tests and marked solver tests.
- [x] Document any pre-existing failing tests before code movement starts.
- [x] Document missing solver binaries separately from test failures.
- [x] Record Four-stream workbook baseline metrics:
      best ESM TAC, quartiles, solved counts, within-2/5/10 percent counts,
      attempted jobs, solved ESM count, best stage count, recovery unit count,
      hot utility unit count, cold utility unit count, best `dTmin`, and best
      derivative threshold, using the expected values listed above.
- [x] Record Nine-stream workbook metrics as final-verification expectations,
      not as the routine baseline gate for every solver-moving PR.
- [x] Save generated JSON/CSV solver artifacts as migration reference artifacts
      or document why they cannot be checked in.
- [x] Define fixture policy for converted OpenHENS examples:
      OpenPinch-compatible JSON fixtures become migrated test inputs; source
      CSVs are source material, not runtime API.
- [x] Define which large workbooks, pickles, plots, and generated artifacts are
      test-only and which are curated sample outputs.
- [x] Add or document structural snapshots for every converted example planned
      for migration: stream count, utility count, economics fields, design-grid
      task counts, private array shapes, axis maps, and `PinchProblem`
      preparation success.
- [x] Add or document at least one order-invariance fixture where stream rows
      are reordered but expected OpenPinch stream identities, target values, and
      future exchanger-link expectations remain stable.
- [x] Add a migration acceptance matrix that lists every invariant later tasks
      must preserve and the command or fixture that proves it.
- [x] Expand the README case acceptance matrix with exact source paths, source
      hashes, tier, required `dTmin` values, required `min_dQ` / derivative
      thresholds, and the command or fixture proving each gate.
- [x] Define adapter snapshot artifact paths and required `dTmin` coverage.
- [x] Define curated best-ESM network snapshot artifact paths and the exact
      fields/tolerances they must contain.
- [x] Include the strict tolerance policy from the migration plan in the matrix.

## General Standards That Apply

- [x] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [x] Do not add public OpenHENS compatibility surfaces.
- [x] Do not accept runtime CSV ingestion as a synthesis workflow.
- [x] Do not widen numerical tolerances in this task.
- [x] Do not move GEKKO/Pyomo model code in this task.
- [x] Keep baseline artifacts keyed by OpenPinch problem or workspace identity
      once they become OpenPinch-owned.

## Verification Checklist

- [x] `rtk uv run pytest` passes for OpenPinch or every failure is documented
      as pre-existing.
- [x] `rtk uv run scripts/build_docs.py` passes for OpenPinch or every failure
      is documented as pre-existing.
- [x] OpenHENS fast tests pass on the recorded source checkout or every failure
      is documented as pre-existing.
- [x] OpenHENS solver tests have been run on a machine with Couenne/IPOPT
      available, or the exact blocker and rerun command are documented.
- [x] The acceptance matrix is reviewed against `OPENHENS_MIGRATION_PLAN.md`
      and covers phases 1 through 11.

## Definition of Done

- [x] A reviewer can reproduce the baseline commands and identify the exact
      source commits used.
- [x] Four-stream benchmark metrics are captured as the primary solver baseline
      with the required expected values.
- [x] Nine-stream benchmark metrics are captured as final-verification expected
      values.
- [x] Fixture and artifact ownership is explicit enough that later PRs do not
      debate CSV runtime support or workbook inclusion.
- [x] The acceptance matrix states precise pass/fail criteria for later parity,
      adapter, workflow, solver, export, and public API tasks.
- [x] The baseline manifest is reproducible from clean commits or stored dirty
      patches, not only aggregate metric files.
- [x] Adapter snapshot and best-ESM network snapshot requirements are concrete
      enough that later agents do not choose their own artifact shapes.
- [x] No production synthesis code moved as part of this task.

## Out of Scope

- Moving OpenHENS solver code.
- Adding the OpenPinch HEN public API.
- Adding optional synthesis dependencies.
- Replacing OpenHENS helper algorithms.

## Implementation Notes

- 2026-06-16: Added `openhens_baseline_results/baseline_manifest.json`
  with schema version `openhens-baseline-manifest/v1`, repository URLs,
  source commits, dependency lock hashes, Python/package versions, source case
  paths and hashes, artifact-generator hash, expected artifact paths, baseline
  metrics, and canonical regeneration commands.
- 2026-06-16: Existing generated artifact provenance records
  `working_tree=dirty` for the refactor artifact snapshot at
  `92e942fec148d5e6a1e052bb3d207b95a4f85379`; retained provenance now records
  that dirty snapshot plus the later clean parity commit
  `2afc14b7779482fc829edb1c3fa187b918d7fb19` without storing dirty patch
  artifacts in this OpenPinch PR.
- 2026-06-16: Updated
  `openhens_baseline_results/refactor/branch_summary.json` so the recorded
  dirty refactor snapshot records the later clean target head while omitting
  patch artifacts from this OpenPinch PR.
- 2026-06-16: Root `.DS_Store` remains a modified tracked binary file, but it
  was already present in the dirty worktree before this HENS-00 implementation
  slice and is unrelated to the baseline metadata. It is excluded from the
  HENS-00 task scope and was not reverted or staged by this slice. Current
  hygiene evidence: `rtk git diff --numstat .DS_Store docs/developer/index.rst`
  reports binary changes for `.DS_Store` and one line in the pre-existing docs
  index change; neither file is part of the HENS-00 implementation scope.
- 2026-06-16: `rtk uv run pytest -m "not solver"` in
  `/Users/ca107/Desktop/ahuora/OpenHENS` passed with 56 passed, 2 deselected in
  8.09s.
- 2026-06-16: `rtk uv run pytest -m solver --collect-only` in
  `/Users/ca107/Desktop/ahuora/OpenHENS` selected exactly the Four-stream and
  Nine-stream solver regression tests.
- 2026-06-16: `rtk uv run pytest` in
  `/Users/ca107/Desktop/ahuora/OpenPinch` passed with 954 passed, 31 warnings in
  204.39s. No failing OpenPinch tests were found before code movement.
- 2026-06-16: `rtk uv run scripts/build_docs.py` in
  `/Users/ca107/Desktop/ahuora/OpenPinch` passed; Sphinx reported 25 warnings
  for pre-existing missing autodoc import targets / `_static` path, but exited
  0 and wrote HTML to `docs/_build/html`.
- 2026-06-16: OpenHENS solver rerun command is
  `rtk uv run pytest -m solver`; local rerun is blocked because
  `rtk which couenne` and `rtk which ipopt` both exited 1 with no solver path.
- 2026-06-16: Expanded `docs/developer/openhens-integration-tasks/README.md`
  case acceptance matrix with Tier 0 source paths, SHA256 hashes, fixture
  paths, routine/final gate roles, `dTmin` grids, derivative threshold grids,
  and proof command/artifact requirements.
- 2026-06-16: Defined converted fixture policy, test-only generated artifact
  ownership, structural snapshot paths, order-invariance fixture paths, adapter
  snapshot paths and coverage, curated best-ESM network snapshot paths and
  required fields/tolerances, and a phase-by-phase migration acceptance matrix.
