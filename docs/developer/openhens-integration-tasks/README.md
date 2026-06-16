# OpenHENS Integration PRD Task Set

These task files break `OPENHENS_MIGRATION_PLAN.md` into implementation-ready
work items. Each task is intended to be owned by one implementation agent or one
reviewable PR slice. Agents should tick checkboxes only after the code, tests,
and evidence for that item are complete.

These task files are the operative implementation contract for delegated
migration work. Where the original plan uses exploratory language or contains a
less strict suggestion, the stricter requirement in these task files controls
until the plan and task files are updated together in a reviewed PR.

## How Agents Should Use These Tasks

- [ ] Read `OPENHENS_MIGRATION_PLAN.md` before starting the task.
- [ ] Read the `Plan Context` section in the assigned task and use those
      linked plan sections as the source of truth for ambiguous details.
- [ ] Read this index and the assigned task file before editing code.
- [ ] Treat `Settled decisions for this task` as non-negotiable unless the
      migration plan is updated in the same PR and reviewed.
- [ ] Keep task checkboxes in the task file current as work progresses.
- [ ] Add short evidence notes under the task file's `Implementation Notes`
      section when a checkbox depends on a command, fixture, or decision.
- [ ] Do not mark the task's definition of done complete until every required
      verification item has either passed or is explicitly documented as blocked.
- [ ] Keep unrelated repository changes out of the task PR.

## Global Migration Standards

Every task in this set must preserve these standards unless the plan is updated
and reviewed:

- [ ] Baselines first, movement second. Do not move solver behavior before the
      relevant baseline, fixture, or parity gate exists.
- [ ] `PinchProblem` is the only public owner of one prepared problem's stream,
      utility, zone, targeting, and HEN design state.
- [ ] `PinchWorkspace` is the only public owner of multi-case, variant,
      comparison, and workflow execution state.
- [ ] HEN synthesis enters through `PinchProblem.design` or
      `PinchWorkspace.solve_variant(..., workflow="heat_exchanger_network_synthesis")`.
- [ ] `heat_exchanger_network_synthesis_service(problem)` is an internal
      service boundary behind `PinchProblem.design` and workspace dispatch. Do
      not root-export it or document it as a user-facing execution path.
- [ ] Do not add public `OpenHENS`, `CaseStudy`, `SynthesisStudy`,
      `run_synthesis_workflow(...)`, raw-stream synthesis, HEN-specific
      workspace, or import-path shim APIs.
- [ ] Do not add public option-owner objects such as
      `HeatExchangerNetworkDesignSpace`,
      `HeatExchangerNetworkMethodSequence`,
      `HeatExchangerNetworkSolveSetup`, or
      `HeatExchangerNetworkOutputs`. If records with those concepts exist, they
      are internal/serialization-only and cannot be accepted by public APIs as
      configuration owners.
- [ ] Do not accept OpenHENS field aliases. Any external JSON alias must be
      serialization-only, must not be an OpenHENS compatibility alias, and
      requires a named non-OpenHENS consumer plus a reviewed plan/task update.
- [ ] Persistent HEN controls live in `TargetInput.options` and
      `Configuration` / `CONFIG_FIELD_SPECS`, not in an alternate case or study
      object.
- [ ] Stream and utility payloads live in `StreamSchema` and `UtilitySchema`;
      utility economics use `UtilitySchema.price`; shared capital-cost inputs
      use existing or generalized OpenPinch costing configuration.
- [ ] Existing OpenHENS CSV examples are migration source material only. Runtime
      HEN synthesis must use OpenPinch-compatible JSON, `TargetInput`, or an
      existing `PinchProblem`, all flowing through `PinchProblem.load(...)` and
      `prepare_problem(...)`.
- [ ] `PinchProblem._results` / `problem.results` is the canonical result store.
      `TargetOutput` carries existing `targets` plus optional `design` data.
- [ ] JSON, CSV, Excel, plots, and manifests are optional exports generated
      from `problem.results`; they are not the terminal output of the core
      workflow.
- [ ] Private solver arrays may exist only behind the synthesis service adapter
      while moved equations still require arrays.
- [ ] Public HEN result access uses OpenPinch-native stream identities,
      `HeatExchangerNetwork`, `HeatExchanger`, and labelled accessors, not raw
      solver axis positions.
- [ ] GEKKO, Pyomo, solver binaries, plotting stacks, and workbook/export tools
      stay optional and lazily imported. `import OpenPinch` must remain light.
- [ ] Move equations behavior-preservingly first. Changes to solver defaults,
      tolerances, objective formulas, evolution heuristics, or scientific
      behavior are separate reviewed work.
- [ ] Regression tolerances begin at OpenHENS' current tolerances:
      `TAC_REL_TOL = 1e-4`, `TAC_ABS_TOL = 1.0`,
      `MAX_REGRESSION_REL_TOL = 1e-2`, with one-count allowance only for tied
      threshold buckets.

## Required OpenPinch Workflow

This workflow is mandatory. A task implementation that adds a public path around
this workflow is non-compliant, even if its tests pass locally.

```text
OpenPinch-compatible JSON, TargetInput, or existing PinchProblem source
  -> PinchProblem(source=JSON path, TargetInput, or OpenPinch-native payload)
  -> PinchProblem.load(...) validation and canonicalization
  -> prepare_problem(...) through PinchProblem's execution path
  -> prepared Zone with StreamCollection-backed streams/utilities
  -> PinchProblem.design.heat_exchanger_network_synthesis(...)
  -> internal heat_exchanger_network_synthesis_service(problem)
  -> HEN configuration read from TargetInput.options / prepared Configuration
  -> ProblemTable/OpenPinch targeting parity path supplies pinch decomposition data
  -> private problem_to_solver_arrays(...) only while moved equations need arrays
  -> PDM tasks over dTmin from problem-rooted settings
  -> TDM tasks from successful PDM HeatExchangerNetwork topology
  -> ESM refinement tasks from successful TDM HeatExchangerNetwork topology
  -> extract HeatExchangerNetworkSynthesisResult
  -> expose HeatExchangerNetwork of HeatExchanger source/sink links
  -> verify with OpenPinch streams, labelled network data, and private arrays only where still necessary
  -> update PinchProblem._results as TargetOutput(..., targets=[...], design=...)
  -> optional JSON/CSV exports generated from problem.results when requested
```

Mandatory workflow constraints:

- [ ] Public synthesis starts from `PinchProblem` or `PinchWorkspace`, never from
      source CSV rows, raw stream lists, raw utility lists, a public case/study
      object, or a standalone synthesis runner.
- [ ] The internal synthesis service is not a public or root-exported execution
      path. It exists so `PinchProblem.design` and workspace dispatch have one
      implementation boundary.
- [ ] `PinchWorkspace.solve_variant(..., workflow="heat_exchanger_network_synthesis")`
      dispatches to the active variant's `PinchProblem.design` path.
- [ ] `HeatExchangerNetworkSynthesis` may exist only as a thin runner bound to a
      live `PinchProblem`; it must not own streams, utilities, variants, cases,
      workspace state, or persistent options.
- [ ] `TargetInput.options` / `Configuration` owns persistent HEN controls.
- [ ] `TargetOutput.design` inside `PinchProblem._results` owns the in-memory
      HEN result.
- [ ] Optional artifacts identify the run by OpenPinch problem/workspace variant
      identity and are generated from `problem.results`; artifacts are not the
      core workflow output.

## Benchmark Policy

Use `openhens_baseline_results/` as the source of expected benchmark values.

- [ ] Use the Four-stream case as the key solver baseline because it is the
      quicker benchmark:
      `openhens_baseline_results/refactor/Four-stream-Yee-and-Grossmann-1990-1/summary.json`.
- [ ] Recreate both Four-stream and Nine-stream examples in the migrated
      OpenPinch-compatible input/result formats.
- [ ] Use Nine-stream as final verification at the end of the migration, not as
      the routine baseline gate for every solver-moving PR.
- [ ] Treat `openhens_baseline_results/comparison.json` and
      `openhens_baseline_results/comparison.csv` as the cross-branch parity
      evidence for the expected values.

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
| `solved_esm_count` | `71` |
| `total_cases_attempted` | `1155` |
| `total_cases_solved` | `886` |
| `best_dTmin` | `18` |
| `best_min_dQ` | `1.7` |
| `best_stages` | `4` |
| `best_recovery_units` | `11` |
| `best_cu_units` | `3` |
| `best_hu_units` | `3` |

## Acceptance Tolerances By Metric

Use these tolerances for pass/fail decisions unless a task explicitly records a
reviewed scientific change. These are separate from solver convergence settings
and from tolerances used only to generate comparison artifacts.

- Solver convergence setting for the reference baseline run:
  `tolerance=1e-3`.
- Existing comparison artifact tolerance:
  `openhens_baseline_results/comparison.json` records `absolute=1e-6` and
  `relative=1e-9` for that artifact's cross-branch comparison.
- Migration pass/fail regression tolerance:
  `TAC_REL_TOL = 1e-4`, `TAC_ABS_TOL = 1.0`,
  `MAX_REGRESSION_REL_TOL = 1e-2`.

| Field family | Acceptance rule |
| --- | --- |
| Counts: solved ESM, attempted, solved, within-2/5/10 percent buckets, stage count, unit counts | Exact integer match. One-count allowance is permitted only for a documented threshold tie at the bucket boundary. |
| Grid selectors: `best_dTmin`, `best_min_dQ` / derivative threshold | Exact match to the expected value. |
| Best TAC / `best_solution` | Passes if both absolute delta <= `TAC_ABS_TOL` and relative delta <= `TAC_REL_TOL`. |
| Quartiles and run-summary TAC aggregates | Passes if relative delta <= `MAX_REGRESSION_REL_TOL`, unless the task declares exact artifact parity as its gate. |
| Utility loads, exchanger duties, exchanger areas, and heat balances | Must use the same numeric tolerance as best TAC unless a task defines a stricter per-field tolerance in the curated network snapshot. |
| Labels, source/sink stream identities, exchanger kind, stage, solver method, task id, run id, and verification-failure status | Exact match. |

## Baseline Provenance Requirements

Do not treat the current aggregate files as sufficient provenance for future
baseline regeneration. HENS-00 must add a checked-in manifest before solver
movement begins.

Required manifest fields:

- [ ] Source repository URL for each baseline run.
- [ ] Clean commit SHA for each run, or a stored dirty-worktree patch hash plus
      the patch artifact. Dirty baselines without a stored patch are invalid.
- [ ] Exact baseline-generation commands.
- [ ] Python version, dependency lock hash, and relevant package versions.
- [ ] Solver binary names and versions, including Couenne/IPOPT where used.
- [ ] Source CSV paths and SHA256 hashes or case-content hashes.
- [ ] Artifact-generation script path/version and artifact schema version.
- [ ] Generated artifact root and expected summary/result file paths.

## Adapter And Network Snapshot Requirements

Future tasks must not rely only on aggregate metrics where array adapters or
network extraction are touched.

- [ ] HENS-03 must create dedicated adapter snapshot artifacts, for example
      `adapter_snapshots/<case>/<dTmin>.json`, or document exact result JSON
      inputs and the extraction command that produces the snapshot.
- [ ] Adapter snapshots must include array shapes, axis maps, unit conventions,
      source stream identities, utility identities, and the covered `dTmin`
      values.
- [ ] The minimum routine adapter snapshot set is Four-stream at the selected
      baseline `dTmin=14`. Broader `dTmin` coverage is required when a task
      touches grid generation or pinch parity.
- [ ] HENS-08/HENS-11 must create curated network snapshots for the best ESM
      result, not an unspecified set of solved rows.
- [ ] Network snapshots must include source artifact path, task id, method,
      source/sink stream identities, exchanger kind, stage, duty, area, utility
      loads, units, and per-field numeric tolerances.
- [ ] If a task expands comparison beyond best ESM, it must name whether the
      comparison set is top `best_solutions_to_save` or all solved ESM rows.

## Case Acceptance Matrix

Use this matrix until HENS-11 expands the regression tier. HENS-00 records the
current source hashes in
`openhens_baseline_results/baseline_manifest.json`; if a source case changes,
the manifest, this table, and the expected metrics above must be updated in the
same reviewed change.

| Case | Tier / role | OpenHENS source CSV and SHA256 | Required migrated fixture | Routine solver gate | Final verification | Required grid / threshold coverage | Proof command or fixture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Four-stream-Yee-and-Grossmann-1990-1` | Tier 0 primary benchmark | `../OpenHENS/examples/cases/Four-stream-Yee-and-Grossmann-1990-1.csv`<br>`0c0a7f9592e7d8e27d35aef523530d70c11826c407dcbe4adba7fff86f4835f2` | `tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.json` plus reordered fixture `tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.reordered.json` | Yes: expected `best_solution=154853.8518602861`, `dTmin=14`, `min_dQ=0.5` | Yes | Full source `dTmin` grid `[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]` for pinch/adapter parity; source derivative threshold grid `[0.5, 0.9, 1.3, 1.7, 2.1, 2.4, 2.8, 3.2, 3.6, 4.0]`; selected routine adapter snapshot at `dTmin=14` and solver baseline at `min_dQ=0.5`. | `rtk uv run pytest -m solver` in OpenHENS for source baseline; migrated fast fixture/adapter tests must consume the named JSON fixtures and `openhens_baseline_results/adapter_snapshots/Four-stream-Yee-and-Grossmann-1990-1/dTmin-14.json`. |
| `Nine-stream-Linnhoff-and-Ahmad-1999-1` | Tier 0 final-verification benchmark | `../OpenHENS/examples/cases/Nine-stream-Linnhoff-and-Ahmad-1999-1.csv`<br>`fbf065509f355584623bfdfd5c38478410ddbab2a7c696e49bb3c15df3161b10` | `tests/fixtures/openhens/Nine-stream-Linnhoff-and-Ahmad-1999-1.json` plus reordered fixture `tests/fixtures/openhens/Nine-stream-Linnhoff-and-Ahmad-1999-1.reordered.json` | No routine gate; recreate fixture early but reserve solver runs for final verification or retirement gates. | Yes: expected `best_solution=2905807.275299348`, `dTmin=18`, `min_dQ=1.7` | Full source `dTmin` grid `[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]` when a task touches grid generation, pinch parity, or adapter semantics; final solver gate at `dTmin=18`, `min_dQ=1.7`. | `rtk uv run pytest -m solver` in OpenHENS for source baseline; final migrated solver gate must compare against `openhens_baseline_results/refactor/Nine-stream-Linnhoff-and-Ahmad-1999-1/summary.json` and `openhens_baseline_results/network_snapshots/Nine-stream-Linnhoff-and-Ahmad-1999-1/best-esm.json`. |
| Additional small cases | HENS-11 expanded regression tier | HENS-11 must name exact OpenHENS source paths and hashes before implementation. | HENS-11 must name exact fixture paths before implementation. | No until named. | Optional tier gate. | HENS-11 must name exact grids and thresholds. | HENS-11 must name exact commands and artifact paths. |
| Additional medium/large cases | HENS-11 expanded regression tier | HENS-11 must name exact OpenHENS source paths and hashes before implementation. | HENS-11 must name exact fixture paths before implementation. | No until named. | Optional tier gate. | HENS-11 must name exact grids and thresholds. | HENS-11 must name exact commands and artifact paths. |

## Adversarial Review Criteria

Adversarial review subagents must review each implementation against the plan,
the assigned task file, and the required workflow above. The review should
prioritize correctness, ownership boundaries, regression risk, and missing
tests. Style-only findings should not displace behavioral or architectural
findings.

Reviewers should treat any violation of the required workflow as a blocking
finding. A change is not acceptable if it works by adding an easier side path
around the OpenPinch-owned workflow.

Required review checks:

- [ ] Confirm the implementation follows the required workflow exactly:
      `PinchProblem` / `PinchWorkspace` -> `problem.design` -> synthesis service
      -> private adapter only where needed -> `TargetOutput.design` -> optional
      exports from `problem.results`.
- [ ] Confirm the synthesis service is internal and not root-exported or
      documented as a user-facing execution path.
- [ ] Confirm there is no public `OpenHENS`, `CaseStudy`, `SynthesisStudy`,
      `run_synthesis_workflow(...)`, raw-input runner, CSV runtime loader,
      HEN-specific workspace, import shim, field alias contract, or old keyword
      option shell.
- [ ] Confirm `PinchProblem` remains the only public owner of one problem's
      stream, utility, zone, targeting, and HEN design state.
- [ ] Confirm `PinchWorkspace` remains the only public owner of variants,
      multi-case execution, comparisons, and workflow dispatch.
- [ ] Confirm any `HeatExchangerNetworkSynthesis` runner is thin, bound to a
      live `PinchProblem`, and cannot own or reload streams, utilities,
      variants, cases, workspace state, or persistent options.
- [ ] Confirm persistent HEN controls are in `TargetInput.options` and
      `Configuration` / `CONFIG_FIELD_SPECS`, not in a HEN case/study/options
      object.
- [ ] Confirm option-like records such as design space, method sequence, solve
      setup, and outputs are not public configuration owners and are not
      accepted by public APIs.
- [ ] Confirm any external JSON alias is serialization-only, has a named
      non-OpenHENS consumer, and is not an OpenHENS compatibility alias.
- [ ] Confirm process streams and utilities are represented through
      `StreamSchema`, `UtilitySchema`, prepared `Stream`, `StreamCollection`,
      and `Zone` objects before any solver-array export.
- [ ] Confirm utility operating costs use `UtilitySchema.price`; capital-cost
      data uses existing or generalized OpenPinch costing configuration unless
      the task explicitly documented a reviewed gap.
- [ ] Confirm converted source examples are OpenPinch-compatible JSON fixtures
      and that runtime synthesis does not read OpenHENS CSVs.
- [ ] Confirm both Four-stream and Nine-stream examples are recreated in the
      expected migrated fixture/result formats when the task touches fixtures or
      benchmark setup.
- [ ] Confirm Four-stream is the routine benchmark gate for solver-moving work
      and uses the expected values in this README or
      `openhens_baseline_results/refactor/Four-stream-Yee-and-Grossmann-1990-1/summary.json`.
- [ ] Confirm Nine-stream is reserved for final verification unless the task is
      explicitly the final verification or retirement gate.
- [ ] Confirm any solver-baseline comparison uses the migration tolerances from
      this README and does not widen them without an explicit scientific review.
- [ ] Confirm baseline provenance includes clean commits or stored dirty patches,
      exact commands, dependency/solver versions, source case hashes, and
      artifact schema/version details before solver movement starts.
- [ ] Confirm adapter snapshots and curated best-ESM network snapshots exist
      before adapter or network-extraction behavior is marked complete.
- [ ] Confirm equation moves preserve solver defaults, objectives, variable
      names, tolerances, stage-reduction logic, topology evolution logic, and
      warm-start/parent-problem behavior unless the task explicitly approves a
      separate scientific change.
- [ ] Confirm PDM -> TDM -> ESM task fan-out semantics are preserved:
      failed PDM does not spawn TDM, failed TDM does not spawn ESM, task ids are
      deterministic, and topology restrictions are required.
- [ ] Confirm downstream TDM/ESM construction receives the solved parent/problem
      context required by source behavior and does not rebuild from cold
      defaults unless parity proves equivalence.
- [ ] Confirm solver arrays are private implementation details and do not become
      public API, serialized public result contracts, or user-facing access
      mechanisms.
- [ ] Confirm public network results use `HeatExchangerNetwork`,
      `HeatExchanger`, stable OpenPinch stream identities, and labelled access,
      not raw `Q_r`, `Q_h`, `Q_c`, `i`, `j`, `k` axis positions.
- [ ] Confirm recovery exchangers are represented as hot process stream -> cold
      process stream, hot utility exchangers as hot utility -> cold process
      stream, and cold utility exchangers as hot process stream -> cold utility.
- [ ] Confirm row-order invariance is tested where stream/utility ordering,
      axis maps, labelled access, or extracted network links are touched.
- [ ] Confirm `TargetOutput.design` is the canonical in-memory design result and
      that existing `TargetResults` are preserved or regenerated according to
      documented workflow semantics.
- [ ] Confirm optional JSON/CSV artifacts are generated only from
      `problem.results`, identify OpenPinch problem/workspace variant identity,
      and are not required as the terminal core workflow output.
- [ ] Confirm GEKKO, Pyomo, solver factories, solver binaries, plotting stacks,
      workbook libraries, and wake-management packages remain optional and lazy.
      `import OpenPinch` must stay lightweight.
- [ ] Confirm missing optional dependencies or solver binaries fail with
      actionable errors and do not silently skip required verification.
- [ ] Confirm fast tests cover schema validation, negative public API checks,
      root-primitive enforcement, fixture validation, adapter snapshots,
      labelled network access, result-cache behavior, workspace dispatch, and
      import smoke behavior as applicable to the task.
- [ ] Confirm solver tests are marked separately and that any skipped solver
      evidence records the exact command, environment, missing binary, and
      expected rerun path.
- [ ] Confirm helper replacements have focused parity tests before replacement
      and compare structural fields, utility loads, costs, areas, unit counts,
      and exports where affected, not only aggregate TAC.
- [ ] Confirm no broad refactor, compatibility bridge, visualization move,
      multi-utility expansion, distributed execution, new solver backend, or
      algorithm improvement is mixed into a behavior-preserving migration task.
- [ ] Confirm documentation and examples describe the OpenPinch-native workflow
      directly and do not imply import aliases, field aliases, command parity,
      or a wrapper package.
- [ ] Confirm public docs, examples, and code comments explain stable
      OpenPinch contracts, public/library semantics, non-obvious domain
      invariants, and migration constraints. Comments should clarify why a rule
      must hold or what contract future callers can rely on; they should not
      merely narrate incidental implementation mechanics, private array shapes,
      or temporary adapter details.
- [ ] Confirm task checkboxes and `Implementation Notes` are updated with
      reproducible evidence before the task is marked done.
- [ ] Confirm unrelated local files, generated caches, lockfile version bumps,
      `.DS_Store`, or unrelated benchmark artifacts are not included unless the
      task explicitly requires them.

Review severity guidance:

- [ ] Mark as blocking any public workflow bypass, public OpenHENS
      compatibility surface, runtime CSV synthesis path, alternate state owner,
      artifact-first result path, eager solver dependency import, missing
      baseline evidence for solver behavior, missing required snapshot artifact,
      unreviewed scientific change, or documentation/commentary that creates,
      endorses, or hides one of those public-contract violations.
- [ ] Mark as high severity any missing negative API test, missing
      row-order/identity test, missing result-cache test, missing fixture
      validation, missing optional-dependency guard, missing task fan-out
      coverage, or missing/misleading documentation and comments for stable
      public semantics, domain invariants, migration constraints, or reviewer
      obligations when the touched code requires them. Treat comments that
      substitute implementation noise for contract-level guidance as high
      severity when they leave future maintainers without the stable rule.
- [ ] Mark as medium severity localized documentation polish gaps that do not
      obscure public/library semantics, unclear evidence notes, weak error
      messages, or insufficient export snapshot coverage.

The adversarial review output should list findings first, ordered by severity
and grounded in file/line references. If there are no findings, the reviewer
must say that explicitly and identify any residual solver, dependency, or
benchmark risk that was not exercised.

## Plan Reference Map

Agents should use these original-plan sections for context:

- [Purpose](../../../OPENHENS_MIGRATION_PLAN.md#purpose)
- [Design Principles](../../../OPENHENS_MIGRATION_PLAN.md#design-principles)
- [Target Architecture](../../../OPENHENS_MIGRATION_PLAN.md#target-architecture)
- [OpenPinch Reuse Commitments](../../../OPENHENS_MIGRATION_PLAN.md#openpinch-reuse-commitments)
- [Root Primitive Mandate and Parallel Workflow Purge](../../../OPENHENS_MIGRATION_PLAN.md#root-primitive-mandate-and-parallel-workflow-purge)
- [TargetInput Boundary for HEN](../../../OPENHENS_MIGRATION_PLAN.md#targetinput-boundary-for-hen)
- [End-to-End Flow Comparison](../../../OPENHENS_MIGRATION_PLAN.md#end-to-end-flow-comparison)
- [Canonical Synthesis Problem Contract](../../../OPENHENS_MIGRATION_PLAN.md#canonical-synthesis-problem-contract)
- [Heat Exchanger Network Domain Model](../../../OPENHENS_MIGRATION_PLAN.md#heat-exchanger-network-domain-model)
- [Result Envelope Model](../../../OPENHENS_MIGRATION_PLAN.md#result-envelope-model)
- [Labelled HEN Data Access](../../../OPENHENS_MIGRATION_PLAN.md#labelled-hen-data-access)
- [OpenHENS Source Disposition](../../../OPENHENS_MIGRATION_PLAN.md#openhens-source-disposition)
- [Migration Phases](../../../OPENHENS_MIGRATION_PLAN.md#migration-phases)
- [Validation Strategy](../../../OPENHENS_MIGRATION_PLAN.md#validation-strategy)
- [Regression Tolerances](../../../OPENHENS_MIGRATION_PLAN.md#regression-tolerances)
- [Risks and Mitigations](../../../OPENHENS_MIGRATION_PLAN.md#risks-and-mitigations)
- [Non-Goals for the First Migration](../../../OPENHENS_MIGRATION_PLAN.md#non-goals-for-the-first-migration)
- [Recommended Review Slices](../../../OPENHENS_MIGRATION_PLAN.md#recommended-review-slices)

## Task Index

- [ ] [HENS-00 Baseline Freeze and Acceptance Matrix](00-baseline-freeze-and-acceptance-matrix.md)
- [ ] [HENS-01 Dependency and Runtime Viability Spike](01-dependency-and-runtime-viability-spike.md)
- [ ] [HENS-02 Synthesis Schemas and Network Domain](02-synthesis-schemas-and-network-domain.md)
- [ ] [HENS-03 JSON Fixture Conversion and Problem Adapter](03-json-fixture-conversion-and-problem-adapter.md)
- [ ] [HENS-04 Pinch Target Parity and Native Replacement Gate](04-pinch-target-parity-and-native-replacement-gate.md)
- [ ] [HENS-05 Design Workflow, Result Cache, and Fake Executor](05-design-workflow-result-cache-and-fake-executor.md)
- [ ] [HENS-06 Equation Kernel Base Move and Solver Boundary](06-equation-kernel-base-move-and-solver-boundary.md)
- [ ] [HENS-07 Stagewise and PDM Model Move](07-stagewise-and-pdm-model-move.md)
- [ ] [HENS-08 Stage Reduction and Topology Evolution Move](08-stage-reduction-and-topology-evolution-move.md)
- [ ] [HENS-09 Public Service and Documentation](09-public-service-and-documentation.md)
- [ ] [HENS-10 Duplicate Helper Replacement](10-duplicate-helper-replacement.md)
- [ ] [HENS-11 Regression Expansion and OpenHENS Retirement](11-regression-expansion-and-openhens-retirement.md)

## Required Evidence Format

Each task file includes an `Implementation Notes` section. Add short, dated
notes there while working:

```text
- 2026-06-16: `rtk uv run pytest tests/test_x.py -q` passed.
- 2026-06-16: Solver baseline blocked because Couenne is not installed; command
  and error recorded in PR description.
```

Evidence must be specific enough for a reviewer to reproduce the result or
understand the blocker.
