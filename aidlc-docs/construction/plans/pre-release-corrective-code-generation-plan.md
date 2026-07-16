# Pre-Release Corrective Code Generation Plan

## Unit Context

- **Change type**: Cross-cutting pre-release correctness correction
- **Baseline**: Local revision `756c1128`
- **Delivery**: Four dependency-ordered stacked pull requests
- **Compatibility policy**: Clean breaking contracts; no legacy aliases,
  migrations, deprecation paths, or period-zero compatibility shims
- **Units**: Domain and Input; HEN PDM and constraints; HEN results; targeting
  summaries and shared HPR economics
- **Story traceability**: N/A - this work closes fifteen validated code-review
  findings against the approved segmented-stream and multiperiod requirements
- **Extensions**: Security and Resiliency disabled; Property-Based Testing Partial
  applies to transaction, serialization, and period-weight invariants

This document is the single source of truth for the approved corrective Code
Generation work. Each checkbox must be updated in the same interaction in which
its work completes.

## Step 1 - PR 1: Domain and Input Correctness

- [x] Add failing regression tests for immutable stream-owned values, segmented
  default-utility extrema, partial period weights, strict canonical inputs,
  workspace schema v2, and validation/preparation parity.
- [x] Make every `Value` stored by `Stream` and `StreamSegment` read-only while
  keeping standalone values mutable; update internal indexed mutations to use
  mutable candidates and explicit stream transactions.
- [x] Calculate default utility temperatures from all ordered child-segment
  shifted extrema in every period.
- [x] Add one canonical period-weight resolver and route zone, utility, HEN, and
  summary consumers through it.
- [x] Enforce canonical-only input fields with forbidden extras; change workspace
  bundles and repository fixtures to strict schema version 2 using `case_input`.
- [x] Share segmented stream/utility semantic checks between validation reporting
  and preparation, including parent aggregate assertions.
- [x] Update domain/input documentation and run focused, full non-solver, Ruff,
  documentation, packaging, and patch-hygiene checks.
- [x] Record PR 1 implementation summary and Build and Test evidence.

## Step 2 - PR 2: Period-Native PDM and Utility Constraints

- [x] Add failing two-period PDM, amalgamation, warm-start, and segmented-utility
  `dt_cont` regression tests.
- [x] Replace singular PDM target and clipped-temperature fields with ordered
  period-native contracts carrying period identities.
- [x] Feed period targets and pinches into PDM preprocessing, side activation,
  stage topology, and utility constraints.
- [x] Amalgamate all recovery, utility, temperature, split, and non-isothermal
  variables for every period and derive shared topology from any-period activity.
- [x] Correct single- and multiperiod warm-start loop nesting and X/Y duty-share
  denominators.
- [x] Add utility-segment `dt_cont` tensors and exact active-segment approach
  mappings, using the larger contribution at an exact segment boundary.
- [x] Run focused, solver-marked, canonical HEN tier 0/1, Ruff, documentation,
  packaging, and patch-hygiene checks; record PR 2 evidence.

## Step 3 - PR 3: Period-Native HEN Results

- [x] Add failing later-period-only match, explicit non-isothermal temperature,
  period-state serialization, and ambiguous-access regression tests.
- [x] Introduce `HeatExchangerPeriodState` and replace exchanger operational
  scalar fields with non-empty ordered period states.
- [x] Require explicit period identity for multiperiod network queries, diagrams,
  exports, and controllability; allow omission only for single-period networks.
- [x] Extract all period recovery and utility states, retain later-only matches,
  and prefer explicit branch outlet temperatures over CP arithmetic.
- [x] Migrate internal consumers, public schemas, exports, diagrams, notebooks,
  examples, fixtures, and documentation to the period-native contract.
- [x] Run focused, solver-marked, canonical HEN tier 0/1, full, Ruff,
  documentation, packaging, and patch-hygiene checks; record PR 3 evidence.

## Step 4 - PR 4: Summary Isolation and HPR Economics

- [x] Add failing success/error state-isolation and shared-HPR aggregation tests.
- [x] Replay each summary period against a fresh baseline-zone copy and restore
  the original zone identity, result cache, and recorded target specification.
- [x] Score shared simulated-HPR candidates with weighted operating cost and
  penalty plus maximum annualized capital cost.
- [x] Aggregate public HPR outputs using weighted operating fields, maximum
  capital fields, and recomputed total annualized cost; retain existing policy
  for non-HPR fields.
- [x] Update targeting/HPR documentation and run focused, full, Ruff,
  documentation, packaging, solver, and patch-hygiene checks.
- [x] Record PR 4 implementation summary and Build and Test evidence.

## Step 5 - Final Review and Handoff

- [x] Re-run all fifteen review reproducers and confirm every finding is closed.
- [x] Run the complete supported test, coverage, solver, documentation, notebook,
  packaging, Ruff, and patch-integrity matrix.
- [x] Confirm no duplicate application files or unintended worktree changes.
- [x] Update AI-DLC state, audit, release notes, and final Build and Test summary.
- [ ] Publish the four stacked branches and create their pull requests only after
  each branch is independently green.

Publication is pending because the required GitHub CLI is not installed and the
restricted environment cannot resolve GitHub to identify the remote branch that
contains the four approved baseline commits. Local `develop` is four commits
ahead of `origin/develop`; it must not be pushed or used as an implicit PR base
without confirming the intended remote baseline.

## Approved Decisions

- Breaking pre-release API and persistence changes are intentional.
- Stream-owned values are immutable; explicit domain mutation APIs are required.
- PDM and HEN operational results are period-native without singular aliases.
- Workspace schema v1 and retired input aliases are rejected, not migrated.
- The four local pending commits remain the implementation baseline.

## Content Validation

This Markdown contains no Mermaid, ASCII diagram, JSON, or YAML block. Headings,
lists, paths, inline code, and checkboxes were checked for CommonMark-compatible
syntax before file creation.
