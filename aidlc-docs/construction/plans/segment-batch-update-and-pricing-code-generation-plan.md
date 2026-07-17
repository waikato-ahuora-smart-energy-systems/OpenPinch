# Segment Batch Update and Pricing Code Generation Plan

## Unit Context

- **Unit**: Segmented stream mutation, utility preparation, and HEN costing
- **Change type**: Brownfield correctness and public-contract refinement
- **Dependencies**: Ordered `StreamSegment` ownership, atomic profile replacement,
  expanded numeric views, segmented HEN thermal profiles, and current single hot
  and cold utility selection
- **Public contracts**: `Stream.update_segments` provides sparse atomic mutation;
  segment prices may differ; segmented parent utility price is duty-weighted;
  parent price assignment remains an explicit broadcast
- **Compatibility**: Flat streams and utilities retain their current behavior;
  temperature-heat profiles retain one parent/default price; multiple-utility HEN
  selection remains deferred
- **Story traceability**: N/A - internal domain and solver correctness refinement

This document is the single source of truth for this approved Code Generation
continuation.

## Execution Steps

### Step 1 - Establish atomic batch mutation

- [x] Add `Stream.update_segments(updates)` with sparse index-to-change mappings.
- [x] Extract one clone-mutate-validate-commit primitive shared by batch, single,
  and uniform segment updates.
- [x] Validate all indexes and attributes before commit; preserve order and make
  empty mappings a no-op.
- [x] Refactor PDM to apply distinct segment `dt_cont` values in one transaction.
- [x] Add rollback, multiperiod, cache, revision, order, and compatibility tests.

### Step 2 - Implement segment-specific pricing

- [x] Change omitted domain price handling to preserve explicit child prices while
  retaining the flat zero default.
- [x] Derive segmented parent price by duty-weighted period and parent cost as the
  exact sum of child costs.
- [x] Preserve explicit parent price assignment as a broadcast and local child
  price mutation as an independent update.
- [x] Cover replacement, copying, serialization, period-indexed mutation, and
  price/cost conservation.

### Step 3 - Prepare segmented utility inputs

- [x] Extend structured utility schemas with mutually exclusive explicit segments
  and temperature-heat profiles.
- [x] Resolve explicit segment price before parent price before the existing
  default, without adding interval-specific prices to profile inputs.
- [x] Build one ordered segmented utility parent while preserving existing flat
  input and utility-selection behavior.
- [x] Add nested validation, precedence, default, and compatibility tests.

### Step 4 - Implement exact segmented-utility HEN costing

- [x] Prepare stable utility segment price, duty, cumulative-duty, cumulative-cost,
  and identity arrays for each period.
- [x] Extend the shared piecewise profile with exact cumulative `cost_at_heat(Q)`.
- [x] Replace average-price objective, extraction, verification, and ranking terms
  with exact traversed-segment costs for the selected hot and cold utilities.
- [x] Retain flat utilities as one virtual segment and preserve existing solver
  behavior outside segmented utility pricing.
- [x] Add hand-calculated partial, boundary-crossing, full-profile, hot/cold, and
  multiperiod tests across supported solver paths.

### Step 5 - Documentation and quality gates

- [x] Update API and HEN documentation for batch mutation, derived parent price,
  broadcast assignment, child prices, and segmented utility restrictions.
- [x] Add Partial-enforcement Hypothesis invariants for atomic rollback and
  price/cost conservation using domain-specific generators and reproducible seeds.
- [x] Run focused tests, full non-solver tests, solver-marked tests, Ruff,
  documentation, packaging, coverage, and patch-hygiene checks.
- [x] Record completion evidence in the code summary, build/test artifacts,
  workflow state, and audit log.

## Extension Compliance

- **Security Baseline**: Disabled; not enforced.
- **Resiliency Baseline**: Disabled; not enforced.
- **Property-Based Testing (Partial)**: Atomicity and cost-conservation invariants
  use reusable domain generators, standard shrinking, Hypothesis, and the existing
  reproducible CI seed policy.

## Content Validation

- Markdown headings, lists, inline code, paths, and checkboxes were checked for
  CommonMark-compatible syntax.
- No Mermaid, ASCII diagram, JSON, YAML, or other structured block is present.
