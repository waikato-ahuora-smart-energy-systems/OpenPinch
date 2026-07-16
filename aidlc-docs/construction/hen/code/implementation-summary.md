# Heat Exchanger Network Implementation Summary

## Outcome

HEN synthesis retains parent streams and parent match binaries while evaluating segmented sensible-duty profiles through ordered cumulative heat coordinates. Extracted networks contain one exchanger per parent match and expose local, duty-aligned segment area contributions without adding topology nodes.

## Delivered

- Per-period parent-to-segment solver tensors with stable identities, masks, temperatures, cumulative duties, CP values, and HTCs.
- Parent cumulative heat-coordinate equations and piecewise `T(Q)` mappings for stage boundaries and non-isothermal branch outlets.
- SOS2-style interval disjunctions for integer-capable paths and warm-started active-interval iteration with explicit failure guidance for IPOPT paths.
- Pinch-side profile clipping that splits the affected numerical segment while preserving parent identity.
- Ordered countercurrent area slicing at every hot and cold segment boundary, including utility exchangers and maximum period-total design area.
- Deliberate two-level area strategy: the topology-search objective retains the
  smooth Chen LMTD surrogate, while solved networks are recalculated from exact
  duty-aligned segment slices.
- Exact segment-summed post-solve capital cost, TAC, TDM derivative, EVM ranking, extraction, and verification data.
- Internal frozen `HeatExchangerAreaSlice` records nested under parent
  exchangers, with parent-level period duty/area totals and authoritative
  maximum period-total design area.
- Stable serialized `segment_area_contributions` payloads without adding a
  slice type to the package-root API, plus segment-profile cache version
  invalidation.
- Parent-only diagram and controllability compatibility through unchanged parent exchanger topology.

## Verification

- Full non-solver suite: 1,941 passed, 1 skipped, 6 synthesis tests deselected.
- Solver-marked suite: 5 passed, covering segmented PDM and TDM, isothermal and non-isothermal formulations, total-cost handling, recovery and utility area contributions.
- Focused hand-calculated and property tests validate slice ordering, local LMTDs, duty conservation, area sums, and multiperiod maximum period-total area.
- Ruff checks passed across `OpenPinch` and `tests`; all changed Python files passed formatting checks.

## Maintainability Follow-up

The internal `HeatExchangerAreaSlice` value model and pure period aggregation/design-area calculations now live in the private `_heat_exchanger_area.py` helper. `HeatExchanger` retains its existing field, validation, property, direct-module import, and serialized payload contracts through thin delegation.

## Segmented PDM dTmin Propagation Correction

PDM copied-zone preparation now applies `max(prepared dt_cont, dTmin / 2)` to
every explicit child segment before direct-integration targeting. Flat streams
retain the existing parent assignment. Segment updates use the transactional
parent API, so each child's per-period values are preserved above the minimum,
the parent aggregate is re-derived, numeric-view revisions are invalidated, and
the source problem remains unchanged because PDM operates on its copied zone.

Verification for this correction:

- 78 focused PDM and segmented-stream tests passed with Hypothesis seed
  `20260716`.
- 79 direct-targeting, problem-table, and segment-domain tests passed.
- 385 broader non-solver HEN tests passed; one solver-marked test was deselected.
- The complete CI-selected non-solver suite passed: 1,955 tests with four solver
  tests deselected.
- Total line coverage remained 99%, above the 95% project gate; the modified PDM
  decomposition module reached 100%.
- Ruff formatting, Ruff lint, and `git diff --check` passed.

## Segmented Parent dt_cont Transaction

Assigning `dt_cont` to a segmented parent now applies the full value to every
ordered child through detached candidates and commits the profile only after all
candidate mutations and complete-profile validation succeed. Indexed
multiperiod updates use the same transaction and apply the selected-period value
to every child while preserving values in other periods. Parent aggregate and
derived state are then rebuilt through the established `replace_segments` path,
which also updates ownership, revisions, and segment-aware cache signatures.

Flat streams retain their previous scalar and indexed mutation behavior, and
explicit segment-level overrides continue to use the existing single-segment
transaction.

Verification for this correction:

- Five focused scalar, multiperiod, rollback, flat-stream, and Hypothesis
  contract tests passed with seed `20260715`.
- 84 stream, segmented-stream, and collection tests passed.
- 156 direct integration, indirect integration, problem-table, HPR, and
  segmented-PDM regression tests passed.
- The complete CI-selected non-solver suite passed effectively: 1,958 tests
  passed in the restricted run and the two environment-dependent Chrome/Sphinx
  checks passed when rerun with their required permissions, for all 1,960
  selected tests passing and four solver tests deselected.
- Total line coverage is 99%, above the 95% repository gate.
- Ruff formatting, Ruff lint, and `git diff --check` passed.

## Pre-Release Period-Native PDM and Utility Constraints

PDM decomposition now carries an ordered target and clipped stream state for
every operating period. Each period uses its own pinch and utility targets,
while process topology is shared from the union of period-active streams.
Above- and below-pinch solutions preserve every period's duties, boundary
temperatures, approach variables, split fractions, and explicit
non-isothermal branch outlet temperatures during amalgamation.

Non-isothermal warm starts now normalize hot fractions over cold matches and
cold fractions over hot matches. GEKKO warm-start assignment also correctly
handles variables with only one finite bound. Segmented utilities expose
period-indexed `dt_cont` tensors and use the first segment contribution at the
inlet and the traversed contribution at the solved match outlet, selecting the
larger adjacent value at an exact segment boundary. Flat utilities retain the
scalar constraint path.

Verification for this unit:

- 443 broad non-solver HEN tests passed with four solver cases deselected.
- The complete non-solver repository suite passed: 1,996 tests with four
  solver-marked cases deselected.
- The solver-marked suite passed three tests with one intentional skip.
- The seven-fixture tier 0/1 canonical matrix produced thirteen accepted
  networks and the established nine-stream tier 1 bounded timeout. A discovered
  zero-stage utility-only decomposition regression was corrected and both Spray
  Dryer tiers then passed.
- Repository Ruff lint, changed-file formatting, warning-free Sphinx,
  wheel/sdist packaging, and `git diff --check` passed.

## Extension Compliance

- Security Baseline: disabled; not enforced.
- Resiliency Baseline: disabled; not enforced.
- Property-Based Testing (Partial): compliant through domain-specific generators, bounded examples, shrinking, deterministic CI seeds, serialization and continuity invariants, target parity, and area-slice invariants.

## Pre-Release Period-Native HEN Results

Heat exchanger operating results now use non-empty, ordered
`HeatExchangerPeriodState` records. Each state carries period identity, duty,
activity, approach temperatures, split fractions, and source/sink inlet and
outlet temperatures. `HeatExchanger` retains only shared topology, area, and
capital design data. Multiperiod network totals, labels, diagrams, exports, and
controllability queries require an explicit period; omission remains valid only
for a single-period result.

Recovery and utility extraction now traverses every solver period, retains
matches that are active only after the first period, and prefers explicit
non-isothermal branch temperatures. Shared exchanger topology is derived from
any-period activity. Verification applies period-specific heat-capacity data
and split fractions when checking explicit branch temperatures. Small negative
solver duties within the established numerical tolerance are normalized to
zero, while materially negative duties remain invalid.

All internal consumers, serialized schemas, diagram rendering, exports,
ranking, controllability analysis, the packaged HEN notebook, tests, and public
documentation use the period-native result contract. No scalar operating-data
aliases or period-zero compatibility fields were retained.

Verification for this unit:

- Focused result, extraction, diagram, export, controllability, and area tests
  passed: 307 tests with nine intentional deselections, plus 52 reporting tests
  and 25 focused controllability/thermal tests.
- The complete non-solver repository suite passed: 1,999 tests with four
  solver-marked cases deselected.
- The solver-marked suite passed three tests with one intentional skip, and the
  canonical Four-stream live-solver baseline passed.
- The canonical tier 0/1 benchmark produced twelve accepted networks. The
  Nine-stream and Six-stream Yee tier 1 cases reached their bounded solve
  timeouts; neither solver invocation returned a result for extraction. The
  complete solver-marked correctness matrix remained green.
- Repository Ruff lint, changed-file formatting, notebook JSON validation,
  warning-free Sphinx, wheel/sdist packaging, and `git diff --check` passed.
