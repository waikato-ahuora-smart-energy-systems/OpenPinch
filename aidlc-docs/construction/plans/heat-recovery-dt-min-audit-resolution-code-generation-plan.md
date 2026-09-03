# Heat-Recovery DT_MIN Audit Resolution Code Generation Plan

This plan is the single source of truth for the one correction unit. Existing
public method names and signatures remain unchanged. The unit depends on the
current `Value`, zone, cascade, problem/workspace accessor, Pydantic, Pint,
pytest, and Hypothesis infrastructure and owns no database or deployment
artifact.

## Step 1: Red tests for strict scalar boundaries

- [x] Add selected-period and low-level contributor regressions proving that
  nested Booleans, numeric strings and bytes, sequences, arrays, arbitrary
  mappings, Boolean `dt_min` and `period_idx`, and fractional period indices are
  rejected while supported scalar and equivalent-unit forms remain accepted.
- [x] Run the focused tests and retain the expected Red failures.

## Step 2: Red tests for physical boundary accuracy and micro recovery

- [x] Pin analytical maximum, interior, and zero boundaries to
  `1e-6 delta_degC`; sample immediately around coincident shifted temperatures
  to require a monotone recovery predicate.
- [x] Require exact zero alone to produce `zero_recovery_boundary`; require
  positive tolerance-scale requests to return `solved` with achieved recovery
  no lower than requested.
- [x] Add post-verification failure tests for invalid bracket sides and retain
  the expected Red failures.

## Step 3: Red tests for local zone resolution and period mapping

- [x] Prove that a foreign same-address `Zone` selects local streams, a missing
  foreign address fails, and batch cases resolve the address independently.
- [x] Prove that exact canonical period mappings named `value` and `unit` take
  precedence over scalar mapping recognition while ordinary explicit-unit
  scalar broadcasting remains supported.
- [x] Run the focused application tests and retain the expected Red failures.

## Step 4: Red tests for strict result contracts

- [x] Add construction and JSON regressions for numeric type strictness,
  temperature-difference and heat-flow dimensionality, non-negativity,
  residual arithmetic, limit ordering, achieved-request feasibility, and
  status-specific relationships.
- [x] Preserve frozen, extra-forbid, finite, compatible-unit, and JSON
  round-trip behavior.
- [x] Run the contract tests and retain the expected Red failures.

## Step 5: Implement precision-preserving inverse cascade evaluation

- [x] Refactor `OpenPinch/analysis/targeting/cascade.py` behind private helpers
  so inverse evaluation can use exact finite temperature levels and strict
  interval overlap while all ordinary targeting defaults and public
  contributor signatures remain unchanged.
- [x] Update `OpenPinch/analysis/targeting/heat_recovery_dt_min.py` to use the
  precise path and strict numerical argument validation.
- [x] Make the new boundary and ordinary-cascade regression tests Green.

## Step 6: Implement exact-zero, micro-recovery, and bracket verification

- [x] Separate exact-zero branching from recovery tolerances.
- [x] Use an achieved-at-least-requested predicate for positive micro recovery,
  preserve ordinary-scale comparison tolerances, and explicitly verify final
  bracket-side predicates and width before returning.
- [x] Make the analytical, plateau, zero, micro-duty, clamping, and fail-closed
  solver tests Green.

## Step 7: Implement strict application normalization and local orchestration

- [x] Add a dedicated approved-shape normalizer in
  `OpenPinch/application/heat_recovery_dt_min.py` before shared unit coercion.
- [x] Resolve `Zone` objects by address against each current root and give exact
  canonical mapping keys precedence over scalar mapping detection.
- [x] Make problem, all-period, workspace, batch, unit, ordering, parallel, and
  non-mutation application tests Green.

## Step 8: Implement dimensional and relational result validation

- [x] Strengthen `OpenPinch/contracts/heat_recovery_dt_min.py` with strict
  numeric validation, field-specific compatible dimensions, non-negativity,
  canonical-unit relationship checks, and status invariants.
- [x] Keep service-generated output-unit overrides and JSON round trips valid.
- [x] Make all contract and service integration tests Green.

## Step 9: Expand properties and synchronize user-facing artifacts

- [x] Expand `tests/strategies/heat_recovery.py` with bounded multi-stream,
  segmented, inactive, threshold, no-overlap, micro-duty, period-ID, and unit
  cases while preserving useful shrinking and fixed seeds.
- [x] Extend property tests for monotonicity, analytical/forward oracles,
  maximality, feasibility, units, order, idempotence, non-mutation, JSON, local
  zones, and sequential/parallel equivalence.
- [x] Add every shrunk counterexample as an example regression.
- [x] Update RTD, release notes, generated notebook sources and packaged
  notebooks only where contracts or numerical outputs changed; validate
  inventories and drift.

## Step 10: Build, test, and summarize

- [x] Run focused analysis, contract, application, architecture, unit,
  notebook, documentation, and packaging tests.
- [x] Run Ruff lint and formatting checks, warning-strict Sphinx, generated
  notebook drift and execution, the complete configured pytest suite with the
  95 percent coverage gate, distribution builds, installed-wheel smoke, and
  patch hygiene.
- [x] Record exact results in the code-generation and build-and-test summaries,
  mark state and plan checkboxes in the same interaction, and present the
  generated correction for approval.

## PBT compliance

- PBT-01 is satisfied by the approved functional-design property catalog.
- PBT-02 through PBT-05 and PBT-07 through PBT-10 are explicit in Steps 4 and
  9 and remain blocking during generation and verification.
- PBT-06 is N/A because no persistent mutable state is introduced; generated
  state snapshots still prove non-mutation.
