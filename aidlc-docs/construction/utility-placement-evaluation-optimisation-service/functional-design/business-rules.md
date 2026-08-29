# Unit 2 Business Rules

## Rule Precedence

Unit 2 applies rules in this deterministic order:

1. reconcile and validate optimizer options;
2. validate resolved scope, selected periods, weights, target identity, units,
   ambient conditions, and turbine settings;
3. extract immutable period snapshots and feasibility bounds;
4. build the Unit 1 model;
5. verify/decode candidate coordinates;
6. replay targeting and verify coverage period by period;
7. calculate thermodynamic evidence;
8. calculate monetary/cogeneration evidence when requested;
9. aggregate, transform the backend scalar, and memoize evaluation;
10. run bounded optimization, filter feasible candidates, and assemble results.

Caller/configuration errors fail before the optimizer. Expected
coordinate-dependent failures become structured infeasibility. An operational
failure is raised only when no candidate can reasonably correct the condition.

## Options and Context Rules

### BR2-001: Optimizer options contract

The specialist options shall expose method, positive run count, non-negative
cluster tolerance, non-empty local method, and sorted immutable JSON-safe
backend overrides in addition to existing limits and seed.

### BR2-002: Default method and mapping

The default method is `cmaes`. Specialist options map to
the existing `OptimisationOptions`; no Unit 2-specific backend dispatch exists.
Callers may still select any supported method explicitly per placement call.

### BR2-003: Method-specific validation

Unknown methods, unknown override names, or invalid override values fail before
context targeting through typed placement validation translated from the
existing optimizer service.

### BR2-004: Bounded execution

Run, iteration, evaluation, and candidate limits are forwarded exactly. Unit 2
does not retry a completed or failed backend call.

### BR2-005: Scope resolution boundary

Unit 2 accepts only resolved `direct` or `total_site` scope. `auto` resolution
belongs to Unit 3 and is invalid at the numerical entry point.

### BR2-006: Period identity

Selected period IDs are non-empty, unique, and retained in canonical caller
order. Snapshot IDs and weights must match them one-for-one.

### BR2-007: Period weights

Weights are finite and non-negative with at least one positive weight. They are
not normalized. Zero-weight periods remain mandatory feasibility cases.

### BR2-008: Detached context

Context preparation may read only an isolated source. Stored context contains
immutable numbers and metadata, not `Zone`, `Stream`, `ProblemTable`, target,
DataFrame, callable, or optimizer objects.

### BR2-009: Snapshot completeness

Every period snapshot contains shifted and real interval temperatures,
process/load evidence, residual demands, approach limits, ambient temperature,
and enough source metadata to reconstruct candidate-local targeting inputs.

### BR2-010: Context units

Absolute temperatures, temperature differences, duties/work, entropy, prices,
and cost rates normalize to the Unit 1 `PlacementUnitSystem`. Currency
conversion remains excluded.

### BR2-011: Positive kelvin

Every ambient, process, and utility temperature used in entropy or turbine
work is finite and strictly above zero kelvin.

### BR2-012: Context repeatability

Building context twice from equivalent detached sources produces equivalent
snapshots and envelopes and does not alter either source.

## Candidate Replay and Coverage Rules

### BR2-013: Exact memo key

The memo key is the signed-zero-normalized exact finite coordinate tuple.
Tolerance quantization is prohibited because it can merge physically distinct
candidates.

### BR2-014: One physical replay per key

Within one execution process/session, an exact key is physically replayed at
most once. Memoized values are immutable and are not shared across service
calls or worker processes. Duplicate evaluation across isolated workers is
permitted; the parent performs canonical exact-coordinate deduplication and
re-evaluation before result assembly.

### BR2-015: Unit 1 verification first

Unit 1 vector verification and decoding precede target reconstruction. A
failed verification is ordinary candidate infeasibility.

### BR2-016: Shared placement

Every period uses the same decoded level identities, supply temperatures,
target temperatures, spans, and coordinate tuple.

### BR2-017: Fresh targeting state

Every period replay reconstructs fresh candidate-local problem tables and
utility streams. No duty, target, or turbine state survives into another
period or candidate.

### BR2-018: Existing allocation owner

Duty assignment delegates to the existing utility-targeting cascade. Unit 2
may adapt immutable inputs/outputs but shall not duplicate its assignment
algorithm.

### BR2-019: Level order

Hot and cold utilities enter targeting in the deterministic physical order
required by Unit 1. Targeting output is remapped to stable template keys rather
than positional guesses.

### BR2-020: Non-negative duty

An allocated level duty below negative coverage tolerance is infeasible. A
smaller negative numerical value normalizes to zero. Zero-duty levels remain in
period results.

### BR2-021: Side coverage

For each side and period, allocated duty minus residual requirement must have
absolute magnitude no greater than the named coverage tolerance.

### BR2-022: No clipping before validation

Coverage residuals are computed from raw finite allocated/required values.
Clipping or rescaling cannot turn an uncovered candidate into a feasible one.

### BR2-023: All-period feasibility

One failed side in one selected period makes the complete candidate
infeasible, including when that period's weight is zero.

### BR2-024: Early stop and diagnostics

Evaluation may stop after the first blocking period in canonical order. The
diagnostic must name the period, side, measured allocation, required duty, and
tolerance where applicable.

## Thermodynamic Rules

### BR2-025: Thermodynamic evidence always calculated

Every otherwise feasible period calculates entropy and exergy, including
monetary-mode periods.

### BR2-026: Balanced-composite basis

The real-temperature process hot/cold composites and allocated candidate
utility cascade use a common temperature basis. They form balanced composite
curves and are aligned on their union heat-load grid. Every entropy temperature
is converted to positive kelvin.

### BR2-027: Physical entropy generation

Thermodynamic placement cost is the physical entropy generated across matched
balanced-composite heat-load intervals. Whole-process entropy is retained only
as a reported decomposition term.

### BR2-028: Side separation

Hot-composite entropy change and cold-composite entropy change are calculated
with their physical signs and added.

### BR2-029: Sensible and isothermal entropy

Each sensible interval uses `CP * ln(T_out / T_in)` in kelvin. An isothermal
interval uses the signed limiting value `Q / T`.

### BR2-030: Deterministic integration

Interval entropy contributions are evaluated deterministically on the common
heat-load grid and yield `kW/K`.

### BR2-031: Union breakpoint grid

The balanced curves are aligned on the sorted union of their heat-load
breakpoints. Coordinates of zero-duty utilities cannot alter the objective.

### BR2-032: Canonical summation

Interval contributions are summed in temperature order using a stable finite
summation routine.

### BR2-033: Entropy balance

The reported process and utility terms sum to balanced-composite entropy
generation within named absolute/relative tolerances.

### BR2-034: Non-negative generation

Total entropy below negative numerical tolerance is infeasible. A value within
noise tolerance of zero normalizes to positive zero.

### BR2-035: Exergy destruction

Exergy destruction equals ambient absolute temperature times total entropy
generation and therefore is non-negative for a feasible period.

### BR2-036: Non-finite thermodynamics

Any non-finite branch or aggregate is never passed to the optimizer as a
feasible objective. Coordinate-dependent cases are infeasible; invariant
calculation defects raise a typed thermodynamic error.

### BR2-036A: Generated default utilities excluded

A positive allocation to a generated fallback named `HU` or `CU` receives a
bounded deterministic infeasible penalty and a `default_utility_forbidden`
diagnostic before objective ranking. Zero fallback duty adds no penalty.

## Monetary and Cogeneration Rules

### BR2-037: Monetary conditionality

Monetary evidence is calculated only for a monetary request. Thermodynamic
requests do not require prices and retain `None` for monetary breakdowns.

### BR2-038: Thermal purchase cost

Each level cost is its non-negative allocated duty in MW multiplied by its
non-negative price per MWh, yielding cost per hour. The period cost is the
canonical sum over hot and cold levels.

### BR2-039: Eligible turbine inputs

Only cogeneration-eligible hot levels with duty above tolerance enter the
turbine adapter. Cold, ineligible, and zero-duty levels are excluded.

### BR2-040: Turbine order and mode

Eligible levels are passed in descending temperature order to a fresh existing
multi-stage turbine using `above_pinch` mode.

### BR2-041: Turbine setting precedence

Explicit placement settings override existing problem power/turbine
configuration. Omitted settings inherit configuration. No independent hard-
coded turbine policy is introduced.

### BR2-042: No eligible work

When no eligible positive-duty level exists, cogenerated work and electricity
credit are exactly zero and the turbine is not called.

### BR2-043: Electricity credit

Electricity credit is cogenerated work in MW multiplied by the explicit
non-negative electricity price per MWh.

### BR2-044: Net monetary objective

Net monetary objective equals thermal purchase cost minus electricity credit.
Negative net cost is valid and must retain its sign.

### BR2-045: Recoverable turbine failure

A physically invalid candidate temperature/duty arrangement under valid
settings is ordinary candidate infeasibility with eligible level and period
context.

### BR2-046: Non-recoverable turbine failure

Missing/invalid settings, unsupported configuration, adapter contract
violation, or unexpected internal failure raises a typed run-level
cogeneration error and aborts the solve.

## Aggregation, Penalty, and Result Rules

### BR2-047: Raw weighted objective

Aggregate selected objective is `sum(weight * period_objective)` in canonical
period order. No average or weight normalization is permitted.

### BR2-048: Aggregate evidence

Thermodynamic aggregate values are always retained. Monetary aggregate values
are retained only for monetary requests. Aggregate decompositions equal their
explicit weighted period sums.

### BR2-049: Positive objective scale

The feasible scalar transform uses one deterministic finite positive scale
derived from context/request magnitudes. The scale is fixed before callback
execution.

### BR2-050: Feasible transform

For physical objective `C`, the backend value is
`0.5 + atan(C / scale) / pi`; it is finite, strictly increasing, and strictly
between zero and one.

### BR2-051: Infeasible penalty

For finite normalized violation `V >= 0`, the backend value is
`1 + V/(1+V)`. Missing magnitude uses a deterministic class-specific unit
violation. Every infeasible value is at least one and below two.

### BR2-052: Final feasibility filter

Backend scalar order never establishes public success. Final ranking discards
every evaluation not proven feasible.

### BR2-053: Candidate source union

Final consideration includes the exact-coordinate union of backend candidates
and deterministic Unit 1 initial points. Existing backend clustering is
accepted; Unit 2 performs no tolerance quantization. The parent process fully
re-evaluates each retained coordinate and does not trust worker-local mutable
memo state as public evidence.

### BR2-054: Candidate order and limit

Feasible candidates order by physical aggregate objective then exact
coordinates. `candidate_limit` includes the best; alternatives contain at most
`candidate_limit - 1` entries.

### BR2-055: Exhaustion

No feasible retained candidate raises a typed no-feasible-placement error with
scope, objective, counts, periods, method, seed, limits, and bounded diagnostic
summary. A least-infeasible result is prohibited.

### BR2-056: Detached result

The result contains only Unit 1 frozen public values. It retains no snapshot,
memo, problem table, stream, target, turbine, callable, or optimizer result.

### BR2-057: Termination translation

Method, seed, status, counts, and configured limits are copied into
`PlacementTermination`. Unknown backend-private fields are omitted.

### BR2-058: Bounded diagnostics

The service may summarize rejected failure classes and the most informative
infeasible evaluation, but it does not expose a full unbounded callback trace.

## Scenario Matrix

| Scenario | Required outcome | Rules |
|---|---|---|
| Valid two-isothermal direct case | Full coverage, finite entropy, detached feasible result | BR2-005 through BR2-036, BR2-052 through BR2-057 |
| Total Site period snapshot | Existing site-profile targeting reused without child mutation | BR2-005 through BR2-018 |
| Zero-duty level | Identity retained; both side balances close | BR2-019 through BR2-023 |
| One failed zero-weight period | Complete candidate infeasible | BR2-007, BR2-023, BR2-047 |
| Coincident and widened profiles | Zero identity, closer-first ranking, and finite exergy | BR2-026 through BR2-035 |
| Monetary case without eligible levels | Purchase cost, zero work/credit, valid net cost | BR2-037 through BR2-044 |
| Eligible extraction levels | Only eligible ordered hot duties enter fresh turbine | BR2-039 through BR2-041 |
| Candidate-specific turbine incompatibility | Candidate infeasible; search continues | BR2-045 |
| Invalid turbine configuration | Typed run-level failure | BR2-046 |
| Negative net monetary objective | Remains feasible and ranks by physical value | BR2-044, BR2-050 through BR2-054 |
| Repeated exact callback | One physical replay and equal evaluation | BR2-013, BR2-014 |
| Backend returns only penalties | Typed exhaustion, never least-infeasible success | BR2-052, BR2-055 |
| Fixed-seed tiny bounded case | Best agrees with structured grid within declared tolerance | BR2-001 through BR2-004, BR2-053, BR2-054 |

## Requirement and Story Traceability

| Requirement/story | Owning rules |
|---|---|
| FR-005 bounds consumption | BR2-009 through BR2-012 |
| FR-007 direct/Total Site | BR2-005, BR2-008, BR2-009, BR2-018 |
| FR-008 and UPO-03/UPO-06 coverage/multiperiod | BR2-006, BR2-007, BR2-016 through BR2-024, BR2-047 |
| FR-009 and UPO-04 thermodynamics | BR2-025 through BR2-036A |
| FR-010 and UPO-05 monetary/cogeneration | BR2-037 through BR2-046 |
| FR-011 and UPO-11 deterministic optimization | BR2-001 through BR2-004, BR2-049 through BR2-055 |
| FR-012 and UPO-08 failures | BR2-023, BR2-024, BR2-036, BR2-045, BR2-046, BR2-051, BR2-055 |
| FR-014 and UPO-07 result content | BR2-048, BR2-052 through BR2-058 |
| UPO-02 numerical workflow | BR2-005 through BR2-057 |
| UPO-12 TDD/PBT | Scenario Matrix and companion PBT-01 property table |
| NFR-001 numerical correctness | BR2-010, BR2-011, BR2-020 through BR2-051 |
| NFR-002 reproducibility | BR2-006, BR2-013, BR2-032, BR2-047, BR2-053, BR2-054 |
| NFR-003 bounded execution | BR2-004, BR2-054, BR2-058 |
| NFR-005 maintainability | BR2-008, BR2-018, BR2-040, BR2-056 |
| NFR-006 observability | BR2-024, BR2-036, BR2-045, BR2-046, BR2-055, BR2-058 |

The companion business-logic property table supplies at least one identified
PBT category for every algorithmic component. Example tests remain mandatory
for all rows in the Scenario Matrix.
