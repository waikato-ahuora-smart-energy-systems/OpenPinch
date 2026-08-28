# Unit 2 Domain Entities

## Modeling Conventions

- Persistent public results reuse the frozen Unit 1 Pydantic contracts.
- Unit 2 internal values are frozen dataclasses or frozen specialist models
  containing tuples, enums, strings, booleans, and finite canonical numbers.
- NumPy arrays may exist only inside short-lived adapters to existing numerical
  owners; they are copied on entry and do not cross a stored context/result
  boundary.
- `Zone`, `Stream`, `ProblemTable`, target, turbine, backend, exception, and
  callable objects are never fields of a stored context or public result.
- `None` means not applicable or unavailable, never physical/economic zero.

## Unit 1 Contract Amendment

### Extended `UtilityPlacementOptions`

| Field | Type | Rule |
|---|---|---|
| existing `candidate_limit` | positive integer | Counts best plus alternatives |
| existing `iteration_limit` | positive integer | Maps to optimizer `maxiter` |
| existing `evaluation_limit` | positive integer | Maps to optimizer `maxfun` |
| existing `seed` | integer | Forwarded unchanged |
| `method` | stable existing method value | Default `dual_annealing` |
| `run_count` | positive integer | Maps to `n_runs` |
| `cluster_tolerance` | finite non-negative float | Maps to `cluster_tol` |
| `local_method` | non-empty string | Maps to `local_method` |
| `backend_options` | sorted tuple of name/value pairs | JSON-safe; method-specific validation in Unit 2 |

This is a test-first correction of the already-approved Unit 1 design. Existing
constructor calls and serialized payloads remain valid through defaults.

## Context Entities

### `ResolvedPlacementSource`

Ephemeral builder input supplied by Unit 3 later in the workflow. It carries an
isolated source zone, resolved direct/Total Site scope, base-target identity,
selected periods/weights, and explicit turbine overrides. It is not stored in a
placement context or returned result.

### `ProcessEntropySlice`

| Field | Type | Rule |
|---|---|---|
| `interval_index` | non-negative integer | Stable position in real-temperature table |
| `side` | hot or cold utility side | Identifies heated or cooled process composite |
| `temperature_in_kelvin` | float | Finite and positive |
| `temperature_out_kelvin` | float | Finite and positive |
| `available_duty` | float | Finite and non-negative kW |
| `heat_capacity_flow` | float or `None` | Finite signed kW/K when directly available |

Purpose: immutable real-temperature process evidence retained for diagnostics
and compatibility. Placement cost uses the aligned residual process and utility
profiles.

### `PlacementTargetSnapshot`

| Field | Type | Rule |
|---|---|---|
| `shifted_temperatures` | tuple of floats | Canonical descending target-grid temperatures |
| `real_temperatures` | tuple of floats | Aligned unshifted temperatures |
| `hot_load_profile` | tuple of floats | Existing cascade hot-side profile |
| `cold_load_profile` | tuple of floats | Existing cascade cold-side profile |
| `pinch_indices` | pair of integers | Valid indices into profile arrays |
| `entropy_slices` | tuple of `ProcessEntropySlice` | Complete real-temperature interval evidence |
| `reconstruction_metadata` | immutable serializable tuples | Only fields required by the existing adapter |

Purpose: immutable numerical snapshot capable of reconstructing fresh local
targeting inputs without retaining a mutable `ProblemTable`.

### `PlacementPeriodContext`

| Field | Type | Rule |
|---|---|---|
| `period_id` | string | Non-empty, ordered unique identity |
| `weight` | float | Finite and non-negative |
| `target_snapshot` | `PlacementTargetSnapshot` | Detached complete snapshot |
| `residual_hot_duty` | float | Finite non-negative kW |
| `residual_cold_duty` | float | Finite non-negative kW |
| `ambient_temperature_kelvin` | float | Finite and positive |
| `coordinate_bounds` | physical bound tuple | Complete Unit 1 blueprint keys |

### `TurbineSettings`

Frozen resolved non-placement settings required by the existing multi-stage
turbine: mode, inlet temperature and pressure, model, minimum/loaded/mechanical
efficiencies, flash behavior, and provenance indicating explicit override or
configuration fallback. For this feature, mode is `above_pinch`.

### `UtilityPlacementContext`

| Field | Type | Rule |
|---|---|---|
| `scope` | resolved direct or Total Site enum | Never `auto` |
| `base_target_id` | string | Stable detached target identity |
| `periods` | tuple of `PlacementPeriodContext` | Non-empty in canonical order |
| `turbine_settings` | `TurbineSettings` | Valid before monetary solve |
| `units` | `PlacementUnitSystem` | Agrees with request/envelope |
| `objective_scale` | positive float | Fixed before callback execution |

It derives the `PlacementFeasibilityEnvelope` consumed by Unit 1 but does not
contain the effective Unit 1 model itself.

## Allocation Entities

### `UtilityAllocationSlice`

| Field | Type | Rule |
|---|---|---|
| `template_key` | Unit 1 `TemplateKey` | Stable level identity |
| `interval_index` | integer | References one real-temperature interval |
| `duty` | float | Finite non-negative kW |
| `process_temperature_in_kelvin` | float | Finite positive |
| `process_temperature_out_kelvin` | float | Finite positive |
| `process_heat_capacity_flow` | float or `None` | Used when existing table provides it |

Purpose: captures the existing cascade's actual assignment of one level to one
process interval as diagnostic evidence. Coverage is owned by residual-duty
allocation and thermodynamic placement by balanced-composite entropy generation.

### `LevelAllocation`

Contains one decoded level, its finite non-negative total duty, and ordered
`UtilityAllocationSlice` values whose duties sum to that total within coverage
tolerance.

### `PeriodAllocation`

| Field | Type | Rule |
|---|---|---|
| `period_id` | string | Matches context |
| `hot_levels` | tuple of `LevelAllocation` | Complete hot inventory in declared order |
| `cold_levels` | tuple of `LevelAllocation` | Complete cold inventory in declared order |
| `allocated_hot_duty` | float | Sum of hot level duties |
| `allocated_cold_duty` | float | Sum of cold level duties |
| `hot_coverage_residual` | float | Allocated minus required |
| `cold_coverage_residual` | float | Allocated minus required |
| `diagnostics` | diagnostic tuple | Empty exactly when allocation is feasible |

## Objective Entities

### `BranchEntropyContribution`

Identifies period, utility level, utility/process branch, interval, duty,
absolute inlet/outlet temperatures, calculation mode (`sensible_log1p` or
`isothermal_limit`), and finite signed entropy. It is internal evidence and is
summarized into the public breakdown rather than serialized in full.

### `PeriodThermodynamicEvaluation`

Contains ordered utility and process branch contributions plus their stable
sums, ambient temperature, total entropy generation, and exergy destruction.
It converts directly to Unit 1 `ThermodynamicCostBreakdown`.

### `CogenerationInput`

Contains descending eligible level keys, extraction temperatures, positive
duties, and resolved `TurbineSettings`. It owns copied tuples; the adapter
creates NumPy arrays and a fresh turbine only during the call.

### `CogenerationEvaluation`

Contains total finite non-negative work, eligible input keys, and a bounded
serializable diagnostic summary. Backend stage objects/details are discarded.

### `PeriodMonetaryEvaluation`

Contains thermal purchase cost, `CogenerationEvaluation`, electricity price,
electricity credit, and net monetary objective. It converts directly to Unit 1
`MonetaryCostBreakdown`.

## Candidate and Optimization Entities

### `CandidateEvaluation`

| Field | Type | Rule |
|---|---|---|
| `coordinates` | exact normalized float tuple | Memo identity |
| `decoded` | `DecodedPlacement` or `None` | Present after successful decode |
| `feasible` | boolean | True only with every period complete |
| `period_results` | immutable period result tuple | Canonical completed periods |
| `aggregate_objective` | float or `None` | Physical selected objective only when feasible |
| `thermodynamic_total` | float or `None` | Present for feasible candidates |
| `monetary_total` | float or `None` | Present only for feasible monetary candidates |
| `backend_scalar` | finite float | Feasible transform or infeasible penalty |
| `violation_magnitude` | finite non-negative float | Zero for feasible evaluations |
| `diagnostics` | ordered diagnostic tuple | Empty for feasible evaluation |

### `EvaluationMemo`

A per-service-call mutable mapping from exact coordinate tuple to immutable
compact callback records. Each record retains the finite backend scalar,
feasibility, physical aggregate when available, normalized violation, and one
bounded diagnostic reference rather than complete per-level/period output. The
memo is encapsulated in one execution process/session, has no public
serialization, is bounded by that worker's evaluation limit, and is discarded
after result assembly. Isolated worker processes may evaluate the same key. A
simple reference model maps the same keys without caching physical replay
counts for stateful/property verification.

### `EvaluationSession`

Owns the immutable request/model/context, resolved turbine adapter settings,
objective scale, and private `EvaluationMemo`. It provides one callback that
returns only `backend_scalar`. Its payload must be pickle-safe for existing
multi-process backends; memo and lock state are excluded from serialization and
recreated per worker process. The parent session exact-deduplicates backend
points and performs canonical full re-evaluation for retained results.

### `StructuredGridOracle`

Test-only reference value defining coordinate grids over tiny analytical
models. It enumerates points, applies the same independent feasibility checks,
and ranks physical objectives. Grid spacing and accepted objective tolerance
are explicit fields, so oracle equivalence is not asserted more tightly than
the discretization supports.

## Error Entities

Unit 2 adds specialist subclasses below `UtilityPlacementError`:

| Error | Boundary |
|---|---|
| `PlacementContextError` | Resolved scope, period, snapshot, or profile cannot form valid context |
| `PlacementTargetingError` | Existing target adapter fails independently of candidate correction |
| `PlacementThermodynamicError` | Entropy/exergy invariant or calculation boundary fails at run level |
| `PlacementCogenerationError` | Turbine settings/adapter fail independently of candidate correction |
| `PlacementOptimisationError` | Existing optimizer validation or operational failure is translated |
| `NoFeasiblePlacementError` | Bounded solve yields no proven feasible retained candidate |

Coordinate-correctable target, coverage, thermodynamic, and turbine problems
remain `CandidateDiagnostic` values and are not exception entities.

## Entity Relationships

- `ResolvedPlacementSource` is consumed to create one
  `UtilityPlacementContext` and then discarded.
- `UtilityPlacementContext` produces a `PlacementFeasibilityEnvelope`; Unit 1
  combines that envelope with the request to create `UtilityPlacementModel`.
- `EvaluationSession` combines the model and context and owns one
  `EvaluationMemo`.
- Each decoded candidate and period context produces one `PeriodAllocation`.
- A feasible allocation produces one `PeriodThermodynamicEvaluation` and,
  conditionally, one `PeriodMonetaryEvaluation`.
- Period evaluations fold into one `CandidateEvaluation`.
- Feasible normalized candidate evaluations become Unit 1 candidate/result
  contracts; internal context, slices, session, and memo are discarded.

## PBT-01 Entity Ownership

| Entity/component | Categories | Required property focus |
|---|---|---|
| Extended options | Round-trip, invariant | Defaults remain compatible; method/options map deterministically and serialize safely |
| Target/period snapshots | Invariant, idempotence | Complete finite extraction and repeat reconstruction without source mutation |
| Feasibility envelope adapter | Invariant | Exact blueprint coordinate and period coverage |
| Allocation slices/levels | Invariant, easy verification | Slice sums equal level duties; side sums satisfy residual coverage |
| Thermodynamic entities | Oracle, invariant, commutativity | Stable formulas, sign, balances, exergy identity, sum tolerance |
| Monetary entities | Oracle, invariant | Purchase/credit/net identities and unit conversion |
| Cogeneration input | Invariant | Eligible positive-duty hot-only filtering and descending order |
| Candidate evaluation | Invariant, induction | All-period feasibility and explicit weighted fold |
| Evaluation memo/session | Idempotence, easy verification | One replay per exact key and equivalence to no-cache reference command model |
| Penalty transform | Invariant, easy verification | Strict feasible/infeasible scalar separation and preserved feasible ordering |
| Structured-grid oracle/coordinator | Oracle | Best feasible result agrees within declared grid resolution/tolerance |
| Public result conversion | Round-trip, invariant | Unit 1 JSON round-trip, feasible-only order/limit, no internal objects |
| Error translation | Invariant | Stable type/code and applicable reproducibility context |

No general commutativity property applies to entities whose tuple order is
publicly meaningful. There is no recursive entity graph; induction applies to
the explicit period/contribution fold rather than object construction.
