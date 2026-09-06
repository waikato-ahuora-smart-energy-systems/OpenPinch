# Unit 3 Logical Components

## Component Inventory

| ID | Logical component | Primary owner | Responsibility |
|---|---|---|---|
| LC-U3-01 | Backend Selector Normalizer | Application target accessor | Normalize and validate public backend input |
| LC-U3-02 | HPR Replay Adapter | Application target/all-period/workspace accessors | Preserve backend intent through ordered scalar calls |
| LC-U3-03 | TESPy Compatibility Preflight | HPR analysis | Reject unsupported scope before optimizer work |
| LC-U3-04 | Target Evaluator Factory | HPR analysis | Select default or lazy optional evaluator |
| LC-U3-05 | CoolProp Target Evaluation Path | Existing HPR targeting/cycles | Preserve the exact default objective calculation |
| LC-U3-06 | TESPy Target Evaluator Session | Concrete optional analysis leaf | Perform independent candidate design solves and cleanup |
| LC-U3-07 | Exact Candidate Cache Coordinator | HPR analysis | Own the 512-entry call-local exact LRU and counters |
| LC-U3-08 | Normalized Candidate Validator | HPR analysis | Enforce duties, power, COP, profiles, and energy closure |
| LC-U3-09 | HPR Accounting Adapter | Existing HPR analysis | Feed normalized physics into existing objective/result logic |
| LC-U3-10 | Winning Simulation Record Builder | HPR analysis/contracts | Snapshot detached nominal facts and provenance |
| LC-U3-11 | HPR Target Translator | Existing service/domain boundary | Populate existing target plus backend field/details |
| LC-U3-12 | Target-to-Map Basis Builder | HPR analysis | Validate target compatibility and build Unit 2 basis |
| LC-U3-13 | Explicit Map Accessor Bridge | Application target accessor | Delegate one validated target/request to Unit 2 |
| LC-U3-14 | Map Generation and Contract Owners | Existing Units 1/2 | Produce, validate, and serialize the versioned map |
| LC-U3-15 | Documentation and Consumer Resources | Documentation/package data | Publish semantics, examples, schema, and fixtures |
| LC-U3-16 | Verification and Release Profiles | Tests/CI/build tooling | Enforce PBT, engine, architecture, docs, and artifacts |

No queue, message broker, database, web endpoint, persistent cache, circuit
breaker, service registry, scheduler, worker pool, or infrastructure component
is introduced.

## LC-U3-01: Backend Selector Normalizer

**Inputs**: the omitted sentinel or caller-provided object.

**Outputs**: normalized `coolprop` or `tespy`.

**Rules**:

- omitted becomes `coolprop`;
- valid strings are stripped and case-normalized;
- invalid values fail before `_hpr`, optimizer, or engine invocation; and
- the normalized result remains separate from HPR cycle and optimizer identity.

**Collaborators**: LC-U3-02 and LC-U3-03.

## LC-U3-02: HPR Replay Adapter

This component places the normalized backend in transient call/replay metadata
used by current selected-period, all-period, and workspace batch mechanisms. It
does not create a new configuration field or persist backend intent as a hidden
future default.

Independent all-period and workspace calls retain canonical order and own
separate evaluator lifecycles. The adapter never mirrors the explicit map
operation over aggregates.

**Collaborators**: LC-U3-01, existing `_hpr` execution, LC-U3-03, LC-U3-11.

## LC-U3-03: TESPy Compatibility Preflight

The pure preflight consumes effective HPR inputs and selected period context. It
validates backend availability, one condenser, one evaporator, one loop,
vapour-compression mode, scalar period, disabled integrated expander, resolved
working fluid, and allowed property backend.

It returns a frozen prepared specification containing stable cycle/model and
physical assumptions. It creates no TESPy network. Shared-vector multiperiod,
multi-loop, REFPROP, and unavailable required property states fail here.

**Collaborators**: LC-U3-02, Unit 2 working-fluid resolution, LC-U3-04.

## LC-U3-04: Target Evaluator Factory

The closed factory accepts the normalized backend and prepared specification.
For CoolProp it preserves LC-U3-05. For TESPy it lazily imports and constructs
LC-U3-06 plus LC-U3-07. Unknown values are impossible after LC-U3-01 but still
fail defensively.

The protocol exposes open/evaluate/close behavior through plain immutable values.
Factories are injectable in focused tests. No entry-point discovery or
third-party plugin registry is used.

## LC-U3-05: CoolProp Target Evaluation Path

This is the existing vapour-compression targeting and cycle implementation, not
a rewritten simulator. It remains the oracle for omitted and explicit CoolProp.
The only additive work is construction of LC-U3-10 from the already selected
winning result.

It may keep the existing CoolProp model in legacy internal details where current
contracts allow. No Unit 3 component reads that model to generate a map.

## LC-U3-06: TESPy Target Evaluator Session

This optional leaf owns TESPy imports, topology shell, public design-solve calls,
characteristic use, connection/component state, result extraction, temporary
state, restoration, engine version, and cleanup.

For each cache miss it applies one complete candidate request and performs one
fresh nominal design solve. It translates connections to detached ordered source
and sink temperature-enthalpy profiles and produces normalized scalar duties and
compressor power. It never returns a TESPy object.

Candidate-local failures return detached diagnostics after successful state
restoration. Restoration or cleanup failure invalidates the session.

**Collaborators**: LC-U3-03, LC-U3-07, LC-U3-08, existing Unit 2 TESPy topology
and resource helpers.

## LC-U3-07: Exact Candidate Cache Coordinator

The coordinator owns:

- an `OrderedDict`-style exact LRU capped at 512;
- callback, hit, miss, solve, insertion, and eviction counters;
- lookup/evaluate/validate/store orchestration; and
- final cache clearing.

Keys are complete frozen normalized candidate requests. Values are immutable
accepted results or candidate-local failures. Fatal failures bypass storage.
The cache is allocated after TESPy preflight and is unreachable after call
cleanup.

Current access is sequential. The component contains no lock. Its value types
avoid engine references and remain suitable for a future separate-process
design, but Unit 3 performs no serialization or distribution of cache entries.

## LC-U3-08: Normalized Candidate Validator

This engine-neutral validator checks model/backend match, convergence, finite
values, nonnegative duties, positive compressor power, mode-specific useful duty,
energy closure, COP, profile ordering, and profile-integrated duty.

It uses separately named thermodynamic tolerances and returns an immutable valid
result or candidate-local typed failure. It has no optimizer, cache, or concrete
engine dependency.

## LC-U3-09: HPR Accounting Adapter

This component maps one validated result and the existing parsed placement state
to ordinary HPR streams and the current `evaluate_vapour_hpr_result` accounting
path. It preserves utility allocation, feasibility penalties, costing, objective,
and normalized result fields.

On candidate-local failure it produces the existing failed backend candidate
shape. It never attempts another backend.

## LC-U3-10: Winning Simulation Record Builder

The builder combines the winning normalized evaluator result, prepared
specification, scalar period identity, and stable package/model metadata into one
strict `HprTargetSimulationRecord`.

CoolProp and TESPy use the same record schema. The builder deep-validates JSON
assumptions and rejects engine objects, non-finite values, mutable collections,
temporary paths, and extra fields. It performs no engine solve.

## LC-U3-11: HPR Target Translator

The existing HPR service translation copies `simulation_backend` to
`hpr_simulation_backend` and retains LC-U3-10 inside normalized HPR details. It
returns the same existing direct/indirect heat-pump/refrigeration target subtype.

It does not attach the evaluator, cache, request, map, or engine object.

## LC-U3-12: Target-to-Map Basis Builder

The pure builder accepts one target, requires success and a consistent complete
LC-U3-10 record, and maps it to the existing `HprTargetMapBasis`.

It rejects wrong type, failure, missing record, aggregate/array nominal facts,
multi-port topology, cycle mismatch, and backend mismatch using the typed map
compatibility error. It reads no current problem configuration or private model.

## LC-U3-13: Explicit Map Accessor Bridge

The public bridge validates the request type, calls LC-U3-12, and delegates once
to LC-U3-14. It accepts no backend override and does not mutate problem or target
state. The return is one detached completed map or an exception.

## LC-U3-14: Existing Map Generation and Contract Owners

Unit 2 receives the basis/request, selects its CoolProp or TESPy point simulator,
and generates one complete map. Unit 1 validates and serializes it. Unit 3 does
not duplicate these components or alter their all-or-nothing semantics.

For TESPy, the winning nominal target record supplies the global map design
condition; map grid points remain offdesign evaluations of that selected design.

## LC-U3-15: Documentation and Consumer Resources

This component comprises API/reference prose, optional-install guidance,
executable lightweight examples, limitation tables, package schema, golden
fixtures, and the Unit 2 characteristic resource.

It explains full TESPy targeting semantics, independent candidate design solves,
winning-design map offdesign behavior, mixtures, scalar/all-period support,
shared-vector rejection, compressor-only power, fixed capacity, interpolation,
failure behavior, and plain-data OpenUtility consumption.

## LC-U3-16: Verification and Release Profiles

This logical component owns:

- deterministic fake evaluator/cache/lifecycle examples and properties;
- the existing CoolProp default oracle;
- small real TESPy targeting/map integration;
- 512-entry, 64-MiB, call-count, cleanup, and 300-second performance gates;
- cold-import and forbidden-dependency architecture tests;
- API/root/CLI drift tests;
- warning-strict documentation;
- source/wheel build and resource parity; and
- isolated core and TESPy artifact smokes.

Hypothesis uses seed `20260715` or logs a replay seed and retains shrinking. Both
base and TESPy source/artifact profiles are blocking.

## Control Flow

### Default CoolProp target

1. LC-U3-01 normalizes omitted or explicit CoolProp.
2. LC-U3-02 carries intent through existing execution.
3. LC-U3-05 performs the unchanged target optimization.
4. LC-U3-10 snapshots the existing winning result without another solve.
5. LC-U3-11 returns the existing target type.

### Explicit TESPy target

1. LC-U3-01 normalizes explicit TESPy.
2. LC-U3-02 resolves the scalar execution context.
3. LC-U3-03 fails fast or returns the prepared specification.
4. LC-U3-04 constructs one call-owned LC-U3-06/LC-U3-07 boundary.
5. For each objective callback, LC-U3-07 returns an exact hit or calls LC-U3-06.
6. LC-U3-08 validates a miss result; LC-U3-09 produces existing accounting.
7. Existing optimization selects a winner.
8. LC-U3-10 snapshots the winning normalized result.
9. LC-U3-11 returns the existing target type.
10. LC-U3-06/LC-U3-07 close and clear once through final cleanup.

### Explicit map follow-up

1. LC-U3-13 receives a successful target and validated request.
2. LC-U3-12 builds a detached basis from LC-U3-10.
3. LC-U3-14 produces and validates one complete map using the target backend.
4. LC-U3-13 returns the detached contract without state mutation.

## Allowed Dependency Matrix

Rows may depend on listed columns only.

| Consumer | Allowed providers |
|---|---|
| LC-U3-01 | Standard library |
| LC-U3-02 | LC-U3-01, existing application runtime/replay |
| LC-U3-03 | HPR contracts/domain configuration, Unit 2 engine-neutral fluid helpers |
| LC-U3-04 | Evaluator protocol, LC-U3-05, lazy LC-U3-06 |
| LC-U3-05 | Existing HPR analysis, CoolProp |
| LC-U3-06 | TESPy, CoolProp property wrapper, Unit 2 optional-leaf resources/helpers |
| LC-U3-07 | Standard library, evaluator protocol, frozen candidate values |
| LC-U3-08 | Standard library, frozen candidate values/errors |
| LC-U3-09 | Existing HPR parsed state, streams, accounting, normalized result |
| LC-U3-10 | HPR contract record, normalized result, package metadata |
| LC-U3-11 | Existing HPR service/domain targets, HPR output contract |
| LC-U3-12 | Domain target, HPR record, Unit 2 basis, compatibility error |
| LC-U3-13 | Public request/map contracts, LC-U3-12, LC-U3-14 |
| LC-U3-14 | Existing Unit 1 and Unit 2 only |
| LC-U3-15 | Public APIs and packaged plain resources only |
| LC-U3-16 | Public/specialist APIs, fakes, tools, and artifacts under test |

No allowed row includes OpenUtility, Pyomo, HiGHS, a reverse application import,
or a TESPy import outside LC-U3-06.

## NFR-to-Component Traceability

| NFR | Components |
|---|---|
| NFR-U3-001 | LC-U3-01 through LC-U3-16, especially LC-U3-16 |
| NFR-U3-002 | LC-U3-01, LC-U3-13, LC-U3-16 |
| NFR-U3-003 | LC-U3-01, LC-U3-05, LC-U3-10, LC-U3-16 |
| NFR-U3-004 | LC-U3-03, LC-U3-04, LC-U3-06, LC-U3-16 |
| NFR-U3-005 | LC-U3-06, LC-U3-15, LC-U3-16 |
| NFR-U3-006 | LC-U3-03 through LC-U3-16 |
| NFR-U3-007 | LC-U3-01, LC-U3-16 |
| NFR-U3-008 | LC-U3-06 through LC-U3-08, LC-U3-16 |
| NFR-U3-009 | LC-U3-08, LC-U3-10, LC-U3-12, LC-U3-14 |
| NFR-U3-010 | LC-U3-03, LC-U3-08, LC-U3-09, LC-U3-12, LC-U3-14 |
| NFR-U3-011 | LC-U3-08, LC-U3-16 |
| NFR-U3-012 | LC-U3-03, LC-U3-06, LC-U3-10, LC-U3-12, LC-U3-15, LC-U3-16 |
| NFR-U3-013 | LC-U3-05 through LC-U3-08, LC-U3-16 |
| NFR-U3-014 | LC-U3-07, LC-U3-16 |
| NFR-U3-015 | LC-U3-07, LC-U3-16 |
| NFR-U3-016 | LC-U3-03 through LC-U3-14, LC-U3-16 |
| NFR-U3-017 | LC-U3-06, LC-U3-07, LC-U3-16 |
| NFR-U3-018 | LC-U3-04, LC-U3-06, LC-U3-07, LC-U3-15 |
| NFR-U3-019 | LC-U3-04, LC-U3-06, LC-U3-07, LC-U3-16 |
| NFR-U3-020 | LC-U3-03, LC-U3-06 through LC-U3-09, LC-U3-16 |
| NFR-U3-021 | LC-U3-04 through LC-U3-09, LC-U3-16 |
| NFR-U3-022 | LC-U3-10, LC-U3-11, LC-U3-16 |
| NFR-U3-023 | LC-U3-12 through LC-U3-14, LC-U3-16 |
| NFR-U3-024 | LC-U3-03, LC-U3-06, LC-U3-08, LC-U3-12, LC-U3-16 |
| NFR-U3-025 | LC-U3-01, LC-U3-02, LC-U3-11, LC-U3-16 |
| NFR-U3-026 | LC-U3-15, LC-U3-16 |
| NFR-U3-027 | LC-U3-13 through LC-U3-16 |
| NFR-U3-028 | LC-U3-04, LC-U3-06 through LC-U3-09, LC-U3-16 |
| NFR-U3-029 | LC-U3-16 |
| NFR-U3-030 | LC-U3-16 |
| NFR-U3-031 | LC-U3-16 |
| NFR-U3-032 | LC-U3-14 through LC-U3-16 |
| NFR-U3-033 | LC-U3-01 through LC-U3-16 |

Every NFR-U3-001 through NFR-U3-033 requirement has a logical owner and an
explicit verification component.

## Extension Compliance

- **Property-Based Testing**: Compliant. LC-U3-16 owns every applicable
  PBT-01 through PBT-10 verification; LC-U3-01, LC-U3-06 through LC-U3-14 expose
  deterministic boundaries suitable for the identified properties.
- **Security Baseline**: Disabled. Security infrastructure components are N/A;
  LC-U3-03, LC-U3-08, LC-U3-10, LC-U3-16 provide required ordinary integrity
  and diagnostic hygiene.
- **Resiliency Baseline**: Disabled. Distributed resilience components are N/A;
  LC-U3-06 through LC-U3-09 provide required in-process failure isolation and
  cleanup.
