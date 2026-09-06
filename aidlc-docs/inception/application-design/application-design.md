# Application Design Summary

The package-wide architecture modernization supersedes the earlier
stream-focused package-placement detail where ownership differs. Its current
design record is
`package-architecture-modernization-design.md` in this directory.

The design introduces an ordered child profile within the existing `Stream` aggregate. Collection and zone boundaries remain parent based. A shared segment projection prevents each numerical service from inventing its own flattening rules. HEN synthesis receives parent axes and segment tensors, preserving topology while replacing constant-CP equations with cumulative heat-coordinate relations. Reporting remains parent first and nests segment detail explicitly.

The approved package-usability refactor is documented separately in
`package-usability-refactor-design.md`. It preserves the two-class root facade,
adds descriptive target/design/workspace accessors, separates execution from
observation, and makes the tutorial/RTD manifest an enforced public-contract
consumer without overwriting the segmented-stream design above.

## Repository Issue Remediation Design

The remediation retains all existing public component boundaries. Workspace
case identifiers gain one shared strict validator used by runtime and bundle
entry points, with a second containment check at batch export. `problem_data`
becomes a detached observation boundary; explicit application methods remain the
only supported mutation paths. Reporting uses exclusive workbook reservation,
and the comparison tool uses one scoped import context that verifies every
OpenHENS module against the requested checkout before injecting its factory into
execution. Current documentation is corrected through a scoped drift guard that
does not rewrite historical records. No new runtime dependency or root export is
introduced.

## Utility Placement Optimisation Design

### Design outcome

Utility placement is an additive specialist analysis under
`OpenPinch.analysis.utility_placement`, reached through
`problem.target.utility_placement(...)`. The application layer owns unique
hierarchy-zone resolution, zone-type-derived scope, existing-utility template
inference, and period resolution; analysis owns templates, bounds, vector
conversion, candidate replay, coverage, objectives, optimisation coordination,
and result assembly. Existing direct, Total Site, aggregate indirect targeting,
and solver-neutral
optimisation are composed at explicit adapter boundaries. Contracts live in
`OpenPinch.contracts.utility_placement` and the
two-symbol package-root facade remains unchanged.

### Public contract

The primary method accepts optional concise `isothermal` and `sensible` counts,
an optional zone, period selection, and typed options. Existing utilities supply
typed templates when counts are omitted; supplied counts generate paired hot
and cold templates. Zone type selects the target profile. The method immediately
constructs the immutable internal request. A problem with canonical
periods uses one shared placement across those periods, and the all-period
accessor makes that behavior explicit rather than launching independent solves.

The public return is a detached normal `PinchProblem` whose utility input is
the best feasible set. Detailed JSON-serializable placement evidence remains at
`optimized_case.utility_placement_result`. `workspace.add(case, name=...,
activate=False)` explicitly registers the case while preserving that evidence.

### Internal component model

1. The application context builder runs existing targeting only against an
   isolated execution-zone copy and extracts immutable period profiles.
2. The pure template model validates counts and identities, derives feasible
   bound intersections, generates starts, and encodes or decodes placements.
3. The optimisation coordinator delegates bounded search to the existing
   solver-neutral service.
4. The candidate engine replays one decoded placement against every period,
   verifies full heating and cooling coverage, and calls one objective evaluator.
5. The thermodynamic evaluator integrates entropy stably in absolute
   temperature and reports ambient-temperature exergy destruction.
6. The application adapter converts only the best candidate's shared
   temperatures into canonical utilities on a new unsolved normal case and
   attaches the complete detached evidence.
7. The existing tutorial generator produces one thermodynamic-only notebook
   using the concise return/add/target/plot workflow; manifest, execution, and
   package-data gates own its delivery evidence.

### Error and state model

Placement-specific exceptions distinguish invalid requests, incompatible
scope, context preparation, empty feasible bounds, targeting/allocation,
non-finite thermodynamics, and optimiser exhaustion.
Ordinary candidate infeasibility is structured data until solve completion.
Feasible candidates always outrank penalties, and an all-infeasible run raises
instead of returning a least-infeasible placement. The source problem,
utilities, configuration, heat targets, workspace selection, and cached study
inputs are unchanged on success and failure.

### Alternatives considered

| Alternative | Decision | Reason |
|---|---|---|
| Put placement equations in `_TargetAccessor` | Rejected | Violates the existing orchestration-only application boundary and makes pure testing harder. |
| Mutate source utility streams | Rejected | The optimized utilities belong to a detached normal case; the source remains unchanged. |
| Solve every period independently through the generic all-period loop | Rejected | Violates the one-placement/all-period feasibility requirement. |
| Add placement logic directly to the general optimiser package | Rejected | Utility thermodynamics are domain-specific; the optimiser should remain solver-neutral. |
| Implement a new optimisation backend | Rejected | Existing bounded, seeded backends already satisfy the architectural need and avoid a new dependency. |
| Reimplement direct or Total Site physics | Rejected | Existing target services remain canonical and are composed through detached context. |
| Create a new root export | Rejected | Specialist imports plus the existing target accessor preserve the stable two-symbol root facade. |

### Requirements and story traceability

| Design area | Requirements | Stories |
|---|---|---|
| Public accessors, scopes, periods, batches | FR-001, FR-007, FR-008, FR-016 | UPO-02, UPO-06, UPO-10 |
| Request and template contracts | FR-002 through FR-006 | UPO-01, UPO-08, UPO-09 |
| Candidate coverage and constraint model | FR-005, FR-006, FR-008, FR-012 | UPO-03, UPO-06, UPO-08, UPO-11 |
| Thermodynamic evaluator | FR-009 | UPO-04, UPO-11 |
| Monetary capability exclusion | FR-010 | UPO-05 |
| Optimisation coordination and alternatives | FR-011, FR-012, FR-014 | UPO-07, UPO-08, UPO-11 |
| Detached optimized case, evidence, and workspace registration | FR-013 through FR-015 | UPO-07, UPO-09 |
| Executable notebook delivery | FR-017 | UPO-02, UPO-12 |
| Numerical, compatibility, maintainability, diagnostics | NFR-001 through NFR-006 | UPO-08, UPO-09, UPO-11, UPO-12 |
| No new infrastructure boundary | NFR-007 | UPO-12 |

All FR-001 through FR-017, NFR-001 through NFR-007, and UPO-01 through UPO-12
have an owning component, interface, and orchestration path. Detailed equations,
tolerance values, penalties, and property definitions remain intentionally
deferred to per-unit Functional Design.

Validation set: FR-001, FR-002, FR-003, FR-004, FR-005, FR-006, FR-007,
FR-008, FR-009, FR-010, FR-011, FR-012, FR-013, FR-014, FR-015, FR-016,
FR-017;
NFR-001, NFR-002, NFR-003, NFR-004, NFR-005, NFR-006, NFR-007; UPO-01,
UPO-02, UPO-03, UPO-04, UPO-05, UPO-06, UPO-07, UPO-08, UPO-09, UPO-10,
UPO-11, and UPO-12.

### Extension compliance at Application Design

- **PBT-01 through PBT-10**: N/A for blocking enforcement at Application
  Design. The enabled extension's applicability matrix begins formal property
  identification at Functional Design. This design nevertheless isolates pure
  round-trip, invariant, oracle, reproducibility, and non-mutation boundaries
  so those obligations can be assigned without architectural rework.
- **Security Baseline**: skipped because it is disabled for this feature.
- **Resiliency Baseline**: skipped because it is disabled for this feature.

## TESPy HPR Performance-Map Design

### Design outcome

OpenPinch owns HPR targeting, thermodynamic point simulation, and production of
a strict fixed-capacity performance map. OpenUtility owns candidate definitions,
thermal-node assignment, period data, investment and operating costs, selection,
dispatch, piecewise-linear Pyomo constraints, and HiGHS execution. Neither package
imports the other. The stable integration surface is versioned JSON-compatible
data plus shared golden fixtures.

The existing `vapour_compression_heat_pump(...)` and
`vapour_compression_refrigeration(...)` methods remain the entry points. Each gains
`simulation_backend="coolprop"`; `tespy` is explicit. Ordinary calls keep current
return types. A separate
`problem.target.hpr_performance_map(target=..., request=...)` follow-up performs
the potentially expensive grid simulation.

### Supported first-release topology

OpenUtility's alpha contract models one source node and one sink or rejection
node. OpenPinch schema `1.0` therefore exports only a single-source/single-sink,
fixed-capacity vapour-compression unit. Current cascade or parallel targeting
configurations with multiple externally active evaporators, condensers, or
coupled stages cannot be flattened into this contract and are rejected clearly.

The first TESPy adapter implements one explicitly documented single-stage
vapour-compression heat-pump/refrigeration network. Analytic Carnot, Brayton, MVR,
multi-port cascade, and multi-port parallel maps require later schemas or adapter
designs. This restriction preserves the requested current-method tie-in without
claiming that every existing HPR configuration fits OpenUtility's single-unit
boundary.

### Public and transport contracts

`OpenPinch.contracts.hpr_performance_map` contains the isolated request, units,
point, map, and JSON-value types. It fixes these schema `1.0` semantics:

- `schema_version` is exactly `1.0`; unknown versions fail closed.
- `mode` is `heat_pump` or `refrigeration`.
- `source_temperature` and `sink_temperature` are external service coordinates
  in `degC`.
- `q_source`, `q_sink`, and total external `electric_power` are nonnegative `kW`
  magnitudes.
- Heat-pump capacity and load fraction use `q_sink`; refrigeration uses
  `q_source`. `reference_capacity_basis` makes this explicit.
- Heating COP is `q_sink / electric_power`; cooling COP is
  `q_source / electric_power`.
- `q_sink = q_source + electric_power` holds within the map's power-balance
  tolerance. Schema `1.0` does not hide thermal losses.
- Each fixed-temperature curve has unique, ascending load fractions and permits
  interpolation only between adjacent breakpoints.
- Structured provenance records OpenPinch and engine versions, cycle identity,
  refrigerant, approach temperatures, auxiliary-power boundary, characteristics,
  convergence policy, and design/offdesign assumptions.

The external field remains `thermodynamic_backend` to match OpenUtility's
implemented decoder; `simulation_backend` is the OpenPinch method argument and
target-result terminology. Both carry the same normalized value.

`energy_balance_tolerance` applies to the thermal-power, useful-capacity, and
COP consistency checks in `kW`. `temperature_match_tolerance` is a distinct
schema field used by OpenUtility only when matching canonical `degC` source and
sink coordinates; OpenPinch does not reuse it for thermodynamic validation.

### Component and service model

1. The target accessor validates the backend selector and records it as runtime
   replay intent.
2. Existing HPR service orchestration selects an internal point simulator while
   preserving the default CoolProp handler path.
3. The returned target stores only a normalized backend string; no engine object
   enters domain or transport state.
4. The map bridge validates a successful supported target and builds an immutable
   generation context from its configuration and explicit request.
5. The grid service traverses coordinates deterministically and operates through
   one simulator-session protocol.
6. CoolProp adapts current cycle calculations. TESPy owns lazy network setup,
   design/offdesign solves, result extraction, and cleanup in its leaf module.
7. The grid service rejects incomplete maps, normalizes units and signs, validates
   all invariants, and returns a detached map contract.
8. JSON Schema and heat-pump/refrigeration fixtures provide the consumer boundary.

### OpenUtility preconditions

OpenPinch must not declare schema `1.0` stable until the OpenUtility consumer:

- enforces adjacent-segment interpolation for `ordered_part_load_curve`;
- requires candidate capacity to equal map reference capacity or applies one
  documented constant scale to all duties and power;
- rejects unknown schema versions and incompatible units/COP conventions;
- validates coordinates, curve structure, load fraction, useful capacity, COP,
  and energy balance;
- uses `q_source` as useful refrigeration duty; and
- separates power-balance tolerance from temperature matching.

The separate HPR electricity overlay may remain an explicitly documented alpha
limitation, but it is not evidence of competition with existing onsite generation
until OpenUtility creates one period-indexed electricity balance.

### Alternatives considered

| Alternative | Decision | Reason |
|---|---|---|
| Make TESPy a required dependency | Rejected | Environment-sensitive simulation must not affect normal imports or CoolProp targeting. |
| Put map generation in OpenUtility | Rejected | OpenUtility owns optimization, not thermodynamic cycle simulation. |
| Return a map from every target call | Rejected | Grid simulation is expensive and would change established return semantics. |
| Add map methods to domain target objects | Rejected | Domain results would need application/analysis behavior and optional-engine knowledge. |
| Relabel the existing TESPy Brayton cycle as a backend | Rejected | Cycle topology and simulation backend are separate dimensions. |
| Export current `HeatPumpTargetOutputs` | Rejected | It contains runtime/domain artifacts and lacks interchange topology semantics. |
| Export multi-stage targets as one node pair | Rejected | It loses coupled ports and permits physically invalid downstream dispatch. |
| Create a shared Python contract package | Deferred | A JSON Schema and fixtures achieve language-neutral decoupling with less release coordination. |

### Requirements traceability

| Design area | Requirements and acceptance criteria |
|---|---|
| Isolated strict transport contract | FR-1 through FR-4; acceptance 1, 4, 6, 10 through 13 |
| Current-method selector and compatibility | FR-5; acceptance 7 and 8 |
| Point simulators and deterministic grid | FR-6, FR-7; acceptance 3 and 5 |
| Explicit target-accessor bridge | FR-8 |
| OpenUtility-independent data boundary | FR-9; acceptance 6, 9, and 13 |
| Optional TESPy leaf | FR-10; acceptance 2 and 5 |

All FR-1 through FR-10 and acceptance criteria 1 through 13 have an owning
component, interface, orchestration path, and verification boundary. Detailed
TESPy component parameters, numeric tolerances, and characteristic datasets
remain for a future Functional Design after implementation authorization.

### Extension compliance at Application Design

- **PBT-01 through PBT-10**: N/A for blocking enforcement at Application Design.
  The extension matrix begins formal enforcement at Functional Design. This
  design nevertheless isolates serialization round trips, deterministic ordering,
  physical invariants, fake-adapter oracles, and default-CoolProp parity so those
  properties can be assigned without architectural changes.
- **Security Baseline**: skipped because it is disabled for this feature.
- **Resiliency Baseline**: skipped because it is disabled for this feature.
- **Finding status**: No applicable enabled-extension blocking findings.
