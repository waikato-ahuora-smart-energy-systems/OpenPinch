# TESPy HPR Performance-Map Requirements

## Intent Analysis

- **Request type**: Design-only enhancement assessment
- **Scope**: Multiple OpenPinch owners (`analysis`, `contracts`, optional dependency
  packaging, tests, and documentation) with an external plain-data boundary to
  downstream optimization packages
- **Complexity**: Moderate; the schema and ownership decisions are clear, while a
  reusable TESPy cycle template needs later validation against real cycle models
- **Implementation status**: Out of scope for this request unless separately
  approved after design review

## Objective

Define where and how OpenPinch could optionally generate heat-pump and
refrigeration part-load performance maps with TESPy while preserving these
boundaries:

1. OpenPinch owns pinch/HPR targeting context and thermodynamic map generation
   through its current HPR targeting methods.
2. CoolProp remains the default thermodynamic backend; TESPy is an explicitly
   selected optional, leaf-level backend.
3. OpenUtility and other optimizers consume only a stable plain-data contract and
   do not import OpenPinch or TESPy.
4. OpenPinch does not take ownership of downstream Pyomo, HiGHS, SOS2, or
   segment-binary model construction.

## Functional Requirements

### FR-1: Engine-neutral map contract

OpenPinch shall define a dedicated, versioned performance-map contract that is
independent of TESPy runtime objects, OpenPinch domain objects, NumPy arrays, and
solver models. `model_dump(mode="json")` shall produce a payload containing only
JSON-compatible mappings, lists, strings, booleans, numbers, and null values.

### FR-2: Required operating-point data

Each feasible point shall identify:

- source temperature;
- sink temperature;
- load fraction;
- source-side thermal duty;
- sink-side thermal duty;
- electrical power;
- heating or refrigeration COP under an explicitly declared convention; and
- units or one map-level unit basis sufficient to interpret every numeric field.

The schema shall use nonnegative flow magnitudes and an explicit operating mode,
leaving balance signs to the downstream consumer. It shall retain both thermal
sides so a utility optimizer can connect the unit to source, sink, and electricity
balances without reconstructing thermodynamics from COP alone.

For schema `1.0`, source and sink temperatures shall denote the external
thermal-service boundary used to match downstream thermal nodes. Refrigerant
evaporation/condensation temperatures and approach temperatures shall be retained
as provenance or optional thermodynamic details, not substituted for the external
coordinates.

The reference-capacity basis shall be explicit. Heat-pump maps use delivered
sink duty and refrigeration maps use extracted source duty unless a future schema
declares another basis. Each point's load fraction shall agree with useful duty
divided by reference capacity within a declared tolerance.

### FR-3: Grid and interpolation semantics

The contract shall declare the independent dimensions, point ordering, reference
capacity, and whether the points form a complete rectilinear grid or a documented
sparse set. It shall not prescribe a Pyomo formulation. Downstream packages may
translate the same points to convex-combination, SOS2, triangulated, or
segment-binary formulations.

The contract shall also define local interpolation topology, such as ordered
fixed-temperature curves or valid cells/simplices. A downstream optimizer shall
not form one unrestricted convex combination across unrelated source/sink
temperature conditions. For multi-period dispatch, the selected point or cell
must be compatible with the source and sink temperatures applicable to that
period.

An `ordered_part_load_curve` shall mean interpolation between adjacent
breakpoints only. Point order alone shall not be treated as sufficient evidence
that a downstream unrestricted convex combination preserves the curve.

### FR-4: Version and provenance

The exported payload shall include a strict schema version and provenance
identifying at least the generator type, OpenPinch version, TESPy version when
used, cycle/model identifier, and material modeling assumptions. Unknown schema
versions shall fail validation rather than being silently coerced.

Provenance shall allow structured JSON-compatible values so versions,
characteristic identifiers, arrays of modeling assumptions, convergence criteria,
and auxiliary-power inclusions can be preserved without string coercion.

### FR-5: Thermodynamic-backend selection on existing HPR methods

The current CoolProp-backed HPR targeting methods shall remain the user-facing
tie-in point. Applicable methods shall accept an explicit thermodynamic backend
selector whose default is `coolprop` and whose optional value is `tespy`. Omitting
the selector shall preserve current behavior and numerical results.

Backend selection shall be independent of the existing HPR cycle identity and
black-box optimization backend. The API and documentation shall use a distinct
name such as `simulation_backend` or `thermodynamic_backend` so it cannot be
confused with cycle topology or the optimization algorithm.

The first supported TESPy path shall be limited to CoolProp-backed simulated-cycle
methods for which result parity can be defined. Analytic Carnot methods shall not
be relabeled as CoolProp-backed, and unsupported cycle/backend combinations shall
raise a clear validation error rather than silently falling back. The existing
TESPy-specific Brayton implementation shall remain unchanged unless a later design
explicitly consolidates it into the same backend abstraction.

### FR-6: TESPy grid generation

An optional TESPy-backed generator shall evaluate the Cartesian product of source
temperatures, sink temperatures, and load fractions. Because TESPy does not define
one universal HPR cycle, the generator shall execute through an explicit cycle
adapter or model factory that defines design/offdesign setup, point inputs, result
extraction, and convergence criteria. It shall be invoked through the same HPR
application and analysis route selected by the existing targeting method, not
through a second unrelated public workflow.

### FR-7: Failure policy

The generator shall use an explicit policy for non-converged points: either fail
the complete generation request or omit invalid points while recording structured
diagnostics. The default shall reject an incomplete map so downstream optimization
cannot silently interpolate across unverified operating regions.

### FR-8: HPR targeting and map-export bridge

OpenPinch shall provide a pure bridge inside the existing HPR targeting route that
can derive a map-generation request or nominal basis from an already solved or
prepared heat-pump/refrigeration target. The bridge may reuse mode, cycle identity,
capacity, evaporating/condensing temperatures, and period metadata, but it shall
not mutate the target or attach TESPy objects to the target schema.

Map generation and export shall remain an explicit option or follow-up operation
on the existing targeting call. A normal target call shall continue to return the
current targeting result and shall not pay the cost of evaluating a part-load grid.

The preferred public shape is a follow-up operation on the existing target
accessor that consumes an HPR target and an explicit grid request. This preserves
the current targeting return type while keeping map generation visibly connected
to `vapour_compression_heat_pump` or
`vapour_compression_refrigeration` rather than creating an unrelated workflow.

### FR-9: Downstream independence

OpenPinch shall publish a JSON example or fixture and field semantics that are
sufficient for OpenUtility to implement its own parser and MILP translation.
Neither package shall require the other at runtime. OpenPinch shall not import
OpenUtility, Pyomo, or HiGHS for this feature.

The exported physical map shall be independent of an OpenUtility investment
candidate. OpenUtility shall combine a map identifier with its own thermal-node
assignments, capacity choice, costs, periods, and selection policy. OpenPinch
shall not encode those downstream optimization decisions in each performance
point.

### FR-10: Optional dependency isolation

TESPy imports shall occur only in the concrete TESPy generator leaf. Importing
OpenPinch, its contracts, or the engine-neutral performance-map package shall
succeed when TESPy is absent. Missing-dependency errors shall use the repository's
standard optional-dependency guidance.

## Non-Functional Requirements

- **Determinism**: Equivalent inputs, model assumptions, and dependency versions
  shall produce stable point ordering and structurally equivalent payloads.
- **Validation**: Reject duplicate coordinates, invalid load fractions,
  non-finite values, negative magnitudes, inconsistent units, invalid COPs, and
  energy-balance violations outside a declared tolerance.
- **Tolerance dimensions**: Energy-balance and temperature-matching tolerances
  shall be separate quantities with declared units; neither package may reuse one
  numeric tolerance for both dimensions.
- **System boundary**: Define whether total electric power includes auxiliary
  pumps and fans, define the heating/refrigeration COP convention, and represent
  any heat loss or additional external energy flow explicitly.
- **Testability**: The grid driver shall be testable with a fake cycle adapter;
  most tests shall not require TESPy or solve a real thermodynamic network.
- **Compatibility**: Schema evolution shall be explicit and versioned. Additive
  metadata changes may be allowed only if the version policy documents them.
- **Backward compatibility**: Existing HPR calls without a backend selector shall
  continue to use CoolProp and preserve their current return types and semantics.
- **Environment safety**: A small real-TESPy smoke test may be optional/slow; the
  core suite shall remain reliable without TESPy.
- **Maintainability**: Existing runtime `HeatPumpTargetOutputs`,
  `HPRBackendResult`, and `HeatPumpTargetBase` shall not become the transport map
  contract because they contain runtime/domain artifacts and different semantics.

## Non-Goals

- Implementing Pyomo constraints, HiGHS execution, unit commitment, dispatch, or
  storage optimization in OpenPinch.
- Serializing TESPy networks, components, connections, characteristic objects, or
  result buses.
- Claiming a generic TESPy heat-pump model without selecting and validating an
  explicit cycle topology and characteristic data.
- Automatically linearizing arbitrary scattered data into a globally valid MILP
  surface in the first increment.
- Implementing the feature during this design-only request.

## Acceptance Criteria for a Future Implementation

1. Contract models round-trip through JSON and reject unsupported versions.
2. Core and contract imports pass in a process where TESPy imports are blocked.
3. A fake adapter proves deterministic grid traversal, COP calculations, failure
   handling, validation, and provenance.
4. Property-based tests cover serialization round trips and physical/schema
   invariants over generated valid maps.
5. A guarded real-TESPy smoke test proves at least one design/offdesign sequence
   can populate the contract.
6. A checked-in JSON fixture is consumable without importing OpenPinch and is
   documented for downstream MILP packages.
7. Existing HPR targeting behavior and public package-root exports remain
   unchanged when the backend selector is omitted.
8. Focused tests prove that the applicable existing targeting methods dispatch to
   CoolProp by default, dispatch to TESPy only when selected, and reject unsupported
   cycle/backend combinations without fallback.
9. A downstream contract test proves that period conditions cannot select or
   interpolate performance points from incompatible source/sink temperatures.
10. Reference-capacity semantics distinguish fixed-capacity absolute maps from
    normalized scalable maps, preventing an undocumented bilinear sizing model.
11. For an ordered curve containing at least three breakpoints, a downstream
    contract test proves that interpolation cannot mix nonadjacent points.
12. Heat-pump and refrigeration fixtures prove their respective useful-capacity,
    load-fraction, and COP conventions.
13. The OpenUtility fixture consumer rejects unknown schema versions and a
    candidate capacity that is inconsistent with an absolute physical map unless
    an explicit constant scaling rule is present.

## Requirements Completeness Assessment

The request is sufficiently specific for design without clarification. The
existing extension decisions remain applicable: Security Baseline disabled,
Resiliency Baseline disabled, and Property-Based Testing enabled. The main
remaining uncertainties are the first supported TESPy cycle topology and
characteristic dataset, plus finalization of the OpenUtility schema semantics and
adjacent-segment formulation. They do not block the ownership or API-boundary
recommendation, but the interchange contract shall not be declared stable until
the cross-package fixture passes both implementations.
