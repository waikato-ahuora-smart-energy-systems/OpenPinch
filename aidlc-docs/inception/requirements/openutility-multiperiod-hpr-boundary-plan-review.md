# Review of OpenUtility Multi-Period HPR Optimization Boundary Plan

## Post-Correction Status

OpenUtility has resolved the contract and formulation findings identified in
the follow-up review. Its alpha schema `1.0` now enforces exact units and COP
conventions, useful-capacity basis, point and curve invariants, structured JSON
provenance, and separate `energy_balance_tolerance` and
`temperature_match_tolerance` fields. Fixed candidate capacity applies one
constant scale to every duty and electric-power point. The Pyomo model exposes
mode-aware useful duty and uses segment binaries to restrict interpolation to
adjacent part-load breakpoints.

The electricity architecture remains deliberately limited: HPR uses a
period-indexed import/export overlay, while the original non-HPR electricity
system remains scalar. This is documented accurately and does not change the
OpenPinch producer boundary, which exports physical electric-power demand only.

OpenUtility's post-correction release gate reports 186 passing tests, 90.24
percent coverage, successful Ruff, formatting, mypy, Sphinx, build, package
validation, dependency-audit, and installed-wheel smoke checks. OpenPinch and
TESPy remain absent from OpenUtility package dependencies and imports. The
consumer contract is therefore sufficiently settled for OpenPinch Units
Generation and producer implementation, while retaining its alpha stability
label until cross-package golden-fixture tests pass.

## Follow-Up Verdict on the Revised Plan and Implementation

The revised plan is architecturally aligned with OpenPinch and is a suitable
consumer boundary. It incorporates the eight amendments from the initial
review: a versioned physical map, source/sink temperature coordinates,
node-keyed periods, fixed-capacity candidates, explicit refrigeration heat
routing, local temperature-compatible curves, a plain-data dependency boundary,
and CoolProp-default/TESPy-selectable OpenPinch provenance.

The implementation also establishes the intended package split: OpenUtility no
longer declares OpenPinch as a runtime dependency, decodes map mappings into its
own immutable dataclasses, builds period-indexed HPR variables and balances, and
tests heat-pump and refrigeration solves with HiGHS. The OpenUtility release gate
reports 172 passing tests, 90.06 percent coverage, successful static checks,
package validation, dependency audit, and installed-wheel HPR smoke solves.

Approval is therefore appropriate for the ownership boundary, but not yet for a
stable `1.0` interchange contract. The current implementation has the following
issues to resolve before OpenPinch publishes a producer against it.

### P1: Ordered curves are not constrained to adjacent segments

`interpolation_topology="ordered_part_load_curve"` is decoded, but the Pyomo
model places one unrestricted convex-combination variable on every point in the
matched curve. With three or more points, it may mix nonadjacent breakpoints and
replace the intended piecewise-linear curve with its convex hull.

OpenUtility should add adjacent-segment binaries or a portable piecewise-linear
formulation, or rename and validate the topology as a deliberately global convex
hull. OpenPinch should export ordered breakpoints and must not certify that the
current consumer enforces adjacency.

### P1: Candidate capacity can disagree with physical map capacity

Map duties are absolute, but `HprCandidate.fixed_capacity` affects equipment
cost only; it does not scale or bound `q_source`, `q_sink`, or electric power.
A candidate can therefore be costed at one capacity while dispatching the map's
reference capacity.

For the fixed-capacity first release, OpenUtility should either require candidate
capacity to equal `reference_capacity` or scale every absolute map value by an
explicit constant candidate-to-map ratio. The contract should declare the useful
capacity basis: `q_sink` for heat-pump mode and `q_source` for refrigeration mode.

### P1: The HPR electricity balance is a separate overlay

The HPR period balance uses separate period-indexed grid import/export variables
and does not include the existing onsite power-generation variables, static grid
limits, transmission efficiency, or static grid balance. The implementation is
therefore a multi-period HPR overlay in the same objective, not yet one integrated
multi-period utility-system electricity balance.

This staging is acceptable if documented, but OpenUtility should not claim that
HPR competes with turbines or other onsite generation until the shared assets are
period-indexed and connected to one balance. OpenPinch should export only physical
power demand and remain independent of either formulation.

### P1: Version and semantic validation are too permissive

The decoder accepts any nonempty schema version, arbitrary unit strings, and an
arbitrary COP convention. Refrigeration maps default to a heating COP convention.
It also does not verify COP against duty and power, load fraction against useful
duty and reference capacity, unique coordinates, ordered unique load fractions,
or constant temperatures within a declared curve.

Before declaring schema `1.0` stable, both packages should agree on strict enum
values, canonical units or conversion rules, and the physical invariants. Unknown
major versions must fail closed.

### P2: Temperature and tolerance meanings need separation

OpenUtility currently reuses `balance_tolerance`, expressed in thermal-power
units, to match source/sink temperatures. The schema needs a separate temperature
matching tolerance or exact canonicalized coordinates. In schema `1.0`, source
and sink temperatures should mean external thermal-service node temperatures;
internal refrigerant evaporation and condensation temperatures belong in
provenance or optional thermodynamic detail fields.

### P2: Refrigeration dispatch uses the heat-pump duty basis

The minimum-load and variable-operating-cost equations use `q_sink` for both
modes. Refrigeration's useful service and reference-capacity basis are
`q_source`. OpenUtility should make these equations mode-aware, or define a
single derived `q_useful` expression and use it consistently.

### P2: Provenance should retain structured JSON values

The map decoder coerces every provenance value to a string. That is too narrow
for reproducible grids that need lists, numbers, booleans, nested characteristic
identifiers, convergence criteria, and package versions. The contract should
accept JSON-compatible structured metadata while continuing to reject runtime
objects.

## Complementary OpenPinch Design

OpenPinch should implement only the producer side after the requirements gate is
approved:

1. Add a strict engine-neutral map contract whose field names match the settled
   OpenUtility decoder. Include a declared capacity basis, external-service
   temperature semantics, canonical units, deterministic curve/point ordering,
   COP convention, separate balance tolerance, and structured provenance.
2. Add `simulation_backend="coolprop"` to the current
   `vapour_compression_heat_pump` and `vapour_compression_refrigeration` targeting
   entry points. TESPy is selected explicitly; cycle identity and black-box
   optimization backend remain separate concerns.
3. Keep a normal targeting call unchanged and inexpensive. Add an explicit
   follow-up operation on the same target accessor that accepts the resulting HPR
   target plus source-temperature, sink-temperature, and load-fraction grids and
   returns an `HprPerformanceMap`.
4. Put CoolProp and TESPy behind internal performance-point adapter protocols.
   Import TESPy only in its concrete adapter and ship it in an optional dependency
   extra. Do not add OpenUtility, Pyomo, or HiGHS to OpenPinch.
5. Publish a golden JSON fixture and JSON Schema, then run consumer-independent
   property tests in OpenPinch and the same fixture as a contract test in
   OpenUtility. The two packages should share data, not Python classes.

The first producer increment should use the existing cascade and parallel
vapour-compression heat-pump/refrigeration methods only, limited to configurations
that expose one external source and one external sink. Multi-port stages cannot
be flattened into OpenUtility's current single-node candidate contract. Carnot
targeting and the existing TESPy-specific Brayton implementation remain outside
backend substitution until separately designed.

## Current Recommendation

Treat the corrected OpenUtility implementation as the agreed downstream alpha
boundary. Proceed with the already approved OpenPinch producer design, publish
canonical JSON Schema and heat-pump/refrigeration golden fixtures, and verify
those fixtures independently in both packages before promoting schema `1.0`
beyond alpha.

## Initial Plan Review (Superseded by the Follow-Up Above)

### Verdict

The ownership boundary is correct, but the plan is not implementation-ready.
OpenPinch should own HPR targeting and thermodynamic simulation; OpenUtility
should own investment, sizing, dispatch, balances, economics, Pyomo, and HiGHS.
The plan needs the amendments below before its data model and MILP formulation
are locked.

### Blocking Findings

#### 1. Operating-temperature compatibility is unconstrained

The proposed point model contains `q_hot`, `q_cold`, power, and load fraction,
but omits source and sink temperature coordinates. The proposed
`lambda[h,p,k]` equations therefore allow OpenUtility to select any point for a
candidate and period, regardless of the period's actual source and sink
conditions.

Required amendment:

- carry source temperature and sink temperature on every map point;
- represent node conditions by period, or explicitly declare nodes to be fixed
  temperature levels;
- either select a valid local map cell/curve for the period conditions or
  constrain interpolation coordinates to those conditions; and
- prohibit a global convex combination over unrelated temperature points.

#### 2. Continuous sizing is claimed but not modeled

The plan says OpenUtility selects, sizes, and dispatches HPR assets, but the
listed variables contain only selection, on/off, lambda, duties, and power.
`max_capacity` plus absolute-duty map points describes a fixed-size candidate;
it does not provide an optimized installed capacity. Multiplying a variable
capacity by convex weights would also be bilinear.

Required amendment: choose one explicit first-release contract:

- fixed-capacity discrete candidates, with absolute map duties and binary
  selection; or
- normalized per-unit-capacity points plus a scale-safe linear perspective or
  disaggregated-capacity formulation, with an installed-capacity variable and
  cost basis.

The fixed-capacity interpretation is the lower-risk first implementation.

#### 3. The interchange contract is incomplete

`HprPerformancePoint(candidate, point, q_hot, q_cold, electric_power,
load_fraction)` is an internal optimization record, not a sufficient OpenPinch
interchange contract. It lacks schema version, units, temperature coordinates,
reference capacity, interpolation topology, thermodynamic backend, cycle/model
identity, and provenance.

Required amendment:

- define a versioned plain-data `HprPerformanceMap` boundary;
- make points belong to a map, not directly to an OpenUtility candidate;
- keep OpenUtility candidate economics, node assignments, and selection policy
  separate from the physical map; and
- add an OpenUtility decoder/validator for the documented mapping without
  importing OpenPinch.

#### 4. Multi-period thermal quantities are not keyed by node

The proposed `OperatingPeriod` has scalar `source_heat_available`,
`sink_heat_demand`, and `cooling_demand`, although the model introduces multiple
thermal nodes. This cannot express different availability and demand at each
node in each period.

Required amendment: use node-keyed period records or mappings for thermal
availability, heating demand, cooling demand, rejection capacity, and, where
variable, node temperature.

#### 5. Refrigeration condenser-heat routing is ambiguous

The candidate carries both `sink_node` and `rejection_node`, while the plan says
condenser heat is recovered when useful and rejected otherwise. No variable or
constraint allocates `q_hot` between those destinations.

Required amendment: either create mutually exclusive candidate routes or add
explicit recovered-heat and rejected-heat variables whose sum equals condenser
heat, with sink-demand and rejection-capacity constraints.

### Important Corrections

#### 6. Convex interpolation needs local topology or a validity certificate

Continuous lambda variables over all points create the convex hull of the whole
dataset. That can interpolate across nonadjacent load or temperature states and
can overstate performance for non-convex maps.

The first implementation should use one of these bounded approaches:

- one fixed-temperature, ordered part-load curve per candidate and period;
- explicit cells/simplices with cell-selection binaries; or
- a map certified as globally convex over the exported domain.

For HiGHS portability, explicit segment/cell binaries are safer than assuming
solver-specific SOS2 behavior.

#### 7. The OpenPinch tie-in must match the revised backend decision

The plan currently refers generically to optional TESPy map generation. It
should state that applicable existing CoolProp-backed OpenPinch HPR targeting
methods are the entry point, CoolProp remains the default, and TESPy is selected
explicitly through a distinct thermodynamic-backend argument. Analytic Carnot
methods and the existing TESPy-specific Brayton method should not silently enter
that substitution model.

#### 8. Energy-balance and power boundaries need exact semantics

The equality `q_hot = q_cold + electric_power` is valid only for a declared
steady-state system boundary with total external work and no omitted heat loss.
The map contract should define nonnegative magnitudes, whether auxiliary pumps
and fans are included in electric power, the heating/refrigeration COP
convention, and the accepted balance tolerance. Any modeled loss or auxiliary
thermal exchange must be explicit rather than hidden in a loose validation
tolerance.

### Recommended Implementation Sequence

The current OpenUtility package still requires OpenPinch at runtime and its
runner, results, scenarios, and solver surfaces explicitly describe a static
model. Converting all equipment, periods, topology, HPR, economics, reporting,
and packaging in one change would make regressions difficult to localize.

Use four independently testable increments:

1. remove the OpenPinch dependency and establish the versioned plain-data map
   decoder plus boundary tests;
2. make the existing utility system multi-period without HPR, proving that a
   one-period input reproduces current results;
3. add fixed-capacity HPR candidates using one temperature-compatible part-load
   curve per period; and
4. add temperature-cell interpolation, refrigeration heat routing, and optional
   variable sizing only after the simpler formulation is verified.

### Plan Elements to Retain

- Remove OpenUtility's runtime OpenPinch dependency.
- Keep OpenPinch and TESPy out of OpenUtility imports and package metadata.
- Use generic thermal nodes and multi-period balances.
- Keep HPR investment and dispatch in OpenUtility.
- Let avoided fuel and cooling costs emerge from balances.
- Use immutable plain dataclasses internally in OpenUtility.
- Test cold imports, package boundaries, heat-pump and refrigeration economics,
  infeasibility, and wheel isolation.

### Recommended Cross-Package Contract

OpenPinch should export a versioned physical performance map containing map
identity, operating mode, source/sink temperature coordinates, load fraction,
source-side duty, sink-side duty, total electric power, COP convention, units,
reference capacity, interpolation topology, and provenance.

OpenUtility should independently combine that map with candidate identity,
thermal-node assignments, fixed or variable capacity, costs, selection rules,
period conditions, and balance constraints. This keeps thermodynamic truth and
optimization decisions separate without requiring either package to import the
other.

### Review Conclusion

Approve the architectural boundary after the eight amendments above are added
to the OpenUtility plan. Do not start the HPR MILP implementation from the
current plan text, because it could select physically incompatible map points
and does not yet implement the promised sizing decision.
