# Unit 3 Business Logic Model

## Purpose and Boundary

Unit 3 makes the existing vapour-compression heat-pump and refrigeration target
methods the only public entry point for selecting CoolProp or TESPy, then exposes
one explicit follow-up operation that generates a versioned performance map.

The user's Functional Design answer selects full targeting semantics:
`simulation_backend="tespy"` means TESPy thermodynamic evaluations participate
inside the ordinary HPR placement optimization and determine the returned target
numbers. It is not merely provenance for a later map. Omitting the selector or
passing `coolprop` preserves the current CoolProp targeting path and numerical
contract.

This decision expands Unit 3 beyond the conceptual bridge originally left by
Unit 2. Unit 2's target-derived map simulator remains unchanged in purpose. Unit
3 adds a targeting-time thermodynamic-evaluation boundary that reuses the same
single-stage physical topology and optional TESPy leaf without making a map
request or requiring a solved target first.

OpenUtility remains outside this runtime graph. It receives only schema `1.0`
plain mappings or JSON emitted from the completed map contract.

## Public Workflow

### Targeting call

The two current public methods gain the keyword-only selector
`simulation_backend`, whose default is `coolprop`:

- `vapour_compression_heat_pump`;
- `vapour_compression_refrigeration`.

The accessor normalizes and validates the selector before it starts targeting.
It records the normalized value in replay intent and passes it independently of
the HPR cycle identity and black-box optimization method. No configuration key
overloads `HPR_TYPE` or the optimizer selector.

An ordinary call returns the same `HeatPumpTargetBase` subtype that it returns
today. It never generates a part-load grid and never attaches a CoolProp state,
TESPy network, simulator session, or map to the target.

### Explicit map call

After a successful supported target is returned, the caller may invoke
`problem.target.hpr_performance_map(target=target, request=request)`. The
operation performs four application-level steps:

1. validate that the input is a successful supported scalar HPR target;
2. extract its frozen targeting simulation record without consulting current
   mutable problem configuration;
3. convert the record to Unit 2's detached `HprTargetMapBasis`; and
4. delegate once to `generate_hpr_performance_map`.

The backend comes only from the target record. The request cannot select or
override it. The returned `HprPerformanceMap` is detached, and serialization is
caller-controlled.

## Backend Selection

### Normalization

Selector normalization accepts strings only, strips surrounding whitespace,
and compares case-insensitively. The sole normalized values are `coolprop` and
`tespy`. Empty, non-string, and unknown values fail before targeting. There is
no automatic fallback in either direction.

The normalized value is copied into:

- transient HPR targeting inputs;
- replay metadata used by selected-period, all-period, and workspace wrappers;
- normalized HPR targeting outputs; and
- `HeatPumpTargetBase.hpr_simulation_backend`.

These are strings or closed literal values only.

### CoolProp targeting

For `coolprop`, the existing vapour-compression objective functions, candidate
encoding, refrigerant selection, cycle calculations, stream construction,
penalties, accounting, optimizer, and result translation remain the reference
implementation. The selector plumbing may not introduce an alternative
calculation, new tolerance, new sort order, additional solve, or changed output
type when omitted or explicitly set to `coolprop`.

The explicit-CoolProp and omitted-selector calls are required to be equivalent
for the same problem, period, options, and deterministic optimizer inputs.

### TESPy targeting compatibility gate

TESPy targeting is supported initially only when the effective configuration is
one refrigerant loop with one evaporator and one condenser. Both heat-pump and
refrigeration modes are supported. The public method may still be named
`vapour_compression_heat_pump` or `vapour_compression_refrigeration`; the
application maps the one-stage result to the stable cycle identifier
`single_stage_vapour_compression`.

The preflight gate rejects the following before optimizer work:

- more than one active evaporator or condenser;
- parallel or cascade configurations that resolve to multiple loops;
- analytic Carnot, Brayton, MVR, integrated-expander, or other cycle families;
- shared-vector multi-period HPR optimization;
- a missing TESPy optional dependency;
- REFPROP; and
- a refrigerant specification that cannot be resolved at the required
  dew/bubble states.

A named pure fluid, a registered blend, or an explicit molar mixture with any
positive component count is not rejected categorically. The installed property
backend and requested states decide actual support.

## TESPy Targeting Evaluation

### Target candidate request

The existing placement optimizer continues to own ambient allocation, candidate
temperatures, useful-duty allocation, bounds, penalties, and ranking. At each
candidate, it creates one immutable targeting thermodynamic request containing:

- heat-pump or refrigeration mode;
- one evaporating and one condensing dew/bubble anchor in degrees Celsius;
- one useful process duty in kilowatts;
- normalized working-fluid identity and composition;
- compressor isentropic efficiency;
- superheat, subcooling, and internal-HX assumptions; and
- the fixed external approach-temperature convention.

The request contains no target, problem, optimizer, NumPy owner, or engine
object.

### Candidate design semantics

Each optimizer candidate represents a possible nominal equipment design.
Therefore the TESPy targeting evaluator performs a design solve for that
candidate, not an offdesign solve against an arbitrarily chosen earlier
candidate. This prevents optimizer order from changing the physical objective.

One lazily created evaluator session may reuse a topology shell or immutable
resource data for efficiency, but every candidate must clear all prior design
specifications, apply the complete candidate request, and solve a fresh design
state. A candidate result may not depend on which candidate ran before it.

The evaluator returns one engine-neutral result with nonnegative source duty,
sink duty, compressor-only electric power, convergence status, ordered
evaporator/condenser temperature-enthalpy profiles, working-fluid identity,
engine version, model identifier, and structured assumptions. It returns no
TESPy connection, component, bus, network, or result object.

### Objective integration

The targeting objective uses the TESPy result in the same existing accounting
pipeline used after a CoolProp cycle solve:

- heating useful duty is sink duty;
- refrigeration useful duty is source duty;
- energy balance is sink duty equals source duty plus compressor power;
- heating COP is sink duty divided by compressor power;
- cooling COP is source duty divided by compressor power;
- normalized thermal profiles become ordinary OpenPinch HPR streams; and
- existing utility, feasibility, cost, and objective calculations remain the
  sole placement accounting rules.

Candidate-local non-convergence or an invalid thermodynamic state becomes an
infeasible candidate with the existing finite failed-candidate objective. It
does not trigger CoolProp fallback and does not terminate other candidates.
Dependency absence, invalid topology, invalid selector, and evaluator lifecycle
failure are request-fatal and produce focused exceptions rather than fabricated
candidate results.

The evaluator closes exactly once after the optimization attempt, including
optimizer errors. If cleanup fails, the targeting call fails because engine
state cannot be certified as released.

## Winning Target Simulation Record

The winning normalized result carries a frozen, transport-safe targeting
simulation record. The record contains the selected backend, mode, stable cycle
and model identifiers, selected working-fluid specification, nominal
evaporating and condensing temperatures, nominal useful duty, source and sink
approaches, compressor efficiency, superheat, subcooling, internal-HX
assumption, topology counts, period identifier, engine version, power boundary,
and structured convergence/model assumptions.

CoolProp targeting populates the same record from the already solved winning
cycle without changing its legacy numerical fields. TESPy targeting populates
it from the winning engine-neutral evaluator result. Engine objects remain in
neither the record nor the target.

The domain target exposes the backend directly as
`hpr_simulation_backend`. The detailed record remains inside its normalized HPR
details and is the sole source for later map-basis extraction. This avoids using
the accessor's current configuration, which may have changed after targeting.

## Map-Basis Extraction

The pure basis builder accepts a target and applies these rules:

1. require a `HeatPumpTargetBase` instance with `hpr_success` true;
2. require one complete targeting simulation record;
3. require `single_stage_vapour_compression`, one evaporator, one condenser,
   one working-fluid specification, and scalar nominal values;
4. require the target backend and record backend to match;
5. require mode, useful-duty basis, temperatures, duty, efficiency, and
   assumptions to be finite and internally consistent; and
6. create a new `HprTargetMapBasis` with a deep-detached JSON provenance copy.

It does not inspect `target.hpr_details.model`, current problem configuration,
process streams, or private engine attributes. Repeating extraction from an
unchanged target produces an equal basis. Mutation of a returned basis or map is
prevented by frozen value contracts and cannot affect the target.

For a map request with no reference capacity, scalar nominal useful duty becomes
the map reference capacity. An explicitly supplied request capacity overrides
that scalar for the map only. It does not rewrite target results.

## Period and Wrapper Behavior

A normal method call with one selected `period_id` supports either backend and
records that period. `target.all_periods` propagates the selector to each
independent scalar replay and preserves canonical period order; each returned
period target owns its own targeting simulation record.

The first release rejects TESPy for the distinct shared-vector multi-period HPR
optimization path because one equipment design with multiple offdesign period
states requires a separate lifecycle and validation design. CoolProp retains
its existing shared-vector behavior.

The map follow-up accepts one scalar successful target, including one target
selected from an all-period result. It rejects aggregate or array-valued HPR
targets. Such a caller may choose a scalar period target and supply an explicit
fixed reference capacity.

Workspace case batches forward `simulation_backend` unchanged to isolated case
calls, preserve input case order, and report failures through the established
batch outcome structure. The map follow-up is not added to all-period or batch
dynamic mirrors in schema `1.0`; callers invoke it explicitly for one chosen
successful scalar target.

## Error Model

Invalid selectors and incompatible keyword combinations raise focused argument
validation errors before targeting. Missing TESPy uses the repository's normal
optional-dependency exception and installation guidance. Unsupported target or
map-basis extraction raises one typed HPR map compatibility error.

TESPy candidate failures preserve backend, model, operating temperatures,
working-fluid identity, and a sanitized reason in internal diagnostics. Public
errors never expose engine object representations, local temporary paths, or an
exception chain containing a TESPy network. A successful target never contains
failed-candidate diagnostics unless existing targeting diagnostics already
expose them through a stable plain-data field.

`HprMapGenerationError` remains the complete-map failure type and retains the
ordered Unit 2 diagnostics. No partial map is returned.

## Publication and Downstream Boundary

Public documentation must describe:

- omitted and explicit CoolProp targeting equivalence;
- TESPy targeting's single-stage restriction and extra installation;
- full targeting semantics selected by `simulation_backend`;
- pure, registered-blend, and explicit molar-mixture examples;
- external source/sink temperatures and dew/bubble internal anchors;
- compressor-only electricity and excluded auxiliaries;
- explicit follow-up map generation and fixed-capacity semantics;
- schema `1.0` ordered adjacent part-load interpolation;
- no automatic fallback, no partial maps, and no multi-port flattening; and
- OpenUtility's plain-data-only consumer role.

The schema and golden fixtures remain canonical packaged resources. OpenPinch
documentation may show `model_dump(mode="json")` or JSON serialization but may
not require OpenUtility as an example dependency.

## Testable Properties

The following properties satisfy PBT-01 and must be carried into Code Generation
planning:

| Component | Category | Property |
|---|---|---|
| Backend normalization | Idempotence | Normalizing a valid normalized backend again yields the same value |
| CoolProp public targeting | Oracle | Omitted selector and explicit `coolprop` produce equivalent target types and numerical fields under the same deterministic optimizer inputs |
| Selector replay | Invariant | Selected-period, all-period, and batch forwarding preserves the normalized backend and canonical result order |
| TESPy candidate evaluation | Invariant | Valid results have finite positive power, nonnegative duties, correct mode-specific useful duty and COP, and close energy balance |
| TESPy candidate evaluation | Easy verification | Every accepted result satisfies convergence, topology, profile-order, and physical consistency checks |
| TESPy evaluator lifecycle | Invariant | Candidate results are independent of prior candidate order and the session closes once on success or failure |
| Target simulation record | Round-trip | Any generated valid plain record survives model-to-JSON-to-model serialization unchanged |
| Target-to-basis extraction | Invariant | Extraction is deterministic, preserves backend/mode/fluid/nominal facts, and does not mutate the target |
| Target-to-basis extraction | Idempotence | Repeated extraction from one unchanged target returns equal frozen bases |
| Map follow-up | Invariant | Backend identity is target-owned, output is detached, and neither target nor problem state changes |
| Unsupported contexts | Invariant | Generated invalid topology, failed target, aggregate target, and backend mismatch cases are always rejected before simulation |
| Refrigerant composition | Invariant | Pure, registered-blend, and explicit N-component molar identity survives target record and map-basis extraction without categorical filtering |

No commutative or induction property is claimed. Unit 3 introduces no reusable
mutable public state machine, so stateful PBT is not required; the internal
evaluator lifecycle is covered by generated operation-order properties and
explicit examples. JSON round-trip properties reuse the Unit 1 domain
strategies and extend them with target simulation records.

## Extension Compliance

- **PBT-01**: Compliant. Every Unit 3 transformation and orchestration boundary
  has been assessed against the required property categories, and applicable
  properties are listed above for Code Generation planning.
- **Security Baseline**: Disabled in workflow state and not enforced.
- **Resiliency Baseline**: Disabled in workflow state and not enforced.
