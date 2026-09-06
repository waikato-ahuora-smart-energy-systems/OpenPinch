# Unit 2 Logical Components

## Component Inventory

| Component | Responsibility | Runtime dependency |
|---|---|---|
| LC-U2-01 Internal values and context builder | Frozen target basis, fluid, context, point, result, settings, and metadata values | Standard library and Unit 1 request |
| LC-U2-02 Working-fluid resolver | Parse/normalize fluid specifications, reject REFPROP, and prove property capability | CoolProp 8 or newer and existing fluid helper |
| LC-U2-03 Canonical point iterator | Lazily expand the source/sink/load Cartesian grid | LC-U2-01 |
| LC-U2-04 Simulator protocol and factory | Define lifecycle seam and select one adapter without eager TESPy import | Standard typing and local imports |
| LC-U2-05 Generation coordinator | Prepare, traverse, normalize, accumulate, close, and return map or error atomically | Unit 1 contracts and LC-U2-01 through LC-U2-07 |
| LC-U2-06 Diagnostics and internal failures | Stable codes, point/session classification, sanitization, and aggregate error | Standard library and LC-U2-01 |
| LC-U2-07 Provenance builder | Assemble deterministic recursive JSON evidence | LC-U2-01, settings, and resource metadata |
| LC-U2-08 CoolProp adapter | Delegate points to the existing vapour-compression cycle | CoolProp and existing HPR cycle |
| LC-U2-09 TESPy convergence settings | Own fixed versioned public solve arguments | Frozen standard-library value |
| LC-U2-10 Characteristic resource loader | Strictly load/version/hash compressor characteristic data | `importlib.resources`, `json`, and `hashlib` |
| LC-U2-11 TESPy adapter | Build design network, restore snapshot, solve offdesign, and extract results | Optional TESPy leaf plus LC-U2-09 and LC-U2-10 |
| LC-U2-12 Fake simulator and PBT model | Deterministic outcomes, failure injection, lifecycle state machine, and domain strategies | pytest and Hypothesis; tests only |
| LC-U2-13 Real-engine profile verifier | CoolProp oracle plus blocking TESPy pure/blend/mixture design/offdesign smokes | pytest and installed engines; tests only |
| LC-U2-14 Architecture and packaging verifier | Dependency directions, extras, versions, cold imports, profiles, and wheel behavior | pytest, uv, and build tools; tests only |
| LC-U2-15 Performance and quality verifier | Linear call counts, 10,000-point bounds, coverage, Ruff, and patch hygiene | pytest, coverage, `perf_counter`, and `tracemalloc`; tests only |

## Proposed Module and Resource Layout

| Path | Owner |
|---|---|
| `OpenPinch/analysis/heat_pumps/performance_maps/__init__.py` | Narrow engine-neutral Unit 2 entry points; no concrete adapter imports |
| `OpenPinch/analysis/heat_pumps/performance_maps/models.py` | LC-U2-01 frozen values |
| `OpenPinch/analysis/heat_pumps/performance_maps/context.py` | LC-U2-01 pure context builder |
| `OpenPinch/analysis/heat_pumps/performance_maps/fluids.py` | LC-U2-02 working-fluid resolver and TESPy token formatter |
| `OpenPinch/analysis/heat_pumps/performance_maps/points.py` | LC-U2-03 lazy canonical iterator and numerical normalization helpers |
| `OpenPinch/analysis/heat_pumps/performance_maps/protocols.py` | LC-U2-04 structural simulator protocol |
| `OpenPinch/analysis/heat_pumps/performance_maps/factory.py` | LC-U2-04 closed selection with local adapter imports |
| `OpenPinch/analysis/heat_pumps/performance_maps/generation.py` | LC-U2-05 atomic coordinator |
| `OpenPinch/analysis/heat_pumps/performance_maps/errors.py` | LC-U2-06 diagnostics and exceptions |
| `OpenPinch/analysis/heat_pumps/performance_maps/provenance.py` | LC-U2-07 deterministic provenance builder |
| `OpenPinch/analysis/heat_pumps/performance_maps/settings.py` | LC-U2-09 fixed convergence settings |
| `OpenPinch/analysis/heat_pumps/performance_maps/resources.py` | LC-U2-10 strict characteristic loader |
| `OpenPinch/analysis/heat_pumps/performance_maps/adapters/coolprop.py` | LC-U2-08 required default adapter |
| `OpenPinch/analysis/heat_pumps/performance_maps/adapters/tespy.py` | LC-U2-11 optional leaf and only Unit 2 TESPy import |
| `OpenPinch/data/heat_pumps/performance_maps/openpinch-single-stage-compressor-v1.json` | LC-U2-10 characteristic resource |

The package initializer exports only the internal service/value surface needed
by Unit 3. It does not become a root `OpenPinch` export and does not import
TESPy. Concrete adapter modules are not aggregate imports from
`adapters/__init__.py`.

## LC-U2-01: Internal Values and Context Builder

LC-U2-01 owns frozen slotted dataclasses for:

- target map basis and resolved working fluid;
- generation context and operating point;
- normalized point simulation and simulator metadata;
- convergence settings and characteristic data; and
- typed diagnostic coordinates/details.

The pure builder combines a valid Unit 1 request with a detached target basis,
calls LC-U2-02 once, derives capacity/mode/approach values, and returns a
complete context. It creates no engine session and mutates no input.

**NFR ownership**: NFR-U2-001, NFR-U2-006, NFR-U2-009 through NFR-U2-012,
NFR-U2-022, NFR-U2-026.

## LC-U2-02: Working-Fluid Resolver

LC-U2-02 adapts the existing CoolProp fluid parser into an immutable
engine-neutral value. It:

- separates property backend from fluid text, defaulting to HEOS;
- rejects case-normalized REFPROP before any property object is constructed;
- distinguishes pure, registered-blend, and explicit molar-mixture forms;
- preserves explicit component order and normalizes fractions;
- creates a short-lived CoolProp state and verifies required dew/bubble states;
- formats a TESPy `|molar` token without importing TESPy; and
- destroys the temporary property state after resolution.

No module-level cache or interaction-parameter mutation is permitted.

**NFR ownership**: NFR-U2-002, NFR-U2-007 through NFR-U2-010, NFR-U2-021.

## LC-U2-03: Canonical Point Iterator

LC-U2-03 lazily yields one point for every already-canonical Unit 1 request
coordinate. It owns useful-duty scaling, approach translation, ordinal-based
curve/point identity, internal absolute-temperature/lift checks, and exact
Cartesian traversal order.

It reports invalid coordinates through LC-U2-06 without calling an adapter and
continues to the next coordinate so aggregate diagnostics remain complete.

**NFR ownership**: NFR-U2-011, NFR-U2-016 through NFR-U2-018.

## LC-U2-04: Simulator Protocol and Factory

The protocol exposes only `prepare(context)`, `simulate(point)`, and `close()`.
The closed factory normalizes backend identity and uses a local import to create
the selected adapter. Unsupported names fail before preparation; explicit TESPy
never falls back to CoolProp.

The context manager guarantees exactly one close attempt. It exposes no raw
engine property, network, connection, temporary path, or logger.

**NFR ownership**: NFR-U2-004 through NFR-U2-006, NFR-U2-019,
NFR-U2-021, NFR-U2-022, NFR-U2-026.

## LC-U2-05: Generation Coordinator

LC-U2-05 is the sole production coordinator. It:

1. constructs one simulator session;
2. prepares it at most once;
3. consumes LC-U2-03 lazily;
4. interprets LC-U2-06 point-local/fatal failures;
5. validates signs, balance, useful duty, and COP independently;
6. retains points only while a successful map remains possible;
7. closes the session once;
8. raises one aggregate error when any diagnostic exists; or
9. builds provenance and constructs one Unit 1 map after total success.

The coordinator knows no concrete engine type and performs no retry, backend
fallback, logging, filesystem write, or target mutation.

**NFR ownership**: NFR-U2-011 through NFR-U2-021, NFR-U2-022,
NFR-U2-024, NFR-U2-026.

## LC-U2-06: Diagnostics and Internal Failures

LC-U2-06 defines closed diagnostic codes, sanitized diagnostic values, the
aggregate `HprMapGenerationError`, and a private adapter failure carrying only
stable fields plus `session_fatal`.

Known point-local failures permit the next point to restore design state.
Unknown or explicitly fatal failures prevent further engine calls and generate
ordered `session_unavailable` diagnostics. The Python cause chain retains local
debug evidence without entering public payload fields or automatic logs.

**NFR ownership**: NFR-U2-016, NFR-U2-019 through NFR-U2-021,
NFR-U2-027.

## LC-U2-07: Provenance Builder

LC-U2-07 consumes only validated scalar/tuple/mapping facts. It produces the
fixed recursive JSON shape for engine versions, mode/model/fluid identity,
composition/anchors, design condition, capacity, approaches, compressor-only
power, convergence settings, characteristic points/digest, and grid counts.

It omits time, host, path, object, and raw exception data. LC-U2-05 calls it
only after total success.

**NFR ownership**: NFR-U2-010, NFR-U2-011, NFR-U2-013, NFR-U2-027.

## LC-U2-08: CoolProp Adapter

LC-U2-08 retains one resolved property basis per session and delegates point
calculation to the current `VapourCompressionCycle`. It supplies requested sink
duty for heat-pump mode or source duty for refrigeration mode and normalizes
cycle watts to kilowatts exactly once.

Its focused oracle compares the adapter with a direct existing-cycle call under
CoolProp 8 or newer for pure, registered-blend, and explicit-mixture cases.

**NFR ownership**: NFR-U2-002, NFR-U2-007, NFR-U2-010,
NFR-U2-012 through NFR-U2-015, NFR-U2-024.

## LC-U2-09: TESPy Convergence Settings

LC-U2-09 owns `openpinch-tespy-hpr-convergence-v1` as one frozen value. The
design and offdesign calls use the selected values from NFRP-U2-012 with no
request override. It provides a deterministic plain mapping for provenance and
tests.

It does not mutate TESPy's private residual constants. Post-solve physical
acceptance remains LC-U2-05's responsibility.

**NFR ownership**: NFR-U2-004, NFR-U2-010 through NFR-U2-012,
NFR-U2-021, NFR-U2-026.

## LC-U2-10: Characteristic Resource Loader

LC-U2-10 reads the one compressor resource through `importlib.resources`. It
requires exact top-level keys, schema/version/quantity identifiers, strictly
ascending finite abscissae, finite positive factors, and enough distinct points
for interpolation. It returns a frozen value and SHA-256 digest of canonical
resource bytes.

Tests pin the exact content, digest, resource size, and installed-wheel access.
No runtime regeneration or write path exists.

**NFR ownership**: NFR-U2-010, NFR-U2-026 through NFR-U2-028.

## LC-U2-11: TESPy Adapter

LC-U2-11 is the only Unit 2 module importing TESPy. It:

- converts the resolved fluid to the accepted TESPy token;
- builds the fixed refrigerant-only network;
- loads LC-U2-09 and LC-U2-10 once;
- solves and validates one target-derived design condition;
- saves one design snapshot in a unique temporary directory;
- restores the snapshot before each requested offdesign point;
- maps public TESPy convergence/output state into normalized results or typed
  failures; and
- removes all session resources on close.

It has no alternate CoolProp-result path. CoolProp remains TESPy's selected
property provider internally, but LC-U2-11 does not invoke the LC-U2-08 cycle.

**NFR ownership**: NFR-U2-003 through NFR-U2-005, NFR-U2-007 through
NFR-U2-010, NFR-U2-019 through NFR-U2-024, NFR-U2-026.

## LC-U2-12: Fake Simulator and PBT Model

This test-only component provides configurable success, point-local failure,
fatal failure, invalid output, preparation failure, and cleanup failure. Its
event log is the observable system paired with a Hypothesis state-machine model
over fresh, prepared, fatal, and closed states.

Reusable strategies generate contexts, fluids, grids, results, and failure
ordinal sets. They preserve domain validity, include boundary values, shrink
normally, and never call an engine.

**NFR ownership**: NFR-U2-011 through NFR-U2-021, NFR-U2-024,
NFR-U2-025; PBT-02, PBT-03, PBT-05 through PBT-10.

## LC-U2-13: Real-Engine Profile Verifier

LC-U2-13 contains two focused layers:

- required CoolProp oracle/glide tests running in the base profile; and
- blocking TESPy pure-fluid, registered-blend, explicit binary/ternary mixture,
  design/offdesign, restoration, and cleanup smokes running only in the TESPy
  profile.

Real-engine comparisons use named tolerances and no elapsed-time assertions.
The optional profile cannot skip because TESPy is missing.

**NFR ownership**: NFR-U2-002 through NFR-U2-005, NFR-U2-007 through
NFR-U2-015, NFR-U2-019, NFR-U2-023, NFR-U2-024, NFR-U2-028.

## LC-U2-14: Architecture and Packaging Verifier

LC-U2-14 proves:

- only LC-U2-11 imports TESPy;
- Unit 2 has no external optimizer, consumer, application, or presentation
  dependency;
- OpenPinch and default map modules import without TESPy;
- metadata contains `CoolProp>=8`, neutral/compatible TESPy extras, and the
  tested TESPy minimum;
- the lock and CI profile commands match metadata; and
- base and optional built wheels behave as declared.

**NFR ownership**: NFR-U2-001 through NFR-U2-006, NFR-U2-023,
NFR-U2-026 through NFR-U2-028.

## LC-U2-15: Performance and Quality Verifier

LC-U2-15 checks point call counts and measured increasing-grid behavior, then
runs the 10,000-point fake generation under the time/memory guard. It also owns
new-module coverage, Ruff, formatting, resource drift, build validation, and
patch hygiene integration.

**NFR ownership**: NFR-U2-016 through NFR-U2-018, NFR-U2-025,
NFR-U2-028.

## Dependency Direction

| Consumer | May depend on | Must not depend on |
|---|---|---|
| LC-U2-01 | Standard library, Unit 1 request types | Engines, Unit 3, application objects |
| LC-U2-02 | LC-U2-01, CoolProp, existing fluid helper | TESPy, OpenUtility, mutable global property configuration |
| LC-U2-03 | LC-U2-01 and LC-U2-06 | Engines, adapters, materialized full grid |
| LC-U2-04 | LC-U2-01 and LC-U2-06; local concrete imports in factory | Eager TESPy or consumer imports |
| LC-U2-05 | Unit 1 and LC-U2-01 through LC-U2-07 | Concrete adapter types, Unit 3, consumers |
| LC-U2-06 | Standard library and LC-U2-01 | Engines, logging configuration, application layer |
| LC-U2-07 | LC-U2-01, LC-U2-09, LC-U2-10 | Engines, host/runtime environment data |
| LC-U2-08 | Protocol values and existing CoolProp cycle | TESPy, Unit 3, consumers |
| LC-U2-09 | Standard library | TESPy private globals or request tuning |
| LC-U2-10 | Standard resource/JSON/hash libraries | TESPy defaults, runtime writes, checkout paths |
| LC-U2-11 | Protocol values, LC-U2-09, LC-U2-10, optional TESPy | Unit 3, consumers, LC-U2-08 fallback |
| LC-U2-12 through LC-U2-15 | Production boundaries and test/build dependencies | Production imports from tests |

The production graph is acyclic. The only optional engine edge terminates in
LC-U2-11. Test and build components depend inward on production behavior; no
production component imports a verifier.

## Runtime Data Flow

1. Unit 3 will supply a detached target basis and validated Unit 1 request.
2. LC-U2-01 and LC-U2-02 create one complete generation context.
3. LC-U2-04 creates the explicitly selected fresh session.
4. LC-U2-05 prepares the session and consumes LC-U2-03 lazily.
5. LC-U2-08 or LC-U2-11 returns normalized point simulations or LC-U2-06
   failures.
6. LC-U2-05 performs independent physical checks and closes once.
7. Total success invokes LC-U2-07 and Unit 1 map construction; any failure
   raises one LC-U2-06 aggregate error.

No intermediate engine state, partial map, or temporary path crosses the Unit 2
boundary.

## Logical-Component Verification Matrix

| Component | Focused evidence | Broader evidence |
|---|---|---|
| LC-U2-01 | Frozen values, context derivation, input immutability | Unit 1 integration |
| LC-U2-02 | Fluid parsing, REFPROP rejection, mixture/glide properties | CoolProp 8 regression suite |
| LC-U2-03 | Size/order/identity/induction properties | 10,000-point guard |
| LC-U2-04 | Selection, cold import, exact lifecycle events | Base/optional wheel smokes |
| LC-U2-05 | Success, corruption, aggregate failure examples | Full map contract validation |
| LC-U2-06 | Sanitization, ordering, local/fatal classification | Failure state machine |
| LC-U2-07 | Exact repeatable provenance | JSON map round trip |
| LC-U2-08 | Direct existing-cycle oracle | Existing thermodynamic regressions |
| LC-U2-09 | Exact settings snapshot | TESPy design/offdesign smoke |
| LC-U2-10 | Strict JSON and digest snapshot | Installed resource access |
| LC-U2-11 | Real pure/blend/mixture and cleanup smokes | Blocking TESPy profile |
| LC-U2-12 | Strategy health and state-machine invariants | Fixed-seed PBT gate |
| LC-U2-13 | Real-engine focused tests | Dependency update evidence |
| LC-U2-14 | Import/metadata/profile assertions | Isolated built distributions |
| LC-U2-15 | Linear time/memory and coverage | Complete quality gate |

## Infrastructure Disposition

Infrastructure Design remains N/A. LC-U2-01 through LC-U2-11 are synchronous
in-process library/resource components, and LC-U2-12 through LC-U2-15 are
test/build components. There is no cloud resource, server, container, database,
queue, cache service, secret, monitor, or deployment topology.
