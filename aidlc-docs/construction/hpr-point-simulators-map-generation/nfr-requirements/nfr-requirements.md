# Unit 2 HPR Point Simulators and Map Generation NFR Requirements

## Scope

These requirements govern Unit 2's internal generation context, deterministic
grid service, default CoolProp adapter, optional TESPy adapter, structured
diagnostics, and simulator resources. Unit 1 remains the owner of serialized map
validation. Unit 3 remains the owner of public targeting and map-generation
methods.

## Approved NFR Decisions

| Concern | Decision |
|---|---|
| TESPy installation | Add a neutral `tespy` extra, retain `brayton_cycle` as a compatible alias, and retain TESPy in `full` |
| TESPy version | Require `tespy>=0.10.1.post2` with no upper bound and enforce compatibility in CI |
| Real-engine CI | Use a blocking TESPy profile plus a separate base profile without TESPy |
| Reproducibility | Exact structure and provenance; tolerance-based thermodynamic floats; no quantization |
| CoolProp version | Raise the package-wide baseline to `CoolProp>=8` and rerun all thermodynamic regressions |
| Property backends | Default and test HEOS; reject REFPROP; permit other CoolProp-native backends after capability checks |
| Concurrency | Fresh isolated sessions per call; no external-engine thread-safety promise; processes for parallelism |
| Grid policy | No production point cap or per-map timeout; linear orchestration and bounded fake-engine benchmark |

## Runtime and Dependency Compatibility

### NFR-U2-001: Supported runtime

Unit 2 must run on the repository's supported Python `>=3.14.2` baseline. It
must use current package conventions and must not introduce a second runtime or
service process.

**Verification**: focused and complete tests run under the repository runtime;
the built wheel imports in an isolated environment.

### NFR-U2-002: CoolProp 8 baseline

The required base dependency must be declared as `CoolProp>=8`. The dependency
lock must resolve CoolProp 8 or newer, and the complete existing thermodynamic
regression suite must pass before Unit 2 can ship.

**Verification**: metadata and lock assertions, runtime version assertion, all
existing heat-pump/fluid tests, and the complete applicable test gate.

### NFR-U2-003: Neutral TESPy extra

Package metadata must expose a neutral `tespy` optional extra containing
`tespy>=0.10.1.post2`. The existing `brayton_cycle` extra must remain a
compatible alias with the same requirement, `full` must retain the requirement,
and the development group must support the blocking TESPy test profile.

**Verification**: parsed metadata assertions, lock consistency, installation
smokes for `.[tespy]`, `.[brayton_cycle]`, and `.[full]`, and no duplicate
incompatible requirement ranges.

### NFR-U2-004: Forward TESPy compatibility

TESPy has a tested minimum of `0.10.1.post2` and no declared upper bound. A
newer resolvable TESPy release is acceptable only while the blocking adapter
compatibility tests pass. Unit 2 must use documented public TESPy APIs rather
than private attributes to minimize unnecessary version coupling.

**Verification**: minimum-version metadata check, adapter static review, locked
real-engine job, and dependency-update CI evidence when the lock advances.

### NFR-U2-005: Cold-import isolation

TESPy must be imported only inside the concrete TESPy adapter after explicit
selection. Importing OpenPinch, Unit 1 contracts, Unit 2 orchestration, or the
CoolProp adapter must succeed when TESPy is absent or import-blocked. CoolProp
remains a required base dependency.

**Verification**: subprocess cold-import tests with TESPy blocked and an
isolated base installation without the `tespy` extra.

### NFR-U2-006: Package boundary isolation

Unit 2 must not import OpenUtility, Pyomo, HiGHS, application accessors,
presentation modules, or Unit 3. Engine objects and mutable target objects must
not cross its plain internal value or Unit 1 output boundaries.

**Verification**: architecture dependency tests, import scans, and type/value
tests for returned maps and diagnostics.

## Working-Fluid and Property Compatibility

### NFR-U2-007: Common working-fluid input domain

Both adapters must accept the same input categories: pure fluids, registered
blends, and explicit mole-fraction mixtures with any component count. Neither
adapter may impose a refrigerant, mixture-pair, or component-count allowlist.
Support is established by constructing the selected property wrapper and
evaluating the states required by the cycle.

**Verification**: shared conformance tests plus pure-fluid, registered
zeotropic-blend, explicit binary-mixture, and explicit ternary-mixture examples.

### NFR-U2-008: CoolProp-native backend policy

HEOS is the default and required CI property backend. An explicitly named
CoolProp-native backend may be used when it can evaluate the required states.
REFPROP must be rejected deterministically before engine preparation regardless
of local installation or licensing. Backend names must be normalized without
silently replacing the requested backend.

**Verification**: HEOS examples, REFPROP rejection before simulator creation,
one injectable native-backend capability example, and preservation of the
normalized backend in provenance and errors.

### NFR-U2-009: Mixture-state safety

Explicit fractions must normalize deterministically without changing component
order. Unit 2 must not install estimated mixing rules, mutate CoolProp global
interaction parameters, or expand an opaque registered blend into a different
property model. Missing interaction data or unavailable required states must
produce a typed working-fluid or preparation diagnostic.

**Verification**: input immutability and normalization properties, malformed
and unsupported-mixture examples, and global configuration before/after checks.

### NFR-U2-010: Saturation and glide reproducibility

Zeotropic evaporation must use the saturated-vapour/dew anchor and condensation
the saturated-liquid/bubble anchor in both adapters. The source specification,
property backend, mixture kind, explicit composition when applicable, anchor
convention, CoolProp version, and selected simulator version must appear in
deterministic provenance.

**Verification**: direct property-state comparisons for registered and explicit
mixtures, provenance assertions, and the existing glide-profile regressions.

## Numerical Determinism and Physical Integrity

### NFR-U2-011: Structural determinism

Equal normalized inputs and equivalent deterministic engine results must
produce exactly equal point order, identifiers, curve grouping, diagnostic
order, metadata keys, and provenance values. Timestamps, random identifiers,
memory addresses, temporary paths, and locale-dependent formatting are
forbidden.

**Verification**: repeat generation with fresh fake and real sessions, exact
structural equality, and cross-process locale-independent examples.

### NFR-U2-012: Tolerance-based thermodynamic comparison

Engine duties, compressor power, COP, and property states must be compared with
explicit quantity-appropriate absolute and relative tolerances. Engine
convergence tolerances, Unit 1 energy-balance tolerance, temperature-match
tolerance, and regression-test tolerances must remain distinct and must never be
substituted for one another.

**Verification**: tolerance-boundary examples, named tolerance fields, and
tests just inside and outside every acceptance boundary.

### NFR-U2-013: No output quantization

Valid finite engine outputs must not be rounded or quantized to manufacture
cross-platform byte identity. Serialized structure remains deterministic, but
floating-point values are scientifically compared within declared tolerances.

**Verification**: high-precision fake values survive map construction and JSON
round-trip; platform compatibility checks use tolerances rather than exact
thermodynamic bytes.

### NFR-U2-014: CoolProp compatibility oracle

For supported nominal conditions, the default adapter must reproduce the
existing `VapourCompressionCycle` duties and compressor power within declared
numerical tolerances. At fixed temperatures, generated duty and power scale
with load while COP remains invariant within tolerance.

**Verification**: generated-domain oracle properties plus explicit heat-pump,
refrigeration, registered-blend, and explicit-mixture regression examples.

### NFR-U2-015: Independent physical validation

The grid service must validate finite signs, mode-specific useful duty, energy
closure, and COP independently of either adapter. An adapter success flag alone
must never authorize an invalid Unit 1 point.

**Verification**: corrupt fake-adapter examples for every physical invariant
and property tests over accepted complete maps.

## Performance and Scalability

### NFR-U2-016: Linear orchestration

For `n` Cartesian operating points, expansion, orchestration, point assembly,
and successful-map working memory must be `O(n)`. Failure diagnostics must also
be `O(n)` with at most one primary diagnostic per requested point plus bounded
preparation and cleanup diagnostics. Unit 2 must not perform pairwise grid
comparisons or retain engine snapshots per point.

**Verification**: code-path review, call-count properties, and measured scaling
at increasing fake-engine grid sizes.

### NFR-U2-017: Bounded fake-engine benchmark

A deterministic 10,000-point fake-engine generation, including Unit 1 map
validation, must complete within 5.0 seconds and use less than 256 MiB of
additional peak traced memory on the primary CI runner. The threshold is a
coarse superlinear-regression guard, not a production latency promise.

**Verification**: `perf_counter` and `tracemalloc` regression test with the
repository's fixed seed and documented CI environment.

### NFR-U2-018: Engine-dominated runtime policy

Production generation has no arbitrary maximum-point rejection and no per-map
wall-clock timeout. Real CoolProp and TESPy tests assert behavior, convergence,
and lifecycle rather than machine-sensitive elapsed time. CI may retain its
ordinary finite job timeout to prevent a stalled external engine from blocking
the pipeline indefinitely.

**Verification**: no point-cap or timeout branch in Unit 2, large fake-grid
acceptance, and real-engine tests without duration assertions.

## Reliability, Cleanup, and Concurrency

### NFR-U2-019: Lifecycle completeness

Every generation attempt must create a fresh simulator, prepare at most once,
close exactly once, and release temporary design state on success, preparation
failure, point failure, cancellation by exception, or cleanup failure. A closed
session must never be reused.

**Verification**: state-machine properties over a fake session, temporary-path
existence checks, and real TESPy success/failure cleanup examples.

### NFR-U2-020: Atomic failure and bounded diagnostics

Any preparation, point, normalization, or cleanup failure must prevent map
return. Aggregate diagnostics remain canonically ordered, JSON-compatible,
bounded per message/detail value, and free of stack traces, engine objects,
temporary paths, and partial maps.

**Verification**: multi-failure examples, generated failed-coordinate sets,
diagnostic size/type assertions, and no-partial-result checks.

### NFR-U2-021: Explicit dependency and backend failure

Missing TESPy, unsupported property state, non-convergence, and rejected REFPROP
must identify the selected backend and corrective action. None may trigger
automatic CoolProp fallback, estimated property behavior, retry, or a successful
partial map.

**Verification**: import-blocking, REFPROP, wrapper-construction,
non-convergence, and point-exception tests with exact error categories.

### NFR-U2-022: Isolated concurrency boundary

Each call must own unshared adapter state, and each TESPy call must own a unique
temporary directory. Calls must avoid OpenPinch process-global mutable state.
Unit 2 does not guarantee thread safety for external thermodynamic engines and
must document isolated processes as the supported parallel-generation
mechanism. It must not add a process-global serialization lock.

**Verification**: distinct-session/path tests, static mutable-state review, and
documentation assertions. Threaded throughput is not a release gate.

## Testability, Maintainability, and Usability

### NFR-U2-023: Blocking real-TESPy profile

CI must include a blocking optional profile that installs `.[tespy]` and runs a
minimal real TESPy pure-fluid, registered-blend, explicit-mixture, design, and
offdesign sequence. A separate isolated base profile must prove OpenPinch and
CoolProp behavior without TESPy installed.

**Verification**: workflow assertions and execution evidence from both profiles;
the TESPy profile must not skip because its declared dependency is absent.

### NFR-U2-024: Layered deterministic testing

Fast fake-adapter tests own exhaustive traversal, lifecycle, aggregation, and
invalid-output behavior. CoolProp owns the existing-cycle oracle. Real TESPy
tests remain small and target adapter integration. A real engine must not be
used where a deterministic fake can prove orchestration behavior.

**Verification**: test ownership review, marker/profile separation, and focused
suite execution.

### NFR-U2-025: Property-based framework and reproducibility

Hypothesis integrated with pytest is the Unit 2 property-based framework. Tests
must use reusable constrained strategies, retain shrinking, and run in CI with
the repository seed `20260715` or emit an exact replay seed. The internal
simulator lifecycle requires a stateful model because it is a state machine;
PBT-discovered counterexamples become permanent examples.

**Verification**: dependency metadata, strategy organization, state-machine and
invariant test discovery, seed-bearing CI commands, and no disabled shrinking.

### NFR-U2-026: Adapter maintainability

The orchestration core must depend only on a typed `HprPointSimulator` protocol
and frozen internal values. CoolProp and TESPy imports, unit/sign conversion,
engine convergence inspection, and property syntax translation remain inside
their respective concrete adapters. The packaged compressor characteristic
must have a stable identifier and drift-detectable content digest.

**Verification**: architecture tests, protocol fake substitution, module import
review, and characteristic snapshot/digest tests.

### NFR-U2-027: Documentation and diagnostic usability

Developer and public-facing documentation owned by Unit 3 must be supportable
from Unit 2 facts: extras, version floors, default/explicit backend behavior,
CoolProp-native mixture syntax, REFPROP rejection, dew/bubble semantics,
compressor-only power, fixed approaches, no hidden fallback, process-based
parallel guidance, and typed failure categories. Errors must be actionable
without exposing engine internals.

**Verification**: warning-strict documentation build in the integrated unit,
diagnostic examples, and metadata/docs consistency checks.

### NFR-U2-028: Quality and regression gate

Unit 2 changes must pass Ruff lint and formatting, focused example and
Hypothesis tests, at least 95 percent statement and branch coverage for new Unit
2 production modules, the complete existing thermodynamic regression suite,
both dependency profiles, package build/validation, installed-wheel smoke, and
`git diff --check`.

**Verification**: Code Generation and integrated Build and Test evidence.

## Extension Compliance

Property-Based Testing is enabled:

- PBT-09 is satisfied by the existing Hypothesis/pytest stack and direct
  development dependency.
- PBT-01 properties from Functional Design are measurable through NFR-U2-009
  through NFR-U2-017 and NFR-U2-019 through NFR-U2-025.
- PBT-02, PBT-03, PBT-05, PBT-07, PBT-08, and PBT-10 remain applicable.
- PBT-06 is applicable because the internal simulator session has a mutable
  prepare/simulate/close lifecycle, even though it is not a reusable public
  object.
- PBT-04 is N/A because no operation claims idempotent mutation.
- No PBT finding is blocking.

Security Baseline and Resiliency Baseline are disabled in project state and are
not enforced. Authentication, authorization, service availability, disaster
recovery, persistent storage, and frontend accessibility are N/A. Ordinary
dependency isolation, bounded diagnostics, cleanup, and failure atomicity remain
required by the Unit 2 design.
