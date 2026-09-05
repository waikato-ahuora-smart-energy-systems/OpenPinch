# Unit 3 HPR Target API Integration and Publication NFR Requirements

## Scope

These requirements govern the public CoolProp-default/TESPy-selectable HPR
targeting methods, targeting-time thermodynamic evaluator, detached winning
target record, explicit target-to-map bridge, replay wrappers, documentation,
and integrated release evidence.

Unit 1 remains the strict public map-contract owner. Unit 2 remains the
target-derived point/map simulator owner. Unit 3 adds full TESPy participation
in ordinary single-stage targeting, as selected by the user, without weakening
the optional dependency boundary or the existing CoolProp result contract.

## Approved NFR Decisions

| Concern | Decision |
|---|---|
| Duplicate TESPy candidates | Bounded call-local exact-key cache; immutable successes and candidate-local failures only |
| Cache isolation | No approximate matching, cross-call reuse, process-global state, or persistence |
| Real TESPy performance | Structural solve bounds plus one marked public target-and-map smoke within 300 seconds on the supported CI profile |
| Timing interpretation | Record elapsed trend evidence; do not impose a tight workstation-specific latency promise |
| Default compatibility | Omitted selector and explicit CoolProp are blocking numerical and structural oracles |
| TESPy availability | Existing neutral optional extra and lazy leaf import; no fallback |
| CI | Separate blocking base and TESPy source/artifact profiles inherited from Unit 2 |

## Runtime, API, and Dependency Compatibility

### NFR-U3-001: Supported runtime

Unit 3 must run on the repository's supported Python `>=3.14.2` baseline and use
the existing in-process library architecture. It must add no service process,
network protocol, database, worker daemon, or independently deployed component.

**Verification**: focused and complete suites run under the supported runtime;
source and wheel installations execute public smokes.

### NFR-U3-002: Additive public API

The only new public application surfaces are the keyword-only
`simulation_backend="coolprop"` argument on the two current vapour-compression
target methods and the explicit keyword-only `hpr_performance_map` target-accessor
method. Existing return classes, positional behavior, package-root exports, CLI,
and unrelated target signatures remain unchanged.

**Verification**: signature snapshots, public API inventory tests, root export
tests, and absence checks for a CLI or second workflow.

### NFR-U3-003: Default CoolProp compatibility

For identical deterministic inputs, an omitted selector and explicit
`simulation_backend="coolprop"` must produce the same target subtype, success
state, candidate ordering, numerical HPR fields, stream data, and public summary
within the pre-existing HPR tolerances. The default path must perform exactly
the same number of CoolProp thermodynamic evaluations as before Unit 3 and must
not import TESPy.

**Verification**: generated-domain oracle properties, pinned representative
heat-pump/refrigeration regressions, call-count spies, and a subprocess import
guard.

### NFR-U3-004: Optional TESPy isolation

TESPy must be resolved only after the normalized selector is `tespy` and only
inside the concrete optional leaf. OpenPinch root, contracts, domain targets,
application accessors, default HPR targeting, target-basis extraction, map
orchestration, and documentation inventory must import successfully when TESPy
is absent or blocked.

**Verification**: static architecture scans, cold-import subprocess tests, a
base wheel installation without TESPy, and a TESPy wheel installation with the
extra.

### NFR-U3-005: Dependency versions and extras

Unit 3 inherits `CoolProp>=8` as the required property dependency and
`tespy>=0.10.1.post2` without an upper bound in the neutral `tespy`, compatible
`brayton_cycle`, and `full` extras. The lock must retain a tested TESPy release;
the current baseline is TESPy 0.11.2. Unit 3 may not add another simulator,
optimization, cache, serialization, or benchmarking dependency.

**Verification**: metadata and lock assertions, profile installation smokes,
and dependency-diff review.

### NFR-U3-006: Forbidden dependency direction

OpenPinch production and test-support code for this feature must not import
OpenUtility, Pyomo, HiGHS, application modules from analysis, presentation from
analysis, or concrete TESPy classes outside the optional leaf. OpenUtility
interoperability is verified only with plain schema/fixture data.

**Verification**: architecture tests and import scans over source and built
artifacts.

## Determinism and Numerical Integrity

### NFR-U3-007: Selector determinism

Backend normalization must be idempotent and locale-independent. Equal valid
inputs yield exactly the same normalized backend; invalid inputs fail before
thermodynamic or optimizer work.

**Verification**: constrained string properties and explicit invalid-type,
empty, whitespace, case, and unknown-value examples.

### NFR-U3-008: Candidate evaluation determinism

A TESPy targeting candidate is keyed by its complete normalized physical
request. Equal keys within one call yield structurally equal immutable results.
Evaluation order, prior failures, cache eviction of unrelated entries, memory
address, temporary path, locale, and wall-clock time must not change an accepted
candidate result beyond named thermodynamic tolerances.

**Verification**: fake-evaluator permutation properties, repeated real-candidate
examples, exact metadata comparison, and tolerance-based numeric comparison.

### NFR-U3-009: Native floating-point preservation

Target and map bridge values must preserve valid finite engine outputs without
rounding or decimal quantization. Deterministic structure and identifiers are
exact; thermodynamic values use named absolute and relative tolerances.

**Verification**: high-precision fake values, JSON round trips, and tests just
inside and outside each physical tolerance.

### NFR-U3-010: Separate tolerance dimensions

Targeting engine convergence, candidate energy closure, existing HPR
feasibility, map energy balance, map temperature matching, and regression
comparison tolerances remain separately named and dimensionally appropriate.
None may be reused merely because another tolerance has the same numerical
value.

**Verification**: explicit value owners, boundary examples, and static review
of tolerance references.

### NFR-U3-011: Independent candidate validation

The engine-neutral targeting layer must validate convergence, finite signs,
mode-specific useful duty, energy closure, positive compressor power, COP, and
thermal-profile duty independently of the concrete evaluator. A TESPy success
flag alone cannot authorize a candidate.

**Verification**: corrupt fake-evaluator results for every invariant plus
generated valid-result properties.

### NFR-U3-012: Mixture fidelity

Pure fluids, registered blends, and explicit N-component molar mixtures must
preserve backend, component order, normalized composition, registered identity,
and dew/bubble anchor convention through selector dispatch, candidate request,
winning target record, map basis, errors, and provenance. No estimated mixing
rule or process-global property mutation is permitted.

**Verification**: pure, registered zeotrope, binary, ternary, and generated
N-component examples/properties; REFPROP and unsupported-state examples.

## Performance and Scalability

### NFR-U3-013: Linear targeting-adapter overhead

For `c` optimizer objective callbacks and `u` unique normalized candidate
requests, Unit 3 orchestration outside the external engine must be `O(c)` time.
TESPy design solves must be no greater than `u` while a key remains resident and
never greater than `c`. CoolProp performs one existing cycle evaluation per
callback with no Unit 3 duplicate evaluation.

**Verification**: increasing fake callback sequences, exact unique-key and solve
counts, and default CoolProp call-count regressions.

### NFR-U3-014: Bounded exact candidate cache

One TESPy targeting call owns an exact-key least-recently-used cache of at most
512 immutable entries. The key contains every physical request field and uses
exact normalized values; tolerance-based or partial-key matches are prohibited.
Only successful results and candidate-local failures may be cached. Fatal
dependency, lifecycle, cleanup, or configuration failures are never cached.

Eviction affects performance only: a later evicted duplicate may be solved
again and must yield an equivalent result. Cache state is discarded when the
targeting call ends.

**Verification**: generated duplicate/permutation/overflow sequences, a
513-unique-key eviction example, fatal-failure non-caching, and post-call
collection checks.

### NFR-U3-015: Cache memory bound

Cache orchestration and immutable normalized results must add less than 64 MiB
of peak traced Python memory for 512 maximum-size fake candidate records on the
primary CI runner. TESPy-native allocations are excluded from this Python heap
threshold and are covered by lifecycle/repeated-call evidence.

**Verification**: deterministic `tracemalloc` regression using constrained
maximum-profile fake results.

### NFR-U3-016: Real TESPy end-to-end smoke budget

One marked real-TESPy public workflow must target a supported single-stage HPR
case through a deterministic bounded candidate-search fixture, extract the
winning target basis, and generate a minimal one-temperature-pair part-load map.
It must complete within 300 seconds on the supported TESPy CI profile.

The test records target, map, and total elapsed times as trend evidence. The
300-second assertion is a stalled/regression guard, not a production latency
promise. Developer machines outside the supported profile are not required to
meet a tighter threshold.

**Verification**: marked source and installed-wheel jobs with a finite job
timeout, elapsed evidence, and no dependency-absence skip.

### NFR-U3-017: Repeated-call resource stability

Sequential TESPy targeting calls must start with empty caches and fresh evaluator
state. After warm-up, ten deterministic fake calls and at least three guarded
real evaluator lifecycle calls must show no monotonic growth in live sessions,
temporary directories, cached entries, or retained engine objects.

**Verification**: fake counters and weak references plus a small guarded real
TESPy lifecycle regression. Threaded throughput is not a release gate.

### NFR-U3-018: Concurrency boundary

Each public targeting or map call owns isolated evaluator/cache/session state.
OpenPinch makes no thread-safety guarantee for external thermodynamic engines and
documents isolated processes as the supported parallel mechanism. Unit 3 adds
no global lock, shared cache, thread pool, process pool, or background worker.

**Verification**: state-identity tests, static mutable-state review, and
documentation assertions.

## Reliability and Failure Isolation

### NFR-U3-019: Lifecycle completeness

Every TESPy targeting attempt creates at most one evaluator session, evaluates
zero or more candidates, and closes exactly once on success, optimizer failure,
candidate failure, translation failure, interruption by exception, or cleanup
failure. Closed evaluator state is never reused.

**Verification**: stateful fake-evaluator model, generated command sequences,
and real success/failure cleanup examples.

### NFR-U3-020: Candidate-local versus fatal failures

Non-convergence and invalid physical states local to one candidate become
infeasible candidate results and permit clean later evaluations. Invalid public
arguments, unsupported topology, absent dependency, evaluator construction,
state restoration, and cleanup failures are fatal to the targeting request.

**Verification**: mixed generated failure sequences and explicit examples for
every category.

### NFR-U3-021: No fallback or fabricated success

A selected TESPy path must never call CoolProp cycle evaluation after TESPy
selection, relabel a CoolProp value as TESPy, reuse a stale prior result, or
manufacture duties for a failed candidate. If no accepted candidate remains,
the existing targeting failure contract is used with backend-specific context.

**Verification**: blocked CoolProp-cycle spies on the TESPy path, all-failure
candidate searches, and provenance equality checks.

### NFR-U3-022: Detached target provenance

Every successful supported target contains a frozen, extra-field-forbidden,
JSON-compatible simulation record with no engine object, mutable collection,
temporary path, stack trace, or process-specific representation. Record JSON
round trips are exact and unknown fields fail validation.

**Verification**: Pydantic round-trip properties, recursive JSON checks,
mutation attempts, extra-field examples, and object-graph scans.

### NFR-U3-023: Atomic map bridge

Target compatibility and basis construction complete before simulator creation.
Any compatibility or Unit 2 generation failure returns no map and changes no
target, problem, configuration, cache, workspace, or request state.

**Verification**: snapshot-before/after properties and explicit failures at
each boundary.

### NFR-U3-024: Bounded actionable diagnostics

Errors identify the selected backend, stable model, mode, fluid identity,
candidate temperature/duty coordinates when applicable, failure category, and
installation action for missing TESPy. Messages and structured details are
bounded and exclude local paths, memory addresses, engine objects, and stack
traces.

**Verification**: generated long/nested failure causes, sanitization limits,
and exact public error-category examples.

## Replay, Usability, and Documentation

### NFR-U3-025: Replay preservation

Selected-period, independent all-period, and workspace batch targeting preserve
the normalized backend, existing option precedence, canonical period/case order,
failure isolation, and target subtype. Shared-vector multi-period TESPy
targeting fails clearly before execution; CoolProp behavior is unchanged.

**Verification**: generated wrapper propagation and order properties plus
explicit supported/rejected examples.

### NFR-U3-026: Public documentation completeness

Warning-strict documentation must cover signatures, full targeting meaning of
the selector, default equivalence, single-stage limits, scalar/all-period versus
shared-vector behavior, optional installation, pure/blend/explicit-mixture
syntax, REFPROP rejection, dew/bubble anchors, compressor-only power, candidate
design versus map offdesign semantics, explicit map generation, fixed capacity,
schema version, interpolation, failures, and OpenUtility independence.

**Verification**: documentation inventory assertions, example execution, link
checks available in the repository build, and Sphinx warnings as errors.

### NFR-U3-027: Plain-data consumer usability

Examples must demonstrate that a completed map can be converted to a plain
mapping or JSON and decoded against the packaged schema/fixtures without
importing OpenUtility. The example must not require Pyomo, HiGHS, or TESPy when
using a checked-in golden fixture.

**Verification**: isolated base-install example and independent mapping/schema
tests.

## Testability, Maintainability, and Release Quality

### NFR-U3-028: Injected evaluator boundary

Targeting orchestration must depend on an engine-neutral evaluator protocol and
closed factory. Fake evaluators own exhaustive dispatch, caching, lifecycle,
failure, record, and non-mutation tests. Real TESPy is reserved for small
integration and public end-to-end smokes.

**Verification**: architecture tests, test ownership review, and absence of
TESPy imports in broad property suites.

### NFR-U3-029: PBT framework and reproducibility

Hypothesis integrated with pytest remains the selected PBT-09 framework. Unit 3
properties use reusable domain strategies, automatic shrinking, and CI seed
`20260715` or emit an exact replay seed. Stateful testing models the internal
evaluator/cache lifecycle; critical paths also have explicit example tests.

**Verification**: strategy/test organization, dependency metadata, fixed-seed
CI commands, no disabled shrinking, and permanent examples for discovered
regressions.

### NFR-U3-030: Coverage and static quality

New Unit 3 production modules and materially changed targeting integration paths
must achieve at least 95 percent combined statement and branch coverage. All
changed Python passes Ruff lint and formatting; all changed text passes patch
hygiene and repository content checks.

**Verification**: path-scoped branch coverage report, Ruff, format check,
Markdown/content validation, and `git diff --check`.

### NFR-U3-031: Layered regression gate

The release gate includes focused examples and PBT, existing complete HPR and
thermodynamic regressions, default CoolProp numerical oracles, TESPy targeting
and map profiles, architecture/API/docs/package tests, source and wheel builds,
archive resource checks, and isolated installed-wheel smokes for both core and
TESPy extras.

**Verification**: recorded commands and counts in Code Generation and integrated
Build and Test summaries.

### NFR-U3-032: Source and artifact parity

Source checkout, sdist, and wheel must expose the same public signatures,
optional-extra metadata, target record schema, map schema/fixtures,
characteristic resource bytes, and default/explicit backend behavior. Installed
core artifacts prove TESPy absence; installed TESPy artifacts execute the marked
public smoke.

**Verification**: archive inspection, byte digests, isolated installations, and
artifact smoke assertions.

### NFR-U3-033: Maintainable ownership

Selector/replay logic remains in application accessors, evaluator orchestration
and target-basis construction remain in HPR analysis, plain winning records
remain in HPR contracts, target fields remain in domain targets, engine code
remains in concrete leaves, and map validation remains in Unit 1. No duplicated
map model or reverse dependency is permitted.

**Verification**: module ownership review and dependency architecture tests.

## Availability, Security, and Accessibility Assessment

Service availability, uptime, failover, disaster recovery, remote persistence,
authentication, authorization, personal-data protection, network encryption,
frontend usability, and accessibility are N/A because Unit 3 is an in-process
scientific library feature with no service, account, stored user data, or UI.

Ordinary library integrity remains mandatory through input validation, optional
dependency isolation, sanitized diagnostics, temporary-state cleanup, bounded
memory, immutable records, and artifact verification. The disabled Security and
Resiliency extensions add no further requirements.

## Extension Compliance

Property-Based Testing is enabled:

- **PBT-09**: Compliant. Hypothesis with pytest is already selected, installed,
  supports domain strategies and shrinking, and uses reproducible CI seeds.
- **PBT-01**: Functional Design properties are made measurable by NFR-U3-003,
  NFR-U3-007 through NFR-U3-017, and NFR-U3-019 through NFR-U3-029.
- **PBT-02, PBT-03, PBT-04, PBT-05, PBT-06, PBT-07, PBT-08, and PBT-10**:
  carried forward to Code Generation planning where applicable. PBT-04 applies
  to backend normalization and repeated pure basis extraction; PBT-06 applies
  to the internal evaluator/cache lifecycle.
- No PBT finding is blocking.

Security Baseline and Resiliency Baseline are disabled in workflow state and
are not enforced. Their stage concerns are N/A as described above.
