# Unit 3 NFR Design Patterns

## Scope and Approved Direction

These patterns implement the 33 approved Unit 3 NFR requirements. The chosen
design keeps execution sequential within each current targeting or map call.
Evaluator inputs and outputs are isolated and immutable so a separately approved
future optimizer may distribute independent candidate evaluations across
processes. Unit 3 adds no scheduler, process pool, thread pool, shared evaluator,
shared cache, or global lock.

The Security and Resiliency extensions remain disabled. The patterns below are
ordinary scientific-library integrity, failure, performance, and maintainability
controls required by the approved Functional Design and NFR Requirements.

## Pattern P01: Closed Selector at the Application Boundary

The target accessor normalizes `simulation_backend` exactly once before any
targeting work. A closed normalizer produces `coolprop` or `tespy`; invalid type,
empty value, or unknown value raises immediately.

The normalized backend travels as explicit runtime intent through selected
period and wrapper replay. It is never inferred from a cycle name, optimizer,
installed dependency, model object, or map request.

**Qualities**: additive API compatibility, deterministic dispatch, clear errors,
and no silent fallback.

## Pattern P02: Default-Path Branch Isolation

The branch for an omitted selector and explicit `coolprop` converges before the
existing CoolProp objective path. No general evaluator wrapper is inserted into
the default inner loop unless it is proven to delegate once without changing
arguments, calls, results, or model ownership.

TESPy preflight, import, cache allocation, record-specific extra work, and
optional failure handling remain outside the default branch. The winning
CoolProp record is constructed from the already available winning result rather
than through a second cycle solve.

**Qualities**: numerical compatibility, zero added thermodynamic calls, cold
TESPy isolation, and easy rollback.

## Pattern P03: Fail-Fast Compatibility Gate

TESPy selection runs a pure preflight before optimizer initialization. It
validates the single-stage topology, one evaporator, one condenser, scalar
period context, supported mode, no integrated expander, working-fluid property
capability, and optional dependency.

The gate returns one frozen prepared specification used by the evaluator factory
and winning-record builder. It does not create a TESPy network or mutate target
configuration.

**Qualities**: avoids expensive doomed searches, separates public errors from
candidate infeasibility, and makes supported scope inspectable.

## Pattern P04: Strategy and Closed Factory

An engine-neutral targeting thermodynamic evaluator protocol accepts one frozen
candidate request and returns one frozen normalized result. A closed factory
selects the default or optional strategy from the normalized backend.

The TESPy strategy is loaded lazily from its concrete leaf. Deterministic fakes
implement the same protocol without subclassing or engine dependencies. Factory
selection is exhaustive and has no dynamic plugin discovery.

**Qualities**: optional-dependency isolation, test substitution, controlled
variation, and explicit ownership.

## Pattern P05: Candidate-as-Design Unit of Work

Each unique TESPy optimizer candidate is one nominal equipment design unit of
work. Evaluation applies every physical input, runs a design solve, validates
and detaches the result, then restores a clean ready state before another key is
evaluated.

Candidate design state is not reused as offdesign state for a different
candidate. Only a topology shell and immutable characteristic resource may be
reused within the call. The later map generator separately uses the winning
candidate as its global design and evaluates requested map points offdesign.

**Qualities**: optimizer-order independence, physically coherent selection, and
clear targeting-versus-map semantics.

## Pattern P06: Exact Bounded Call-Local Memoization

One TESPy targeting call owns one 512-entry least-recently-used mapping. The key
is the complete frozen normalized physical candidate request. Lookup uses exact
equality only.

Immutable accepted results and candidate-local failures may be stored. Fatal
failures are never stored. A hit moves the entry to the most-recent position; a
miss evaluates once; insertion beyond 512 evicts the oldest entry. Cache
eviction changes performance only, and the cache is discarded during call
cleanup.

No key includes callback ordinal, memory identity, temporary path, or diagnostic
formatting. No physical field is omitted. No float rounding, tolerance bucket,
disk storage, decorator-global cache, or cross-call reuse is permitted.

**Qualities**: bounded memory, elimination of exact duplicate design solves,
determinism, and no stale dependency/model state.

## Pattern P07: Layered Independent Validation

Validation occurs at distinct boundaries:

1. application arguments and selector;
2. TESPy target compatibility and working fluid;
3. candidate request domain;
4. concrete engine convergence and extraction;
5. engine-neutral duties, power, COP, profiles, and energy closure;
6. winning simulation record;
7. target-to-map compatibility and basis construction; and
8. existing Unit 1 completed-map validation.

Each layer owns named tolerances appropriate to its dimensions. An inner success
flag cannot bypass an outer invariant check.

**Qualities**: defense in depth, dimensional correctness, and localized errors.

## Pattern P08: Recoverable/Fatal Failure Partition

Candidate-local property-state failure, non-convergence, or invalid normalized
physics produces one immutable infeasible result. After verified restoration,
later candidates continue. No retry occurs.

Invalid public arguments, unsupported topology, missing dependency, evaluator
construction, restoration failure, session corruption, and cleanup failure are
fatal to the complete target call. A fatal state prevents new engine calls and
is surfaced through the appropriate focused exception.

Neither partition permits CoolProp fallback, stale-value reuse, fabricated
success, or hidden partial map return.

**Qualities**: useful optimizer fault tolerance without masking model failure.

## Pattern P09: Explicit Session State and Finally Cleanup

The evaluator coordinator follows these internal states:

| State | Allowed action | Next state |
|---|---|---|
| Created | open optional evaluator and resources | Ready or Fatal |
| Ready | cache lookup or evaluate candidate | Ready, Evaluating, or Fatal |
| Evaluating | extract, validate, and restore | Ready or Fatal |
| Fatal | no new evaluation | Closing |
| Ready at completion | finish optimizer translation | Closing |
| Closing | close session, clear cache, release temporary state | Closed or CleanupFailed |
| Closed | none | Closed |
| CleanupFailed | none; propagate failure | CleanupFailed |

Cleanup runs once in a `finally`-equivalent path. Cache clearing, temporary-state
release, and evaluator close are observable through deterministic fakes.

**Qualities**: leak prevention, state-machine testability, and atomic teardown.

## Pattern P10: Immutable Anti-Corruption Boundary

Concrete CoolProp/TESPy outputs are translated immediately to engine-neutral
frozen requests, results, profile points, diagnostics, and metadata. No concrete
engine object crosses upward.

The normalized result adapter enforces nonnegative magnitude conventions,
kilowatts, degrees Celsius, ordered thermal profiles, mode-dependent useful duty
and COP, and compressor-only electricity before existing HPR accounting consumes
the value.

**Qualities**: prevents engine leakage, stabilizes accounting inputs, and allows
future implementation replacement without public schema change.

## Pattern P11: Detached Winning-Record Snapshot

After the winning candidate is selected, a strict frozen
`HprTargetSimulationRecord` snapshots all map-relevant nominal facts and
structured provenance. CoolProp creates it from the existing winner without a
new solve; TESPy creates it from the detached winning evaluator result.

The record is extra-field-forbidden, JSON-compatible, and contains no mutable
configuration, engine model, stream collection, optimizer, cache, temporary
path, or stack trace. The target exposes only the normalized backend directly;
the full snapshot stays in normalized HPR details.

**Qualities**: reproducibility, safe replay, immutable provenance, and no stale
configuration reads.

## Pattern P12: Pure Target-to-Basis Adapter

A pure builder validates the target and winning snapshot, then copies matching
facts into Unit 2's existing `HprTargetMapBasis`. It deep-detaches structured
JSON provenance and returns a frozen basis.

The adapter never reads current problem configuration, private engine models,
or stream implementation details. Equal targets produce equal bases, and
repeated extraction does not mutate input.

**Qualities**: deterministic handoff, idempotent observation, and strict unit
dependency direction.

## Pattern P13: Atomic Explicit Map Facade

The application map method performs compatibility validation, pure basis
construction, and exactly one Unit 2 generation call. Backend selection is
target-owned and cannot be overridden by the request.

The target, request, problem, configuration, cache, workspace, and existing
target caches are snapshotted as non-mutating inputs. Any failure returns no map.
Success returns one detached Unit 1 contract.

**Qualities**: visible expensive work, no hidden grid solve, and all-or-nothing
consumer output.

## Pattern P14: Deterministic Replay Adapter

Selected-period, independent all-period, and workspace batch wrappers forward
the normalized backend through their established ordered execution. Each scalar
result owns its own evaluator session, cache, and winning snapshot.

The distinct shared-vector multi-period TESPy route fails at preflight. The
default CoolProp route retains existing multi-period behavior. The explicit map
facade is not automatically mirrored over aggregates or batches.

**Qualities**: wrapper consistency, case/period isolation, canonical ordering,
and bounded first-release scope.

## Pattern P15: Process-Ready Isolation Without a Scheduler

Current target and map calls are sequential. Candidate requests, results,
failures, and stable ordinals contain only picklable/plain immutable values where
practical. Factory construction and cache ownership occur inside one call rather
than at module import.

A future approved optimizer may deduplicate requests in a parent process,
evaluate independent keys in worker processes with one evaluator and local cache
per worker, and merge by stable ordinal. Unit 3 does not implement or test that
scheduler and makes no thread-safe shared-engine promise.

**Qualities**: future process parallelism without present concurrency risk or
premature infrastructure.

## Pattern P16: Bounded Sanitized Diagnostics

Internal failures translate to stable categories and bounded JSON-compatible
details. Backend, model, mode, fluid identity, and physical candidate coordinates
are retained when useful. Stack traces, engine object representations, memory
addresses, temporary paths, unbounded nested values, and environment-sensitive
text are removed.

Missing TESPy errors use repository-standard extra installation guidance.

**Qualities**: actionable scientific diagnostics, artifact-safe output, and
ordinary information-hygiene safeguards.

## Pattern P17: Structural Performance Gates

Fast deterministic fakes assert linear callback handling, exact unique-key solve
bounds, 512-entry eviction, less than 64 MiB traced Python cache overhead, empty
post-call state, and no additional default CoolProp solve.

One marked supported-profile public TESPy target-and-map smoke uses bounded
candidate search and a minimal map, must finish within 300 seconds, and records
target/map/total elapsed trends. The CI job provides an outer timeout.

**Qualities**: portable regression detection without a false universal latency
claim.

## Pattern P18: Layered Test Substitution

Deterministic protocol fakes own combinatorial selector, cache, lifecycle,
failure, replay, record, basis, and non-mutation coverage. Existing CoolProp
behavior is the default oracle. Small real TESPy examples prove pure, registered
blend, explicit mixture, design solve, target translation, and map offdesign
integration.

Hypothesis strategies remain domain-constrained, shrinkable, and seed
reproducible. Critical paths also have explicit examples. Real TESPy is not used
inside broad generated tests.

**Qualities**: high coverage, reproducibility, fast feedback, and meaningful
engine integration.

## Pattern P19: Dual-Profile Artifact Gate

The base source/wheel profile blocks TESPy and proves default targeting, cold
imports, records, basis extraction, fixture consumption, and docs. The TESPy
source/wheel profile installs the neutral extra and runs real targeting/map
smokes. Both are blocking for pull requests and releases.

Archive checks verify optional metadata and exact schema, fixture, and
characteristic resources. Sphinx runs warning-strict; API/root/CLI inventories
prevent accidental surface growth.

**Qualities**: source/artifact parity, optional-install confidence, dependency
firewall enforcement, and publication quality.

## PBT Design Integration

| PBT rule | Design treatment |
|---|---|
| PBT-01 | Functional properties map to P01, P06 through P15, and P18 |
| PBT-02 | Winning-record and completed-map serialization round trips use constrained records/maps |
| PBT-03 | Selector, physics, cache bounds, ordering, replay, composition, and non-mutation invariants use generated domains |
| PBT-04 | Backend normalization and pure repeated basis extraction have idempotence properties |
| PBT-05 | Omitted selector and explicit CoolProp compare with the existing path as oracle |
| PBT-06 | Evaluator/cache states and mixed local/fatal/close sequences use a stateful fake model |
| PBT-07 | Reusable strategies cover candidates, profiles, records, targets, failures, mixtures, wrappers, and maps |
| PBT-08 | Hypothesis shrinking remains enabled and CI uses seed `20260715` or logs replay seeds |
| PBT-09 | Existing Hypothesis with pytest remains the documented framework |
| PBT-10 | Default/explicit backends, pure/blend/mixture, failure classes, eviction, and public smokes have explicit examples |

No PBT rule has an unresolved or blocking design finding.

## NFR Traceability

| Pattern | NFR requirements |
|---|---|
| P01 | NFR-U3-002, NFR-U3-007, NFR-U3-025 |
| P02 | NFR-U3-003, NFR-U3-004, NFR-U3-021 |
| P03 | NFR-U3-004, NFR-U3-011, NFR-U3-020, NFR-U3-024 |
| P04 | NFR-U3-004 through NFR-U3-006, NFR-U3-028, NFR-U3-033 |
| P05 | NFR-U3-008, NFR-U3-011 through NFR-U3-013, NFR-U3-019 |
| P06 | NFR-U3-008, NFR-U3-013 through NFR-U3-015, NFR-U3-017 |
| P07 | NFR-U3-007 through NFR-U3-012, NFR-U3-022 through NFR-U3-024 |
| P08 | NFR-U3-019 through NFR-U3-021, NFR-U3-024 |
| P09 | NFR-U3-017, NFR-U3-019, NFR-U3-020 |
| P10 | NFR-U3-006, NFR-U3-009 through NFR-U3-012, NFR-U3-024, NFR-U3-033 |
| P11 | NFR-U3-003, NFR-U3-008, NFR-U3-012, NFR-U3-022, NFR-U3-033 |
| P12 | NFR-U3-012, NFR-U3-022, NFR-U3-023, NFR-U3-033 |
| P13 | NFR-U3-002, NFR-U3-006, NFR-U3-023, NFR-U3-027, NFR-U3-033 |
| P14 | NFR-U3-002, NFR-U3-025 |
| P15 | NFR-U3-017, NFR-U3-018 |
| P16 | NFR-U3-004, NFR-U3-020, NFR-U3-021, NFR-U3-024 |
| P17 | NFR-U3-003, NFR-U3-013 through NFR-U3-017, NFR-U3-030 |
| P18 | NFR-U3-003, NFR-U3-007 through NFR-U3-012, NFR-U3-019 through NFR-U3-029 |
| P19 | NFR-U3-001 through NFR-U3-006, NFR-U3-026, NFR-U3-027, NFR-U3-030 through NFR-U3-032 |

All NFR-U3-001 through NFR-U3-033 requirements map to at least one pattern.

## Extension Compliance

- **Property-Based Testing**: Compliant. PBT-01 through PBT-10 have explicit
  pattern ownership and no blocking gap.
- **Security Baseline**: Disabled. Authentication, authorization, sandboxing,
  and service threat patterns are N/A; P03, P07, P10, P16, and P19 provide the
  ordinary integrity controls required by the NFRs.
- **Resiliency Baseline**: Disabled. Distributed failover and recovery patterns
  are N/A; P08 and P09 provide the local evaluator failure/cleanup behavior
  required by the NFRs.
