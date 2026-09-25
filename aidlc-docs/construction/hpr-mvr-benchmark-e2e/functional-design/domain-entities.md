# Domain Entities

These are test-design entities. They do not add production domain types or
public APIs.

## Benchmark problem

A benchmark problem is one absolute path returned by the shared sorted discovery
of `examples/stream_data/p_*.json`.

Attributes:

- `path`: unique stable fixture path;
- `case_id`: filename used in pytest parameter IDs and failure messages;
- `ordinal`: zero-based position in the sorted corpus;
- `project_name`: filename stem without the `p_` prefix.

Invariants:

- paths are unique, regular JSON files, and deterministically sorted;
- the benchmark corpus equals the existing e2e direct-target corpus;
- the current corpus contains 54 problems.

## HPR profile

An HPR profile is an immutable declarative test configuration.

Attributes:

- `profile_id`: stable descriptive identifier;
- `service_name`: public target-accessor method;
- `is_utility`: direct or utility placement flag;
- `topology`: cascade, parallel, or optimized VC+MVR;
- `configuration`: explicit fluid, port, stage, load and search controls.

The six ordered profiles are direct/cascade heat pump, utility/parallel heat
pump, direct/cascade refrigeration, utility/parallel refrigeration,
direct/optimized MVR, and utility/optimized MVR.

## Benchmark assignment

An assignment is the pair `(benchmark_problem, hpr_profile)` produced by
`profiles[problem.ordinal % 6]`.

Invariants:

- every discovered problem appears exactly once;
- assignment order matches sorted problem order;
- every profile has nine assignments for the current 54-case corpus;
- adding a problem preserves deterministic assignment without changing a
  filename list.

## Benchmark outcome

An outcome has one of three mutually exclusive states:

1. `solved`: public service returns a committed HPR target;
2. `no_op`: public service returns no target because no positive applicable load
   exists;
3. `typed_failure`: public service raises `HPRTargetingError` with diagnostics.

An unexpected exception is not an outcome. It is a defect requiring root-cause
analysis.

## Convergence witness

A convergence witness is an ordered, observation-only trace of distinct
uncached search evaluations for a strict-success sentinel. Each trace item
contains the design point, success state and finite objective when viable.

Derived values are the first viable objective, ordered incumbent-best sequence,
best observed viable objective, selected final objective and relative/absolute
improvement. It is test evidence only and is not added to the public result API.

## Problem-state snapshot

A snapshot is the public JSON representation of `problem.results` immediately
before the HPR call, plus the existing target count. It is the oracle for
transaction atomicity on no-op and failure and for exactly-one-target commit on
success.

## Direct process-MVR scenario

The direct scenario consists of packaged `process_mvr.json`, source stream
`Evaporator vapour`, explicit compression settings, the returned component,
replacement streams, and a downstream direct-integration target. It is separate
from the 54-case HPR assignment because those inputs lack pressure-qualified gas
streams.

## Testable properties

- **Invariant**: assignment preserves corpus size and uniqueness.
- **Invariant**: every assignment uses one of the six declared profiles.
- **Oracle**: the HPR corpus is exactly the existing e2e corpus owner output.
- **Invariant**: no-op and typed failure preserve the public state snapshot.
- **Invariant**: success commits exactly one target and yields finite positive
  thermodynamic duties.
- **Easy verification**: every profile sentinel has at least two viable points,
  a strict incumbent improvement, and selects the best observed feasible point.
- **Round-trip**: detached success records and typed diagnostics remain
  JSON-serializable and copyable through their public contracts.
- **Stateful/model property**: the transition model has only unchanged state for
  no-op/failure and one appended target for success.

PBT-01 is satisfied by the identified categories. Generated testing is useful
for assignment and transition invariants only if it exercises more than the
complete fixed corpus; implementation must avoid tautological generated tests.
