# Delivery logical components

These responsibilities stay within existing script, workflow, and packaging
test owners. They do not require one module per row or a new service.

| Component | Inputs | Outputs and constraints |
|---|---|---|
| Lane policy | Event/profile and policy revision | Required cells and selectors; single source for execution and proof. |
| Candidate resolver | Repository/event and immutable references | Evaluated commit/tree and trusted provenance; no implicit latest-head substitution. |
| Evidence reader | Candidate and bounded API access | Complete latest-attempt evidence or a typed unavailable reason. |
| Gate evaluator | Policy, executed results, accepted reuse evidence | Pure pass/fail decision with missing/conflicting lane reasons. |
| Manifest verifier | Source identity, trusted manifest, bundle bytes | Exact verified distributions or terminal error. |
| Index transport | Validated endpoint and remaining timeout | One classified response; no independent retries; bounded payload. |
| Release inspector | Response and expected file map | Absent, partial, complete, or permanent conflict; pure comparison. |
| Verification coordinator | Inspector/transport, monotonic clock, sleeper, budget | Bounded polling, safe progress output, typed final outcome. |
| Recovery planner | Verified manifest and external observations | Next permitted transition or conflict; no direct network mutation. |
| Workflow orchestration | Validated request and transition decision | Scoped authorized mutation; serialized publishing and exact artifact handoff. |
| Reporter | Sanitized evidence/results | Gate summary, release state, JUnit/duration artifacts and recovery hints. |

## Dependency and failure boundaries

Policy and candidate resolution precede evidence collection and gate
evaluation. A failed gate prevents release mutation. Manifest verification
precedes index inspection and upload. The recovery planner never repairs a
conflict by changing expected identities. The reporter observes results and
cannot convert a failure into success.

Keep parsing and decision functions pure where practical; isolate API, clock,
filesystem, and publishing effects at their boundaries. Fake boundaries in
tests rather than mocking numerical solvers or touching live package indexes.

## Property-based testing obligations

Carry forward the functional-design property catalogue: invariant and oracle
tests for the gate; idempotence and stateful model tests for recovery; exact
file-map comparison invariants; and round-trip properties for any new
manifest serialization pair. Use bounded realistic strategies and retain
shrinking and the existing reproducible seed convention.

PBT-01 remains satisfied by the inherited property catalogue. PBT-09 remains
satisfied by the selected pytest/Hypothesis stack. PBT-02 through PBT-08 and
PBT-10 have no implementation-stage verification due in NFR Design; their
tests remain required in Code Generation. No blocking design finding exists.
Security and Resiliency extensions remain disabled; explicit NFR security and
recovery constraints are nevertheless designed above.
