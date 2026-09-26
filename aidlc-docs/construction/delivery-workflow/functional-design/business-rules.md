# Delivery rules

## Validation and merge rules

1. Required lane membership comes from a shared versioned policy, including
   expanded matrix cells. Workflow success by itself is insufficient evidence.
2. Failed, cancelled, missing, duplicated, or unexpectedly skipped mandatory
   results fail the aggregate decision. A skipped execution is acceptable only
   when verified reuse explicitly supplies that lane's proof.
3. Separate functional test success from measured runtime. Do not retry a
   failing numerical test until green or weaken feasibility/convergence bounds.
4. Ordinary coverage remains 95 percent. All existing benchmark assignments
   and specialized lane responsibilities remain represented.
5. Version preparation never runs concurrently as a branch-writing part of
   candidate validation. Only explicit release validation requires a new
   releasable version; ordinary integration does not.
6. Gate activation requires actual repository protection configuration. The
   migration must preserve existing protection until the replacement check is
   deployed and verified. No design document constitutes remote activation.

## Index and artifact rules

| Observation | Preflight action | Postflight action |
|---|---|---|
| Absent | Upload permitted after manifest verification | Poll until complete or deadline. |
| Matching subset | Missing distributions may be uploaded | Poll until complete or deadline. |
| Exact complete set | No upload needed | Success. |
| Unexpected filename, duplicate, or wrong hash | Stop | Stop immediately. |
| Transient transport error | Bounded retry; never infer absence | Bounded retry within the same overall deadline. |
| Permanent access/protocol error | Stop with diagnostic | Stop with diagnostic. |

Each operation has a finite overall time budget. Nested retries cannot reset
that budget. Network time counts toward it; request timeout and waiting are
bounded by the remaining allowance. No new request starts after expiry.
Diagnostics identify destination, version, elapsed time, attempt, state, and
missing filenames where known; they never reveal credentials. Retry numeric
defaults and HTTP classification details are deferred to NFR Design.

Every upload validates bundle provenance and exact contents first. An upload
tool's success, including an existing-file skip, does not replace postflight
verification. Unexpected remote state must not be overwritten or deleted.

## Testable Properties and PBT assessment

| Rule | Planned verification | Stage assessment |
|---|---|---|
| PBT-01 | Properties identified across all three design artifacts. | Compliant. |
| PBT-02 | Round-trip any newly introduced manifest serializer/parser; otherwise document absence of an inverse pair. | Implementation-stage check pending. |
| PBT-03 | Generate evidence and file maps; assert exact identity and fail-closed invariants. | Implementation-stage check pending. |
| PBT-04 | Repeated recovery preserves identity and completed external state. | Implementation-stage check pending. |
| PBT-05 | Gate reference predicate and release-state reference model. | Implementation-stage check pending. |
| PBT-06 | Generate interruption/retry/observation sequences using an in-memory model. | Implementation-stage check pending. |
| PBT-07 | Reusable strategies for realistic SHAs, versions, cells, manifests, and observations. | Implementation-stage check pending. |
| PBT-08 | Existing Hypothesis seed convention, shrinking, fake clocks; no live publishing. | Implementation-stage check pending. |
| PBT-09 | Retain existing pytest/Hypothesis stack. | NFR-stage check pending. |
| PBT-10 | Permanent example regressions complement generated properties. | Implementation-stage check pending. |

Only PBT-01 applies as a blocking check to Functional Design; the remaining
rules are N/A for stage completion, with explicit downstream obligations above.
Security and Resiliency extensions remain disabled. No frontend artifact is
needed because this unit has no frontend.

## Requirement traceability

D01-D02: event model and immutable preparation. D03-D05: lane policy, candidate
proof, aggregation, and migration. D06: bounded index observations. D07-D08:
manifest-bound release/recovery sequence. D09: diagnostic and reporting rules.
All requirements are represented; infrastructure permission mapping remains
an intentionally separate design stage.
