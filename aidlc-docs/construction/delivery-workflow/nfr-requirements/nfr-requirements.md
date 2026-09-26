# Delivery non-functional requirements

## Reliability and bounded execution

- N01: Default package-index postflight visibility budget is proposed as 300
  seconds per destination, configurable within a documented finite range.
  Network attempts and waits share one monotonic budget; nested retries must
  not restart it. The outer job timeout provides a separate final bound for
  stalled underlying I/O. NFR Design must distinguish the logical polling
  deadline from any limitations of the network library's socket timeout.
- N02: Absence, matching partial visibility, and classified transient I/O may
  retry. Unexpected files, duplicate filenames, wrong hashes, authentication
  failures, and malformed responses must stop immediately. No upload result
  may substitute for exact postflight verification.
- N03: Gate decisions fail closed for missing, failed, cancelled, duplicate,
  or unexplained skipped required evidence. Evidence lookup failure may cause
  full validation, never a synthetic pass.
- N04: Concurrent release requests for the same project/version must not
  perform competing mutations. A newer request must not automatically cancel
  a publishing operation. Recovery must revalidate external state and retain
  original source and artifact identities.
- N05: A release interrupted after uploading must resume without rebuilding,
  overwriting, retagging, or silently declaring completion. Lost original
  bytes are a reported blocker, not grounds to generate replacement bytes.

Verification: fake-clock and transport tests, generated evidence/state tests,
and explicit regressions for every transition boundary. No external uptime
SLA is asserted for GitHub, TestPyPI, or PyPI.

## Performance and capacity

- N06: Preserve all existing benchmark cases, specialized test lanes,
  numerical quality assertions, and the 95-percent ordinary branch coverage
  gate. There is no blanket retry-to-green policy for tests.
- N07: Description-only PR edits must not start expensive test lanes. A source
  or evaluated merge-tree change must invalidate incompatible evidence.
- N08: Use a single lane policy and reusable validation owners. Retain job
  durations and compare before/after hosted runs of equivalent profiles,
  separating queue/setup time from test time and reuse from fresh execution.
  Do not claim a speedup from incomparable runs or a single local measurement.
- N09: New delivery helper tests must make no live index calls, publish no
  packages, and perform no actual retry sleeps. Keep their default generated
  domains bounded and retain Hypothesis shrinking.
- N10: Support the existing repository/matrix scale; no new cloud service,
  persistent worker, or autoscaling layer is required. API pagination and
  finite request budgets must handle run/job lists without silently truncating
  mandatory evidence. Resource exhaustion fails closed or falls back to tests.

## Security and provenance

- N11: PR validation has no branch-writing or publishing credentials. Write
  permissions and OIDC publishing capability are granted only to the stages
  that require them. Untrusted event fields are data, not shell commands.
- N12: Bind release requests to authorized main commits and a validated
  committed version. Bind retained artifacts to repository, source, build
  run/attempt, immutable artifact identity, and exact distribution hashes.
- N13: Preserve current review requirements, tag immutability, pinned action
  references, and trusted-publishing boundaries. Document any workflow-name
  or environment changes affecting publisher configuration before activation.
- N14: Logs must exclude tokens, credential-bearing URLs, and secret headers.
  Recovery must not delete or overwrite remote state to bypass a conflict.

## Observability and usability

- N15: Each gate summary identifies the evaluated SHA/tree, profile, policy,
  executed/reused proof, and reasons for rejection. Test reports preserve
  failing node IDs and per-test durations when execution reaches pytest.
- N16: Verification logs show destination, version, attempt, elapsed time,
  last observed state, and missing files where available. Final errors
  distinguish propagation timeout, transport exhaustion, malformed response,
  authorization failure, and content conflict.
- N17: Request 30-day retention for test reports and release recovery bundles,
  subject to repository limits. Durable matching draft-release assets may
  provide verified recovery beyond artifact expiry. No retention mechanism
  guarantees recovery if assets are deleted; missing evidence must be explicit.
- N18: Document version preparation, ordinary merging, explicit publication,
  interrupted-release recovery, and protection migration as separate tasks.
  Report partially published states honestly. Do not label locally tested YAML
  as a verified hosted rollout or an active required-check configuration.

## Acceptance and traceability

| Requirements | Functional coverage | Evidence |
|---|---|---|
| N01-N05 | D04-D08 | Fake-clock tests, generated state/evidence tests, interruption regressions. |
| N06-N10 | D03-D05, D09 | Lane partition, coverage, workflow event contracts, duration reports. |
| N11-N14 | D01-D02, D04, D07-D08 | Permission/provenance contracts and staged configuration review. |
| N15-N18 | D04-D09 | Diagnostic assertions, report artifacts, recovery and rollout instructions. |

These are proposed requirements, not completed implementation or measured
performance results. No further authority to publish, merge, or change remote
settings is implied by their approval.
