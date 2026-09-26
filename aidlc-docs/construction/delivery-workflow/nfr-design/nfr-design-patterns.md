# Delivery NFR design patterns

## One deadline, one retry owner

The package-index verification coordinator owns all retry scheduling. The
transport performs one attempt and returns a classified observation; it does
not nest another independent retry loop.

- Postflight default: 300 seconds; supported explicit budgets: 1-600 seconds.
- Poll interval: 10 seconds, capped by remaining time. No sleep after a
  complete or permanent-failure observation.
- Per-attempt socket timeout: at most 20 seconds and no more than remaining
  budget. Use a monotonic clock; wall-clock adjustments cannot extend retries.
- Check the deadline before requests and waits and again after a response.
  A result returned after expiry is reported as late, not timely success.
- Retry HTTP 408, 425, 429, and 5xx, and classified connection/timeouts.
  Reject authentication, TLS certificate validation, malformed payloads,
  unexpected file sets, duplicate filenames, and hash conflicts immediately.
- HTTP 404 is a typed absent observation. Preflight may accept absence but
  must not convert an exhausted transport failure into absence. Give
  preflight a separate short 60-second total budget for transient I/O.
- Honor a valid Retry-After delay within remaining time; if it exceeds the
  budget, expire rather than retry sooner. Invalid values use the normal poll
  delay. Finalize exact header/date handling during implementation tests.

The monotonic deadline bounds scheduling; it is not a guarantee that a blocked
DNS call or slow response stream is interrupted by a socket timeout. Retain an
outer workflow job timeout exceeding the largest supported polling budget and
setup allowance. Document this distinction, cap response size, and test the
logical deadline with injected clock, sleeper, and transport. No added retry
thread or daemon is needed. Runner limits belong to Infrastructure Design.

## Exact evidence, conservative fallback

Normalize observations into a lane-keyed mapping only after rejecting
duplicate identities. Evaluate against one required-lane policy. Accept each
lane through either successful execution or explicit compatible reuse proof.
Do not infer success from skipped execution or aggregate workflow colour.

Evidence records bind repository, event provenance, source/merge tree, policy,
lock/toolchain context, run and attempt. Select the latest applicable attempt;
an older success cannot hide a newer failure. Policy changes invalidate
incompatible proof. Pagination failure or resource limits invalidate reuse
and cause fresh validation, while a final gate without proof fails.

Use finite API/subprocess timeouts and bounded pagination. Stop with a named
reason when a bound is reached, never treat truncated evidence as complete.
Do not cache a mutable branch name as validated identity.

## Idempotent recovery with conflict detection

Progress is derived from the immutable manifest plus verified external facts.
Stored checkpoints accelerate lookup but never override conflicting current
state. Repeated identical operations are no-ops; differing bytes or identities
are terminal conflicts. Serialize mutation for a release identity, disable
automatic cancellation of active publication, and recheck after lock entry.

Before first upload, retain the original bundle and manifest with verified
source provenance. Request 30-day CI retention and attach matching evidence
to draft assets for recovery beyond CI expiry. Hash equality alone does not
establish provenance: verify the manifest's origin and source binding too.
If a trusted original manifest/bundle cannot be recovered, stop. Do not
regenerate it from the current branch or invent a replacement source run.

Publication states remain separate from overall completion. If PyPI succeeds
but GitHub finalization fails, diagnostics state that the package is already
public and only finalization remains. No destructive rollback is attempted.

## Scoped authority and validation boundaries

Read-only candidate validation is separate from trusted release preparation
and mutation. Validate external identifiers before use, pass values through
argument arrays or safely quoted environment variables, and never evaluate
PR titles/body text as shell source. Reject credential-bearing index URLs.
Only trusted configured destinations are used by publishing workflows.

Publishing jobs consume verified distributions, not an arbitrary artifact
name supplied without provenance. Grant OIDC/write authority only there.
Do not run untrusted PR code under elevated release permissions. Workflow
contract tests cover permissions, triggers, artifact selection, and job needs.
Infrastructure Design maps these rules to actual jobs and environments.

## Efficient validation and useful output

Shared lane definitions own selectors, required matrix cells, and compatible
reuse profiles. Description-only edits do not dispatch numerical tests;
candidate changes do. Existing full-quality numerical tests are preserved.
No test retries or smaller physics budgets are introduced by this refactor.

Produce JUnit and duration outputs where pytest runs, plus a small delivery
summary for source identity, evidence, states, and remediation. Attempt report
retention on failed runs without masking the original failure; report missing
outputs explicitly when setup or cancellation prevents their creation.
Redact secret-bearing fields before emitting diagnostics.

## Verification and traceability

| Pattern | Requirements | Verification |
|---|---|---|
| Single deadline and typed retry | N01-N02 | Fake clock, late success, exhausted transient I/O, Retry-After, permanent conflict. |
| Complete compatible evidence | N03, N07-N10 | Reference predicate, generated job maps, pagination failure, stale attempts. |
| Serialized exact-state recovery | N04-N05, N12, N17 | Generated transitions, repeated resumes, missing bytes, conflicting manifest. |
| Scoped authority | N11-N14 | Event/permission contracts, hostile input examples, provenance rejection. |
| Shared lanes and reporting | N06-N09, N15-N18 | Selection equivalence, coverage gate, failure output tests, comparable timing reports. |

All patterns are specifications, not claims of implemented safeguards.
