# Business logic model

## Scope and sequence

Normal review and successful review tests precede a generated bump PR into
develop. Maintainer merges that PR, then merges the existing develop-to-main PR.
Main validates the bumped candidate with evidence-backed reuse and builds a new
release bundle. The standalone publisher verifies and publishes that bundle.
No automatic approval, merge, protected-branch write or application change.

## State transitions

1. WAITING_REVIEW: only ready same-repository develop-to-main PRs are eligible.
   Require current reviewDecision APPROVED and successful required validation
   lanes for the reviewed candidate. Draft, changes requested, missing reviews,
   incomplete checks or ambiguous candidate are waiting/blocked, never publish.
2. READY: independently verify heads, merge tree, lane evidence and versions.
   Allocate once and generate only allowed version edits plus preparation record.
3. PREPARING: create a deterministic automation-owned branch and bump PR into
   develop. An existing exact branch/PR is reused; conflicting content blocks.
   A branch created before an interrupted PR-create call is recoverable.
4. AWAITING_BUMP_MERGE: retain the original PR. Generated PR satisfies develop's
   protections; no automated approval or merge. Closure without merge blocks
   and requires maintainer direction, rather than repeatedly recreating it.
5. PREPARED: bump merged into develop, transformation/record verified. Original
   main PR reruns its lightweight proof/package checks and obtains any required
   renewed approval. Its complete merge gate now permits merging.
6. MAIN_VALIDATION: verify prepared identity against merged source. Reuse only
   proven unaffected lanes, execute all other required lanes and build fresh
   artifacts. Missing test proof causes full execution, not publication denial
   by itself. Invalid preparation/provenance is a separate blocking error.
7. PUBLISH: exact main bundle follows existing tag/draft, TestPyPI verification,
   PyPI verification and public-release finalization. Complete state is a verified
   no-op. Interrupted state resumes the original identity/artifact, not a new bump.

## Avoiding a preparation deadlock

Separate review-validation evidence from the merge-readiness gate. Before bump,
tests can all pass while the complete main-PR gate correctly says preparation
is missing. Coordinator evaluates concrete lane successes, not the workflow's
aggregate success, which may reflect that intentional gate failure. It must
still reject every failed, cancelled, missing or unsupported required test lane.
Develop-targeted bump PRs do not themselves require another preparation record
to be generated, preventing recursive preparation.

## Event and authority boundary

Reconcile when review or validation completion changes; either ordering must
work. A read-only review-event notification workflow can wake a default-branch
coordinator through workflow completion, alongside completion of PR validation.
Treat event PR/run identifiers only as lookup hints. Fetch current eligible PRs,
reviews, heads and validation results before acting. Manual reconciliation is
a bounded recovery entry point, not a bypass of these checks.

All write-capable logic is loaded from trusted main, never the PR checkout or
downloaded executable artifacts. Fetch candidate blobs as inert data; do not
install/build/source them with write credentials. Scope writes to preparation
job contents/pull-requests permissions; validation remains read-only and publisher
retains scoped OIDC. Workflow trigger details and job-level contracts must be
verified during Code Generation planning. No credential/settings changes here.

GitHub references consulted for review notifications and current review state:

- https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows
- https://docs.github.com/en/graphql/reference/pulls

## Concurrency and stale work

Serialize preparation repository-wide, without cancelling in-progress writes.
Before branch creation and PR creation, compare observed heads with the reviewed
snapshot. Branch creation must fail if it already exists; never force-push.
A race after the final check may create a stale PR but cannot authorize its
merge/release: bump validation and main gate repeat the head/diff checks.

If source or main base changes while preparation is pending, block the stale
preparation and require fresh review/validation. Do not silently refresh a
reviewed bump PR. A maintainer closes/replaces stale preparation explicitly.
If additional changes arrive after bump merge but before main merge, retain
the allocated unpublished version; require fresh review and full validation
when reuse is no longer valid. Never allocate a second patch for the same
pending release merely because its test proof became stale.

## Activation

Initial rollout must install the new coordinator, proof policy and merge gate
together through the existing reviewed workflow. New policy enforces preparation
for subsequent releases, not retroactively for the commit installing it.
Document the single bootstrap boundary explicitly; do not introduce a permanent
user-controlled skip flag. Required-check configuration and bot PR-run approvals
must be verified before activation. No live activation is part of local coding.
