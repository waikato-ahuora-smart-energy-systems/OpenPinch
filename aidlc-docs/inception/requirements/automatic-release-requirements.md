# Automatic release requirements

## Intent and decision

Restore automatic release and PyPI publication after successful main
validation. The user selected option A: versions are prepared in reviewed
commits, never automatically bumped by CI. This is a moderate-complexity
delivery-policy change in the existing brownfield workflow, with irreversible
external publication risk. Requirements depth is standard.

This supersedes D01's manual-initiation requirement in
`delivery-workflow-requirements.md`. D02-D09 and the existing verification,
recovery, permission and quality constraints remain applicable.

## Acceptance requirements

1. A main push automatically starts publication only after the complete
   full-profile validation succeeds for the exact candidate commit.
   Failed, cancelled, incomplete or skipped validation cannot publish.
2. The version must already be committed consistently in project metadata,
   bump configuration and lockfile. CI does not modify source or versions.
3. Publish a new version using the exact distributions built and tested for
   that candidate. Do not rebuild between validation and publication.
4. A verified completed version produces a visible successful no-op.
   A tag alone is not proof of completion. Determine completion against the
   original release identity and bytes, not a new same-version build.
5. Existing draft, partial, prerelease, conflicting or unverifiable state
   stops automatic publication with an actionable recovery diagnostic.
   Never overwrite artifacts, move tags or silently adopt another build.
6. Retain manual verified resume. The legacy 0.6.10 recovery remains separate;
   this change neither dispatches it nor authorizes package replacement.
7. Preserve TestPyPI upload and exact-hash verification before PyPI, then
   verify both destinations before finalizing the stable GitHub release.
8. Preserve existing trusted publishers and environment protections. PR and
   develop runs never publish or acquire publication credentials.
9. Serialize automatic and manual release mutations; do not cancel active
   publication when another main push arrives. Retries must be safe.
10. Summaries distinguish published, already complete, blocked and failed.
    Document version preparation, automated initiation and recovery clearly.

## Verification and scope

Add behavioral and workflow-contract tests for successful/failed validation,
new/completed/partial/conflicting versions, provenance, duplicate events,
manual recovery and permission boundaries. Preserve all current validation
lanes, coverage thresholds, bounded polling and release safeguards.
Local tests must not publish or mutate external releases.

In scope: workflows, existing release helpers, delivery tests and docs.
Out of scope: automatic version bumps, automatic merges, numerical behavior,
branch-protection changes and live publication during local implementation.
Hosted activation and publisher compatibility require explicit verification;
local tests alone cannot prove hosted OIDC or scheduling behavior.

User stories remain skipped by prior user direction. Existing architecture
artifacts and current delivery source establish the affected boundaries.

## Extension compliance

PBT-01 through PBT-10: N/A to requirements-only output; retain the enabled
Hypothesis framework and require generated decision/provenance invariants
alongside regression examples during implementation. Security and Resiliency
extensions remain disabled; delivery safeguards above remain mandatory.

## Progress

- [x] Record and validate option A without ambiguity.
- [x] Define functional, safety and verification acceptance criteria.
- [x] Obtain requirements approval before workflow planning (`Approved`).

## Review

A) Approve & Continue to Workflow Planning.

B) Request Changes to these requirements.

X) Other (describe the requested direction).

[Answer]: A (user replied `Approved`).
