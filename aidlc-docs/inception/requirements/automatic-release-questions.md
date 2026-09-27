# Automatic release restoration

The requested behavior replaces the earlier explicit-release initiation policy:
successful main validation should automatically lead to TestPyPI, PyPI and
GitHub publication, retaining exact-artifact verification and recovery guards.
PR/develop validation must never publish. No release is being dispatched now.

## Question 1: Version preparation

How should automatic publishing obtain a fresh package version?

A) Publish explicitly prepared versions (recommended): keep version updates
in reviewed commits. After successful main validation, automatically publish
an unpublished committed version. Already completed versions are a reported
no-op; interrupted or conflicting versions require verified recovery.

B) Release every main merge: automatically prepare a patch-version bump
before validating and building the release candidate. This also restores
automated source/version writes and requires loop prevention and separate
validation of the changed commit.

X) Other (describe the desired versioning policy).

[Answer]: A

Policy superseded by the subsequent user request, "Version bump should be
automatic." Bump selection is pending in `automatic-version-bump-questions.md`.
The answer above is retained as historical context, not the current requirement.

## Existing state and constraints

- Current committed version: 0.6.10; local tag v0.6.10 already exists.
- Never overwrite published distributions or move an existing tag.
- Do not reuse a new build as recovery evidence for an existing version.
- Retain manual resume, bounded verification and prerelease rejection.
- Retain existing publisher/environment protections.
- Property-Based Testing enabled; Security and Resiliency extensions disabled.
- User stories remain unnecessary for this delivery-only policy change.

## Progress

- [x] Inspect current main, publishing, version and regression-test contracts.
- [x] Identify the version-selection decision before changing publish behavior.
- [x] Resolve version policy and finalize requirements.
- [ ] Approve implementation plan, implement, and verify locally.
