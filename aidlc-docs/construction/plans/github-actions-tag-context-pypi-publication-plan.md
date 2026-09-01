# GitHub Actions Tag-Context PyPI Publication Plan

## Scope and authorization

The user explicitly authorized implementation and commit of the recommended
two-stage release hand-off. The change must publish the GitHub Release before
production PyPI, execute the production deployment from the matching version
tag, retain the protected `pypi` environment and reviewer gate, and reuse the
validated release artifacts without rebuilding them.

## Testable contract

- A main-ref release run validates and builds distributions once, publishes
  them to TestPyPI, and publishes the GitHub Release before dispatching the
  production phase.
- The production phase is a separate invocation of the same trusted workflow
  at the exact `vX.Y.Z` tag ref, preserving the existing PyPI Trusted Publisher
  workflow identity.
- The production phase rejects branch refs, mismatched tag inputs, draft
  releases, prereleases, mismatched distribution names, and altered artifacts.
- The production job enters the existing `pypi` environment only after release
  validation and publishes the downloaded release distributions without a
  rebuild or `skip-existing` escape hatch.
- Version metadata advances to 0.6.3 because `v0.6.2` already identifies the
  previous main commit and cannot contain the corrected workflow.

## PBT assessment

PBT-02 through PBT-07 and PBT-10 are N/A because the change is a fixed,
declarative GitHub Actions dependency and permission contract rather than
Python business logic, transformation, serialization, algorithm, or mutable
state. PBT-08 remains compliant through the unchanged fixed Hypothesis seed in
all repository test jobs. PBT-09 remains compliant through Hypothesis in the
locked development dependency set. Example-based static workflow regressions
are the appropriate TDD mechanism for this change.

## Execution plan

- [x] Step 1: Record authorization, inspect the stopped run and current
  environment contract, and preserve unrelated workspace changes.
- [x] Step 2: Add failing static workflow regressions for published-release
  ordering, tag-ref dispatch, release-asset integrity validation, protected
  production publication, least privilege, and removal of the main-ref PyPI
  path.
- [x] Step 3: Refactor `ci-publish.yml` into main-ref orchestration and an
  explicitly dispatched tag-ref production phase while retaining its filename
  for PyPI Trusted Publishing.
- [x] Step 4: Generate and publish distribution checksums, validate the public
  release and exact tag-bound assets, and publish only those assets to PyPI.
- [x] Step 5: Update maintainer documentation to describe the tag-context
  hand-off, approval point, retry behavior, and public-release-before-PyPI
  tradeoff.
- [x] Step 6: Advance canonical version metadata from 0.6.2 to 0.6.3 so the new
  workflow can exist at a fresh immutable release tag.
- [x] Step 7: Run focused workflow tests, YAML parsing, release/version gates,
  formatting, documentation, and package build verification.
- [x] Step 8: Run the complete applicable test suite, review the final diff for
  necessity and security, update AI-DLC evidence and checkboxes, and commit the
  verified changes to `develop`.
