# GitHub Actions Review Provenance and Retry Corrections Plan

## Scope and authorization

The user authorized correction of both actionable Codex findings on PR 94.
The existing 0.6.3 release identity remains unchanged because it is still
strictly forward of main at 0.6.2 and no release tag has been created for the
PR commit.

## Testable contract

- The release manifest is created by the unprivileged build job, and its
  SHA-256 digest is carried through the tag dispatch independently of mutable
  GitHub Release assets.
- The draft-release job must reproduce the build-owned manifest digest before
  publishing any release assets.
- The tag-ref job must verify the downloaded manifest against the immutable
  dispatch input before parsing it or trusting the distribution hashes.
- PyPI upload and PyPI API availability verification are separate jobs, so a
  transient verification failure can be retried without repeating a successful
  immutable upload.
- Documentation directs maintainers to use **Re-run failed jobs** on the
  original tag run, preserving its validated inputs.

## PBT assessment

PBT-02 through PBT-07 and PBT-10 are N/A because this is a deterministic
declarative workflow permission, provenance, and retry contract. PBT-08 remains
compliant through the unchanged fixed Hypothesis seed in CI, and PBT-09 remains
compliant through the locked Hypothesis dependency. Example-based static
workflow regressions are the appropriate TDD mechanism.

## Execution plan

- [x] Step 1: Inspect both PR 94 review threads, validate the findings against
  the local workflow, record authorization, and preserve the existing audit
  change.
- [x] Step 2: Add failing workflow regressions for a build-owned manifest
  digest, tag-dispatch digest transport, pre-use digest verification, and an
  independently retryable PyPI verification job.
- [x] Step 3: Generate the manifest and its digest in the unprivileged build
  job, verify it before release creation, and pass the digest into the tag-ref
  workflow invocation.
- [x] Step 4: Verify the immutable manifest digest before trusting downloaded
  release assets, and split PyPI API verification into a dependent unprivileged
  job.
- [x] Step 5: Update maintainer documentation to prescribe retrying failed jobs
  on the original tag run rather than starting a fresh production dispatch.
- [x] Step 6: Run focused workflow tests, YAML parsing, release checks,
  documentation, distributions, Ruff, formatting, and the complete applicable
  test suite.
- [x] Step 7: Review the final diff for necessity and least privilege, then
  update AI-DLC evidence and plan checkboxes.
