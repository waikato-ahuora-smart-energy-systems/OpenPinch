# GitHub Actions Comprehensive Hardening Code Generation Plan

## Unit Context

- **Unit**: GitHub Actions validation, release, publication, and repository controls
- **Workspace**: Brownfield Python package at `/Users/timothyw/Github_Local/OpenPinch`
- **Application and configuration paths**: `.github/workflows/`, `scripts/`, and `tests/`
- **Documentation paths**: `README.md`, `docs/`, and `aidlc-docs/`
- **Dependencies**: GitHub Actions artifacts and environments, TestPyPI and PyPI JSON APIs, IDAES solver binaries, uv, pytest, and the existing packaging scripts
- **Interfaces**: Pull-request checks, develop-branch validation, main push release orchestration, tag-context production publication, and protected PyPI deployment

## Traceability

- AUDIT-01: Enforce successful PR validation before main merges.
- AUDIT-02: Bind production artifacts to a verified immutable source-run artifact.
- AUDIT-03: Prevent manual main dispatch from entering the release path.
- AUDIT-04: Never overwrite an already-public GitHub Release.
- AUDIT-05: Make TestPyPI and PyPI publication retry-safe after partial or ambiguous uploads.
- AUDIT-06: Run solver validation before merging to main.
- AUDIT-07: Revalidate retargeted pull requests.
- AUDIT-08: Remove redundant CI work and parallel uv cache-write contention.
- AUDIT-09: Harden repository Actions, branch, environment, and tag settings.
- AUDIT-10: Re-audit the completed workflow and repository controls.

## Execution Plan

- [x] **Step 1 - Pin failing workflow contracts**: Add example-based tests for source-run artifact provenance, event separation, public-release immutability, retry-safe package-index verification, PR aggregation, solver gating, retarget events, cache writers, and develop/PR deduplication.
- [x] **Step 2 - Build the package-index verifier under TDD**: Add a standard-library script and unit tests that compare expected distribution filenames and SHA-256 digests with TestPyPI or PyPI JSON responses, accepting only absent or exact partial state before upload and requiring exact complete state afterward.
- [x] **Step 3 - Anchor production to an immutable source artifact**: Replace the caller-supplied manifest digest with verified source run, attempt, artifact identity, and artifact digest inputs; download that immutable artifact in the tag run; validate source event, branch, commit, prerequisite jobs, artifact metadata, release files, and checksums before publication.
- [x] **Step 4 - Make release mutation fail closed**: Restrict main release orchestration to push events, reject public-release reuse, verify annotated tags, and reuse only exact draft assets without `--clobber`.
- [x] **Step 5 - Make package publication idempotent**: Add TestPyPI and PyPI preflight, exact-hash partial retry support, conditional upload with `skip-existing`, complete post-upload validation, and independently retryable production verification.
- [x] **Step 6 - Strengthen and streamline CI**: Add PR retarget handling, a main-PR solver gate, one required aggregate result, one uv cache writer per platform/key, and develop-run suppression while an open develop-to-main PR is validated separately.
- [x] **Step 7 - Update maintainer documentation**: Document immutable artifact provenance, safe retry behavior, required checks, release recovery boundaries, and repository protection expectations.
- [x] **Step 8 - Run focused validation**: Execute the new script tests, packaging/workflow contracts, YAML parsing, formatting, linting, documentation, build, and patch-hygiene checks.
- [ ] **Step 9 - Apply safe live GitHub controls**: Least-privilege Actions defaults and full-SHA enforcement are applied, the PyPI environment remains reviewer-protected and tag-scoped, stable release tags are protected, and the stale Code Owners requirement is removed. Requiring ``pr-gate`` remains pending until its first successful pushed run.
- [x] **Step 10 - Run complete validation and re-audit**: Execute the complete configured-solver test suite, inspect the final diff and live settings, perform a second threat/retry/concurrency audit, record PBT applicability, and report any residual external sequencing requirement.

## PBT Applicability

The workflow is declarative and the package-index verifier is deterministic I/O validation. PBT-02 through PBT-07 and PBT-10 are expected to be N/A unless implementation introduces a general transformation with a meaningful generated-input property. PBT-08 remains applicable through the fixed Hypothesis seed in CI, and PBT-09 remains satisfied by Hypothesis in the development dependency set.

This plan is the single source of truth for this correction.
