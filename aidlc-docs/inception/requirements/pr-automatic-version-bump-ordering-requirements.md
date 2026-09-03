# Pull-Request Automatic Version-Bump Ordering Requirements

## Intent Analysis

- **User request**: Correct the pull-request release gate so the configured
  automatic version bump happens before the candidate release version is
  compared with the base branch.
- **Request type**: Bug fix restoring intended release automation.
- **Scope estimate**: One GitHub Actions workflow, its static contract tests,
  and release-process documentation.
- **Complexity estimate**: Simple implementation with moderate CI sequencing
  risk because a bump changes the PR head while its workflow run is active.
- **Workspace**: Brownfield Python library. Existing reverse-engineering
  artifacts remain current; no reverse-engineering refresh is needed.

## Functional Requirements

### PRVB-01 - Automatic bump applicability

For a ready, non-draft, same-repository pull request targeting `main`, the PR
workflow must automatically advance the project version before forward-version
validation when the candidate version still equals the base version.

### PRVB-02 - Bump selection

The bump part must follow the established policy: a `major`, `minor`, or
`patch` pull-request label takes precedence; otherwise a matching
`[major]`, `[minor]`, or `[patch]` title marker is used; otherwise the workflow
defaults to `patch`.

### PRVB-03 - Canonical metadata synchronization

The bump must use the repository's `.bumpversion.toml` contract so
`pyproject.toml`, the OpenPinch entry in `uv.lock`, and
`.bumpversion.toml` advance together. The bump must not create a tag; the
release workflow remains the sole tag owner.

### PRVB-04 - Ordering and updated-head validation

Forward-version validation must depend on successful bump evaluation. After a
bump commit is pushed, validation must read the latest pull-request head branch,
not the immutable SHA or synthetic merge commit captured when the workflow run
started. This ensures the validator sees the automatically updated version.

### PRVB-05 - Repeat-run behavior

If the candidate version is already strictly greater than the base version,
the automatic-bump step must make no commit and must succeed. Any subsequent
synchronize event or manual rerun must therefore converge without a second
bump.

### PRVB-06 - Divergent-version behavior

If the candidate version is lower than the base version, automation must fail
closed with an actionable diagnostic instead of applying repeated speculative
bumps.

### PRVB-07 - Fork behavior

The workflow must never attempt to write to a fork. Fork pull requests targeting
`main` must retain read-only forward-version validation and require contributors
to provide a forward version in their branch.

### PRVB-08 - Branch scope

Pull requests targeting `develop` must retain their existing validation behavior
without release-version bumping or forward-release validation.

### PRVB-09 - Aggregate gate integration

The `pr-gate` aggregate must require a successful bump evaluation for a
same-repository PR targeting `main`, a skipped bump for a fork targeting
`main`, a successful release-version result for both, and both jobs skipped
when the base is `develop`.

## Non-Functional Requirements

### PRVB-NFR-01 - Least privilege

Repository write permission must be scoped to the automatic-bump job. Other PR
jobs remain read-only, and no pull-request write permission is introduced unless
the implementation actually posts or edits pull-request content.

### PRVB-NFR-02 - Reproducibility

Actions and the bump tool must use immutable or exact versions. Dependency
installation must retain the frozen-lock workflow used by the current CI.

### PRVB-NFR-03 - Testability

Static workflow tests must fail before implementation and then prove job
applicability, dependency order, updated-head checkout, synchronized bump-tool
use, no-tag behavior, repeat-run prevention, fork safety, and aggregate-gate
semantics. All workflow YAML must parse successfully.

### PRVB-NFR-04 - Documentation consistency

The README and developer release documentation must describe automatic PR
version advancement and the manual requirement for fork pull requests.

## Edge Cases

- Candidate equals base: perform exactly one selected bump, commit, and push.
- Candidate exceeds base: do not bump again; validate the existing candidate.
- Candidate is below base: stop with a clear stale/divergent-version error.
- A subsequent `synchronize` run or manual rerun after a bump detects the
  forward version and remains mutation-free.
- A fork cannot receive a base-repository token write: skip mutation and perform
  read-only validation.
- A pull request is retargeted between `develop` and `main`: the existing
  `edited` trigger reevaluates the release policy.

## Property-Based Testing Assessment

The enabled Property-Based Testing extension was assessed. Hypothesis is not a
useful fit for declarative GitHub Actions dependency and permission structure.
PBT-01 identifies job-order and repeat-run invariants, but deterministic static
YAML contract tests are the stronger oracle. PBT-02 through PBT-10 are N/A
because this change adds no serialization round trip, numerical computation,
stateful application model, generated domain data, or algorithmic oracle.

## Out of Scope

- Changing the main-branch tag, GitHub Release, TestPyPI, or PyPI publication
  sequence.
- Automatically writing to fork branches.
- Adding a new versioning dependency to the OpenPinch runtime package.
- Changing the strict `X.Y.Z` release-version policy.
- Bumping the repository version as part of this bug-fix implementation.

## Acceptance Summary

The PR workflow advances equal same-repository versions exactly once, validates
the latest bumped head afterward, remains read-only for forks, leaves develop
PRs unchanged, and has deterministic tests and synchronized release-process
documentation.
