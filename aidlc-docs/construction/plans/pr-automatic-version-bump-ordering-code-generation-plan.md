# Pull-Request Automatic Version-Bump Ordering Code-Generation Plan

## Unit Context

- **Unit**: GitHub Actions PR release preparation.
- **Requirements**: PRVB-01 through PRVB-09 and PRVB-NFR-01 through
  PRVB-NFR-04.
- **Dependencies**: Existing bump configuration, strict release-version script,
  PR gate, and main-branch publication workflow.
- **Interfaces**: GitHub pull-request event metadata, the PR head branch, and
  canonical version files.
- **Persistent application state**: None.

## TDD Execution Plan

- [x] **Step 1 - Red workflow contracts**: Replace the obsolete prohibition on
  PR mutation with assertions requiring a same-repository main-PR bump job,
  exact bump-tool pin, no-tag execution, canonical metadata checks, release-job
  dependency, latest-head checkout, fork-safe behavior, and aggregate-gate
  integration. Run the focused test and record the expected failure.
- [x] **Step 2 - Automatic bump job**: Restore a least-privilege `bump-version`
  job for ready same-repository pull requests targeting `main`. Select the bump
  part from labels/title with patch default, bump only when candidate equals
  base, validate synchronized metadata, and push the generated commit.
- [x] **Step 3 - Ordered validation and aggregate gate**: Make
  `release-version` wait for bump evaluation, explicitly check out the latest
  PR head, retain read-only fork validation, and extend `pr-gate` to enforce
  correct same-repository, fork, main, and develop outcomes.
- [x] **Step 4 - Documentation and evidence**: Update README and developer RTD
  release instructions, then add the implementation summary and synchronize
  AI-DLC state and audit evidence.
- [x] **Step 5 - Focused verification**: Run packaging/workflow tests, parse all
  workflow YAML, run Ruff and formatting checks on changed Python, and build
  warning-strict documentation.
- [x] **Step 6 - Regression and scope verification**: Run the configured
  non-solver suite with the 95 percent coverage gate, inspect the complete diff,
  run patch hygiene checks, and confirm no version file was bumped locally.

## Property-Based Testing Compliance

- **PBT-01**: Assessed; workflow convergence is an invariant, but deterministic
  event/job contract tests are a better oracle than generated examples.
- **PBT-02 through PBT-10**: N/A; no domain serialization, numerical model,
  generated input space, stateful application service, or alternative algorithm
  is introduced.

This plan is the single source of truth for code generation. The user's explicit
standing authorization approves all six steps and continuation through Build
and Test unless material input is required.
