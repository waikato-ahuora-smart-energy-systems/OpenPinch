# Develop validation reuse

Full validation runs on every develop push, including when a main PR is open.
A same-repository develop-to-main PR reuses shared checks only when the latest
develop push run for its exact head commit passed all 14 expected job instances
and the PR merge tree matches that commit's tree. Skipped jobs are not proof.

Missing, running, failed or stale validation, a changed merge tree, or an API
lookup failure causes normal PR validation. Opening a PR before develop finishes
can therefore still produce duplicate runs. The PR job summary records the
decision and links the reused run when available.

The required test job remains successful on verified reuse. Solver tests and
version/release checks still run on main PRs. Publishing and branch settings
are unchanged. The preflight explicitly installs the project's Python version.

## Verification

- Packaging suite: 167 passed, 3 skipped in 106.75 seconds.
- Final focused decision/workflow suite after Python setup adjustment: 66 passed.
- Ruff and git diff whitespace checks passed.
- YAML parsing, job dependency references, immutable action pins and bash syntax
  passed; expanded develop matrices match all 14 expected job names, and shared
  PR commands match develop commands.
- Fixed-seed property tests reject any required job being missing or unsuccessful.
  Regression cases cover stale commits, changed trees, failed lookups and PR gates.

No push or live execution of the modified GitHub Actions workflows was performed.
The first remote run remains the integration check of runner and API wiring.

## Extension compliance

PBT-01, PBT-03, PBT-05, PBT-07, PBT-08, PBT-09 and PBT-10 are compliant through
documented decision invariants, generated job mutations, fixed seed, and passing
regressions. PBT-02, PBT-04 and PBT-06 are N/A: no inverse operation, persistent
state or normalization is introduced. Security and Resiliency remain disabled.
