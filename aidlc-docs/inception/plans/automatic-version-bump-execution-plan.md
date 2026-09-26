# Automatic patch-version execution plan

## Approved baseline

Requirements: `../requirements/automatic-version-bump-requirements.md`, approved
with `Approve`. Patch by default; explicit major/minor changes retained; every
published artifact must come from the validated bumped source commit. The user's
subsequent amendment requires normal review first and reuse of already-passed
tests on main where equivalence is proven, rather than unconditional reruns.
The existing automatic publisher in commit dc0bb5ac remains the implementation
baseline. Application APIs, thermodynamic behavior and dependencies are unchanged.

## Recommended protected-branch-compatible path

1. Normal review and required tests establish evidence for the proposed changes
   before automatic version preparation. A privileged coordinator verifies that
   evidence and current review state; it must not execute untrusted PR code with
   write credentials. Main validation is no longer the preparation trigger.
2. A separate narrowly privileged job creates a patch-version commit on a
   dedicated automation-owned branch and opens a version-preparation PR into
   `develop`, based on the reviewed develop head.
   Update only version metadata plus a strictly validated preparation record.
   Do not write directly to protected main or alter contributor branches.
3. Merge under existing protections. Automation does not approve or merge its PR.
   First merge the bump PR into `develop`, then merge the existing updated
   `develop` to `main` PR. This is the user's selected option A in
   `../requirements/automatic-version-bump-flow-questions.md`. Do not assume
   review approval survives a new commit; obtain renewed approval when required.
   The original reviewed PR is retained, not superseded by a new release PR.
4. Main validation rebuilds and checks the bumped distributions while reusing
   proven normal-review results for unaffected test lanes.
   The standalone publisher verifies preparation identity and publishes those
   exact tested bytes. The preparation merge must not create another bump PR.
5. An explicit reviewed major/minor increase can enter validation/publication
   without an additional patch bump, using an independently verified baseline.
   Functional Design will pin the exact baseline and comparison rules.

Requested sequence: normal review; automatic patch-bump PR; merge; main
validation with proven test reuse; automatic release publishing. The former
main-validation-first proposal is superseded. The selected path has two merges:
bump into develop, then develop into main. No automatic merging is added.

## Review evidence reuse

The existing `reuse_develop_ci.py` only proves exact tree equality for
develop-to-main PR validation. It is not sufficient for the new main path.
Design a lane-level proof binding repository, PR, reviewed head/base and tested
merge tree, workflow/policy identity, run attempt and successful required jobs.
Resolve any existing reuse chain to actual successful executions; a green
aggregate containing skipped jobs is not proof on its own.

Permit only parsed version-field substitutions and validated preparation data
between the tested candidate and released source, not blanket exclusions of
`pyproject.toml` or `uv.lock`. Dependency, build configuration, tests, workflows,
application changes or an incompatible base invalidate affected evidence.
Reuse only lanes demonstrably unaffected by the bump. Fresh main checks always
cover version consistency, wheel/sdist build and metadata, distribution install
smokes and artifact provenance. Version-sensitive lanes must rerun.

Missing, stale, failed, ambiguous or unverifiable proof falls back to normal
validation. The publisher and validation gate must verify the evidence chain
under a revised policy instead of merely accepting skipped main jobs. Never
publish the pre-bump review artifact. Functional Design must define precise
lane mappings, policy compatibility and generated false-acceptance regressions.

## Identity, races and recovery

- Use a stable preparation identity based on the original source commit and
  target release series; retries resolve the existing record/branch/PR first.
- Allocate against verified source/version and existing release identities;
  never derive success solely from a commit message, branch name or PR title.
- Define an allowlisted diff for generated commits. Verify metadata consistency,
  source ancestry and recorded request identity before trusting preparation.
- Serialize allocations and compare expected remote heads before writes.
  Never force-push over user changes. A stale preparation must be identified
  and safely refreshed or blocked, not silently merge a stale version.
- Coalesce compatible changes arriving while preparation is open where safe;
  document when maintainer action is needed instead of opening duplicate bumps.
- Repeated events, reruns and recovery must reuse the allocated version and
  original publication bundle. Never allocate another patch to hide a failure.
- Retain manual verified recovery, immutable releases and the legacy 0.6.10
  boundary. No actual recovery/publication is performed during implementation.

## Authentication and activation

Default implementation uses job-scoped `contents: write` and
`pull-requests: write` for the automation-owned branch/PR only. PR/main
validation stays read-only. Publishing keeps its existing job-scoped OIDC.
Do not add a personal token, GitHub App secret or branch-protection bypass.

GitHub's current documentation says a PR created/updated using `GITHUB_TOKEN`
starts its opened/synchronize/reopened workflows in an approval-required state.
A maintainer must approve those runs. Repository policy must also allow Actions
to create PRs. If either prerequisite is unavailable, report a configuration
blocker instead of weakening checks or silently dispatching substitute proof.
An optional GitHub App for unattended PR checks would require separate approval
and configuration; it is not assumed by this plan.

Primary references checked during planning:
- https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow
- https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches

Actual repository permissions must be inspected during the design/activation
handoff where access permits. No remote setting is changed as part of planning.

## Scope, risks and component sequence

One existing delivery unit gains version-preparation state and source writes.
Risk: high at source-write/publication boundaries; numerical/application risk
is low. Reverting local code is straightforward, but external version/tag or
publication state must never be rewritten as rollback.

Implement in dependency order:
1. Pure allocation/identity helpers and state/property regressions in scripts
   and packaging tests; reuse version parsing and metadata checks.
2. Preparation branch/PR adapter with expected-head checks and strict diff
   constraints, tested through fake git/GitHub boundaries.
3. Workflow routing between prepare, already prepared, publish, no-op and
   blocked states; preserve the existing publisher's proof and artifact checks.
4. README/release documentation and behavioral workflow contracts.
5. Packaging/Hypothesis, docs, Ruff, formatting, actionlint/ShellCheck and diff
   verification; report hosted activation separately from local results.

## Stage selection and progress

- [x] Workspace/requirements context resumed; existing reverse engineering reused.
- [x] Requirements approved and this execution plan prepared.
- [x] User Stories skipped by prior explicit user direction.
- [x] Incorporate review-first ordering and evidence-backed test reuse amendment.
- [x] Resolve bump-PR target: develop, followed by the existing main PR (A).
- [x] Approve the revised execution plan before Functional Design (`Approve`).
- [x] Functional Design: define version baseline, request schema, transitions,
  retry/race oracle, trusted preparation recognition, review-evidence reuse and
  permission boundaries.
- [x] Approve Functional Design before Code Generation planning.
- [x] Code Generation Part 1: detailed file/method/test checklist and approval.
- [x] Code Generation Part 2: implementation plus focused verification.
- [x] Build and Test: combined delivery/packaging checks and activation handoff.

Separate Application Design/Units Generation are skipped: a single existing
delivery component, no application service decomposition. Separate NFR
Requirements/Design reuse prior bounded I/O, immutable identity, minimal
permissions and full-validation requirements. No new hosting infrastructure
is introduced; Functional Design must include the GitHub permission and
activation mapping rather than create a separate Infrastructure Design stage.
Operations remains a placeholder. No commit, push, PR creation, merge,
credential configuration or live release is authorized by plan approval.

Text workflow: approved requirements; plan approval; focused Functional Design
and approval; Code Generation planning and approval; implementation; Build and
Test; separately authorized deployment. Skipped stages are listed above.

## PBT planning compliance

PBT-01: specify monotonic allocation, consistent metadata, idempotence and
no-mutation-on-conflict properties during Functional Design. PBT-02: preparation
record round trips. PBT-03: generated allocation/provenance invariants. PBT-04:
duplicate preparation/retries reuse identity. PBT-05: independent allocation
and state-decision oracle. PBT-06: generated change/retry/race sequences with
real helpers and fake external effects. PBT-07: bounded structured versions,
commit identities and release observations. PBT-08: fixed seed and shrinking.
PBT-09: existing Hypothesis stack. PBT-10: explicit observed and critical edge
cases complement generated tests. All planning obligations covered; no claim
of implementation compliance yet. Security/Resiliency extensions disabled.

## Review

A) Approve & Continue to Functional Design with the selected develop-targeted
bump PR, review-first preparation and evidence-backed reuse on main.

B) Request Changes to the proposed approach.

C) Add a skipped stage (specify which).

X) Other (describe the requested direction).

[Answer]: A (user replied `Approve`).
