# Reuse completed develop validation on main PRs

## Authorized requirements
Keep full tests on every develop push. A ready same-repository develop-to-main
PR may reuse a completed successful CI Develop push run for its exact head SHA.
Require the PR merge tree to equal that head tree, so main-only integration
changes never bypass testing. Verify all expected develop job/matrix results
are successful; an overall success with skipped tests is insufficient.
No success, running/failed/stale runs or unavailable API means normal PR checks.
Keep the required PR test status successful via a lightweight reuse job path;
keep version bump/release validation, solver tests and post-merge release gates.
Do not change remote branch protections, publish, or post messages.

## Investigation
Existing develop workflow skipped tests when a ready main PR existed, so remove
that gate to make push validation authoritative. Main ruleset requires test;
keep that name and existing pr-gate. Legacy protection read returned 403; active
ruleset was read successfully. No changes to unrelated local edits.

## Design and testable properties
Add a standard-library script to check eligibility, exact commit and merge-tree
identity, latest develop run conclusion and every expected job result. Only
successful proof emits reuse=true; incomplete/error cases emit false.
A PR preflight exposes that decision. test retains a success-producing path;
other shared jobs skip only on proved reuse; pr-gate accepts their skipped state
only when the preflight succeeded and proved reuse. Solver checks always run
on main PRs because develop does not run them.
Properties: changing any required successful job to failed/skipped/missing must
prevent reuse; stale SHA and changed merge trees prevent reuse; failed lookup
never suppresses PR validation. Fixed-seed Hypothesis complements explicit
workflow and decision examples. PBT-01/03/05/07/08/09/10 applicable; PBT-02/04/06
N/A (no inverse, persistent state or normalization). Security/Resiliency disabled.
Single tooling unit; stories, application decomposition, infrastructure and
operations skipped. Prior rules loaded in this workflow are reused.

## Execution
- [x] 1. Inspect triggers, main ruleset and user preference.
- [x] 2. Add reuse decision script and regression/property tests.
- [x] 3. Keep develop validation unconditional; wire safe reuse into PR jobs/gate.
- [x] 4. Validate YAML, decision tests, packaging contracts and lint; document behavior.
