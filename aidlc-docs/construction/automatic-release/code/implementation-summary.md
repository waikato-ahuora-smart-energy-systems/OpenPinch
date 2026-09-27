# Automatic release implementation

## Outcome and boundaries

Local implementation restores automatic publication after successful `CI Main`
validation. The publisher remains standalone in `ci-publish.yml` for trusted
publishing compatibility. Manual new/resume remain available. No version bump,
commit, push, release dispatch or remote configuration change was performed.
Activation requires merging the workflow onto the default branch and observing
its hosted behavior; local verification does not establish OIDC compatibility.

## Source and state safeguards

- `scripts/plan_release.py` validates event and fresh API identity, selects a
  unique retained original artifact, verifies its archive and source proof,
  then classifies new, complete or blocked state without release mutations.
- `scripts/release_manifest.py` exposes its existing proof verifier for reuse;
  stage/finalize and the prior prerelease/Latest corrections are preserved.
- The automatic listener uses trusted default-branch automation code but binds
  publication to the triggering source commit and immutable original bundle.
- Completed-version checks recover original CI evidence and compare original
  GitHub assets and both package indexes. Later same-version build differences
  are not mistaken for conflicts in the original published release.
- Missing/expired/ambiguous original artifacts, failed lanes, source mismatch,
  tag-only state, drafts, prereleases, partial publication and hash conflicts
  stop automation. They do not cause overwrites or silent rebuilding.
- The standalone non-cancelling publisher lock is shared by manual and
  automatic runs. Main remains read-only; its cancellation cannot terminate
  an already-started separate publisher. Pending runs are not a FIFO queue.
- New automatic releases verify both indexes are absent before staging. Index
  failures are not absence; bounded reads fail closed and can be retried by
  rerunning the failed publisher job with the original source identity.

## Requirement traceability

Requirements 1-3: guarded completion listener, fresh source proof and original
artifact handoff, with no automatic bump or duplicate validation/build.
Requirements 4-6: original completion proof, strict blocked states and retained
manual recovery. Legacy 0.6.10 is not silently adopted or republished.
Requirements 7-9: existing index ordering, hashes, environment, permissions and
publication lock retained. Requirement 10: planner and publication summaries,
updated README and developer release/CI guides.

This summary amends the explicit-only initiation assumptions in the earlier
delivery functional and infrastructure designs. No application architecture,
numerical contract, validation lane, coverage threshold or dependency changed.

## Verification evidence

- Formal Build and Test handoff: final combined suite 386 passed, 6 expected
  skips in 84.71 seconds, including the added condition cases and Sphinx.
  Ruff, formatting, actionlint/ShellCheck, diff and lock-version checks pass.

- Pre-change release/workflow baseline: 86 passed in 6.85 seconds.
- First planner regression correctly failed collection before implementation.
- Focused planner/release/workflow run: 112 passed in 5.01 seconds.
- Extended planner/workflow/docs contracts: 82 passed in 12.63 seconds.
- Initial full packaging run: 313 passed, 6 skipped; only Sphinx heading
  underline failed. Fixed the heading; dedicated docs smoke passed in 7.75 seconds.
- Final full packaging rerun: 314 passed, 6 expected skips in 72.02 seconds,
  including warning-strict Sphinx. Subsequently added 72 job-condition matrix
  cases all pass (102 workflow tests total in 1.28 seconds). These extend the
  completed packaging collection rather than replacing existing checks.
- Ruff, changed-file formatting, actionlint with ShellCheck and diff checks pass.
- During development the new condition matrix initially used pytest's reserved
  parameter name `request`; renamed to `request_result` before verification.

## PBT compliance

PBT-01 compliant: authorization, exact identity, completion and no-mutation
properties identified in the approved plan. PBT-02 compliant: existing
manifest/archive round trips retained. PBT-03 compliant: event authorization
and completion invariants generated. PBT-04 compliant: duplicate reads are
idempotent after each generated observation change. PBT-05 compliant:
independent authorization and completion predicates. PBT-06 compliant:
generated empty/nonempty observation sequences cover complete/draft/partial/
expired transitions, with real planner/proof logic and mutation-rejecting fake
boundaries; existing publication recovery state machine retained. PBT-07
compliant: bounded domain observations and existing structured manifests.
PBT-08 compliant: seed 20260926 and default shrinking retained, ordinary CI
selection. PBT-09 compliant: existing Hypothesis dependency unchanged.
PBT-10 compliant: explicit regression and CLI cases alongside properties.
Security and Resiliency extensions disabled and skipped.

## Reproduction

Run `.venv/bin/pytest tests/packaging -q --hypothesis-seed=20260926`.
Run Ruff check and format-check for the changed Python files, actionlint with
ShellCheck, and `git diff --check`. Tests use fake GitHub/index boundaries and
do not publish. Full numerical/solver reruns are outside this delivery-only
change; existing hosted lane definitions and 95-percent coverage gate remain.

The execution plan under `aidlc-docs/inception/plans/` is ignored by existing
repository rules; include it explicitly if a later authorized commit is meant
to retain that planning artifact. No ignore rule was changed.

## Review

A) Continue to Next Stage: approve Code Generation and proceed to Build and Test.

B) Request Changes to the implementation.

X) Other (describe the requested direction).

[Answer]: A (user replied `Approved`).
