# GitHub Actions Release and Codex Review Corrections Plan

## Scope and authorization

The user explicitly authorized implementation, validation, and commit of the
GitHub Actions audit corrections, automated tag/release/PyPI delivery, and the
current GitHub Codex review findings.

## Testable properties

- PBT-02 round trip: generated valid direct-contact temperature multipliers and
  nested zone locations survive canonical serialization and reload.
- PBT-03 invariant: optimizer candidates accepted after solving satisfy bounds
  and constraints within the configured feasibility tolerance.
- PBT-06 stateful model: generated sequences of successful loads, targeting,
  and failed replacements preserve the observable `PinchProblem` model state
  after every command, including the empty sequence.
- PBT-08 reproducibility: all Hypothesis tests remain shrinkable and run under
  the repository's fixed CI seed.
- PBT-10 complementarity: generated properties supplement the existing concrete
  regressions for each corrected behavior.

## Execution plan

- [x] Step 1: Reconfirm the audited workflow defects and retrieve every current
  GitHub Codex review comment from PR 92.
- [x] Step 2: Reconcile the bot-created remote version commit without disturbing
  local audit evidence or the unrelated Excel workbook.
- [x] Step 3: Add failing contract tests for workflow triggers, least-privilege
  permissions, immutable actions, trusted release sequencing, and automated
  tag/GitHub-release/PyPI behavior.
- [x] Step 4: Replace PR-time mutation with read-only validation for `main` and
  `develop`, and make dependency installation reproducible from `uv.lock`.
- [x] Step 5: Implement main-push release preparation, guarded automatic tag and
  draft GitHub release creation, immutable trusted publishing, and post-PyPI
  GitHub release finalization.
- [x] Step 6: Align optimizer feasibility filtering with the configured solver
  tolerance and add example-based regression coverage.
- [x] Step 7: Add PBT-02 multiplier round-trip and PBT-06 transactional loader
  state-machine coverage with domain-valid generated inputs.
- [x] Step 8: Run focused workflow, optimizer, application, and property tests;
  validate YAML and static workflow security contracts.
- [x] Step 9: Run the complete configured-solver test suite, Ruff, documentation,
  distribution build, and patch hygiene checks.
- [x] Step 10: Review scope, update AI-DLC evidence and checkboxes, preserve
  unrelated files, and commit the verified improvements to `develop`.

## PBT compliance target

- PBT-02: applicable and blocking.
- PBT-03: applicable and blocking.
- PBT-04: N/A; no new idempotent application operation is introduced.
- PBT-05: N/A; no alternate algorithm with a reference oracle is introduced.
- PBT-06: applicable and blocking.
- PBT-07: applicable; strategies must generate valid zone trees, locations,
  multipliers, and command sequences.
- PBT-08: applicable and blocking.
- PBT-10: applicable and blocking.
