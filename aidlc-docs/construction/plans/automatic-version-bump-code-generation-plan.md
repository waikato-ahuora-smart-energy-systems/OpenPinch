# Automatic version bump code generation plan

Approval: user explicitly approved through code design, implementation and local
commit. This supersedes individual pause points for this bounded delivery unit.
No push, merge, remote configuration or publication is authorized.

Context: existing scripts/CI policy, version metadata, preparation/evidence models
and standalone publisher. No application services, database or frontend; user
stories skipped. Source files remain at repository root.

- [x] 1. Review functional design and existing delivery interfaces; record approval.
- [x] 2. Implement strict version/preparation codec, exact diff validation and
  GitHub coordinator in scripts/release_preparation.py; pure decisions separate
  from external writes. Preserve manual protected merges and retry identity.
- [x] 3. Implement review evidence verifier and CI gate adapter; connect shared
  validation and publisher proof without trusting skipped jobs or PR metadata.
- [x] 4. Add coordinator/review notification workflows and wire PR/main/develop
  routing, fresh package verification and release-preparation merge gate.
- [x] 5. Add example and Hypothesis tests (codec, invariant, oracle, idempotence,
  state sequences, constrained generators, seed 20260715, shrinking) and workflow
  contracts. Exercise Git/API boundaries through fakes, not live publishing.
- [x] 6. Update release documentation, implementation summary and build/test
  instructions. Explain activation, review approval, stale preparation and recovery.
- [x] 7. Run focused and complete packaging tests, Ruff, actionlint, metadata,
  build checks and diff checks. Audit changes; repair failures before commit.
- [x] 8. Update stage progress and audit; stage only scoped files and commit locally.

PBT-01 properties in functional design drive steps 2, 3 and 5. PBT-02 through
PBT-10 covered by generated codec/decision/state tests plus concrete regressions,
existing Hypothesis and fixed-seed verification. Disabled extensions skipped.

## PR 103 corrective follow-up

Approved by user "Go." after review assessment. No commit, push or merge is
included in this follow-up. Preserve the existing token/approval activation path.
Invariant: only explicitly identified, completed metadata-only runs may be
ignored; newer actual, incomplete or ambiguous validation remains authoritative.

- [x] 9. Add metadata-only marker and conservative review-run classification.
- [x] 10. Add concrete regressions and generated run-order/selection oracle tests.
- [x] 11. Run packaging tests and lint; record evidence and review response.

Verification: 434 passed, 6 expected skips in 142.13 seconds; Hypothesis seed
20260715. Ruff, formatting, actionlint and git diff --check pass. Review response
posted to PR 103 (issuecomment-5850389548); threads remain unresolved. Local
changes are not committed or pushed.

PBT-01/03/05/07/08/09/10 apply to selection invariant and oracle (Hypothesis,
seed 20260715, normal shrinking). PBT-02/04/06 N/A: no new codec, mutation or
state machine. Security/Resiliency disabled and skipped. Plain Markdown only.
