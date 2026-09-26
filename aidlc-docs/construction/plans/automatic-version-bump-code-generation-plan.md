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
