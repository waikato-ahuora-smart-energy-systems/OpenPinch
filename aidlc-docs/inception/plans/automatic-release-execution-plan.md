# Automatic release restoration execution plan

## Approved baseline and impact

Requirements: `../requirements/automatic-release-requirements.md`, approved
with `Approved`. Version policy A is fixed: reviewed commits prepare versions;
CI never bumps or writes source. This extends the existing delivery unit.

Affected owners are main/publish workflow orchestration, release-state helpers,
packaging tests and release documentation. No application API, thermodynamic
logic, dependency stack or branch-model change is needed. Existing delivery
designs remain the baseline except for explicit-only initiation.

Risk is high at publication boundaries because package uploads cannot simply
be rolled back. Code rollback is straightforward; published artifacts must
remain immutable. Testing complexity is moderate: fake external boundaries
must exercise real decision and provenance logic, not merely YAML strings.

## Proposed implementation direction

1. Keep `ci-main.yml` as the full validation owner. Trigger the existing
   standalone `ci-publish.yml` on successful CI Main completion, resolving
   exact artifact ID, digest, run and candidate identity from verified evidence.
   Retain manual dispatch for deliberate new releases and verified recovery.
2. Do not revalidate or rebuild the automatic candidate. Preserve full
   original-source evidence checks in the publisher. Confirm reusable-workflow
   input, permission, job-name and trusted-publisher semantics against official
   documentation before implementing the handoff.
3. Add a read-only release-state decision before mutations. Classify a new
   version, verified completed version, or blocked state. A completed-version
   check uses the original manifest/distributions and tag provenance, not the
   current same-version build. Verify both indexes and stable GitHub state.
   Missing or inconsistent original evidence fails closed. Do not relax
   recovery provenance rules to make a no-op succeed.
4. Share one non-cancelling publication lock across manual and automatic
   entry points. Remove the parent cancellation path that could interrupt an
   active publisher. Recheck state while serialized to handle duplicate runs.
   Document that GitHub concurrency is not a guaranteed FIFO queue.
5. Preserve TestPyPI then PyPI then GitHub finalization, exact hashes, deadlines,
   prerelease rejection and non-forced Latest selection. Retain the `pypi`
   environment and publishing workflow filename. No implicit environment or
   trusted-publisher configuration changes.

Dependencies: main validation produces the artifact and proof; release-state
inspection selects publish/no-op/blocked; the existing publisher owns staged
mutations and recovery. PR and develop stay read-only and cannot call the
automatic publishing path with authority.

## Stage selection

- [x] Workspace Detection: existing repository and merged delivery unit inspected.
- [x] Reverse Engineering: reuse current architecture and live delivery inspection.
- [x] Requirements Analysis: completed and approved; option A recorded.
- [x] User Stories: skip by prior explicit user direction and internal scope.
- [x] Workflow Planning: this plan prepared.
- [x] Workflow Planning approval (`Approved`).
- [x] Code Generation Part 1: exact helper contracts, state decision table,
  workflow input/permission map, tests and numbered implementation checklist.
- [ ] Code Generation Part 2: execute the approved checklist.
- [ ] Build and Test: focused delivery, packaging, docs and workflow lint gates.

Separate Application Design and Units Generation are skipped: one existing
delivery unit and no new application boundary. Separate Functional Design,
NFR Requirements, NFR Design and Infrastructure Design are skipped: reuse the
approved delivery designs and safeguards, with the bounded decision table and
workflow handoff amendment captured in Code Generation planning. No new cloud
resources or publisher identities are proposed. If implementation planning
reveals a need to change those boundaries, stop and revise this plan.
Operations remains a placeholder; no live publication is part of local testing.

## Workflow visualization

Text sequence (canonical): completed workspace/requirements and current
planning; approval; code-generation planning and approval; implementation;
build/test; handoff. The conditional design stages listed above are skipped
because their existing artifacts are reused.

## Change and verification sequence

1. Establish current baseline and regression cases in
   `tests/packaging/test_delivery_workflows.py` and
   `tests/packaging/test_release_manifest.py`; preserve current protections.
2. Implement release classification and original-completion verification in
   `scripts/` using existing release and index helpers; test before wiring
   workflow mutations. Exact new helper boundaries are selected in Part 1.
3. Update `.github/workflows/ci-main.yml` and `ci-publish.yml`, including
   artifact handoff, strict success dependencies, scoped permissions and locks.
4. Extend generated release-state/provenance invariants and executable fake
   boundary tests. Cover new, complete, draft, partial, prerelease, conflicting,
   absent-proof, expired-proof, duplicate and interrupted cases. A new build
   with the same version must never replace the original release identity.
5. Update `docs/developer/releasing.rst` and affected documentation contracts;
   reconcile obsolete explicit-only language without erasing historical audit.
6. Run deterministic Hypothesis tests, complete packaging tests, warning-strict
   documentation smoke, Ruff, changed-file formatting, actionlint with
   ShellCheck and diff checks. Run broader tests if application scope changes.

Do not remove validation lanes, numerical cases or coverage thresholds.
Hosted permission/OIDC and scheduler behavior must be reported separately from
local test success. No commit, push, PR, merge, settings mutation or live release
dispatch is included without the corresponding user authorization.

## Success criteria and handoff

New committed versions publish automatically only after complete main
validation, using its tested bytes. Completed versions visibly no-op after
verification; ambiguous/partial/conflicting versions block with recovery
guidance. Manual recovery and all prior review fixes continue to work.

Legacy 0.6.10 is not silently recovered or overwritten. Record any missing
original evidence and explain the need for separately authorized recovery or
a reviewed new version. Do not claim this planning change has activated releases.

## Extension compliance

PBT-01: identify publish authorization, exact identity and no-op invariants in
Part 1. PBT-02: retain manifest round trips. PBT-03: generated classification and
provenance invariants. PBT-04: repeated completed checks have no mutations.
PBT-05: compare state decisions to an independent truth table. PBT-06: retain
and extend interrupted/duplicate recovery sequence coverage. PBT-07: reuse
bounded domain strategies. PBT-08: fixed seed, shrinking and CI inclusion.
PBT-09: retain Hypothesis. PBT-10: concrete observed failure regressions beside
properties. All are compliant as planning obligations; implementation proof is
pending. Security and Resiliency extensions remain disabled and skipped.

## Review

A) Approve & Continue to Code Generation planning.

B) Request Changes to this plan.

C) Add one or more skipped stages (specify which).

X) Other (describe the requested direction).

[Answer]: A (user replied `Approved`).

## Compatibility amendment at Code Generation planning

PyPI's official troubleshooting documentation states that reusable workflows
cannot currently be used as the workflow in a Trusted Publisher:
https://docs.pypi.org/trusted-publishers/troubleshooting/

Replace the proposed reusable publishing call with a standalone `workflow_run`
completion listener in the existing publishing filename. Validate repository,
push event, main branch, exact workflow path, successful full proof, source SHA
and artifact identity before privileged jobs. Use source-run SHA rather than
the listener's default-branch SHA for release provenance. Main validation can
remain read-only. The separate publisher cannot be cancelled by main's
validation concurrency. Retain its shared non-cancelling publication lock.
This amendment awaits approval with the code-generation plan; no code changed.
