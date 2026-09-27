# Automatic release code-generation plan

## Status and scope

Part 1 complete; Part 2 and the compatibility amendment approved (`Approved`).
Single source of truth for implementation progress. Workspace:
`/Users/timothyw/Github_Local/OpenPinch`.

Implements acceptance requirements 1-10 in
`../../inception/requirements/automatic-release-requirements.md`.
User stories, application APIs, database schemas and new infrastructure are
not applicable. Existing delivery designs and safeguards remain the baseline.
No version bump, publication, commit, push, PR or settings mutation is included.

## Compatibility amendment

Keep `ci-publish.yml` standalone: PyPI currently does not support registering
a reusable workflow as the Trusted Publisher. Add `workflow_run` for completed
`CI Main` runs on main, retaining `workflow_dispatch`. GitHub delivers completed
events regardless of success, and the listener's SHA is the default-branch
SHA, not the triggering source SHA. Both distinctions require explicit guards.

Sources checked during planning:
- https://docs.pypi.org/trusted-publishers/troubleshooting/
- https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows
- https://docs.github.com/en/rest/actions/artifacts

Execute helper code from the trusted listener checkout. Treat triggering-run
data and downloaded manifests as data, never executable code or shell text.
Use the verified original source SHA for tag/provenance checks, never implicit
listener HEAD. Keep the production environment and publisher filename unchanged.

## Interfaces and decision table

Add `scripts/plan_release.py` as the read-only orchestration owner. Input is
the GitHub event JSON and authenticated API observations. Output is structured
validated data: action, version, source SHA/run/attempt, artifact ID/digest and
reason. Write only validated scalar fields to workflow outputs; diagnostics go
to stderr and the summary. Do not introduce another publishing implementation.

| Observation | Outcome | Mutation allowed |
|---|---|---|
| Successful same-repository main push, exact CI Main proof, new version | publish | Existing publisher only |
| Failed/cancelled run or unrelated event | ignore with reason | None |
| Claimed successful main run but invalid/missing proof | block | None |
| Verified original stable release, exact assets and both indexes complete | complete/no-op | None |
| Tag only, draft, prerelease, partial index, missing/expired proof, conflict | block with recovery guidance | None |
| Explicit manual resume with exact original identity | Existing recovery path | Only existing verified recovery actions |

The new-version path checks absence/conflicts before staging. The completed
path must not compare a later same-version build to the original release.
Resolve original run/attempt from the released manifest, recover its unique
retained artifact from API metadata, verify actual archive digest and existing
source proof, then compare release assets and both indexes against those bytes.
Reject missing, duplicate, expired or substituted artifacts. Version and
repository identities must agree throughout; original source must be trusted
main history. No fallback to unproven release assets when CI proof has expired.

## Workflow and permissions

1. `CI Main` remains read-only and builds once through full shared validation.
2. Publishing listener accepts successful same-repository main push evidence
   only; repeat critical checks using API data, not just event predicates.
3. Read-only request/planning and bundle jobs resolve and verify source data.
   Failed planning cannot be converted to a skipped successful publication.
4. Automatic mode skips the publisher's own build/validation and consumes the
   original bundle; manual new mode retains its existing validation path.
5. Stage, preflight, publish, verification and finalization depend explicitly
   on successful required predecessors and a publish decision. Review skipped
   validation propagation for both automatic and resume modes.
6. `contents: write` stays on stage/finalize only; `id-token: write` stays on
   index-upload jobs. Preserve read-only API permissions elsewhere. No new PAT,
   source writes, PR privileges or permissions on main validation.
7. Preserve the standalone publisher's `openpinch-release` non-cancelling lock;
   make decisions inside that lock. Independent main cancellation cannot stop
   a publisher that has started. Document pending-run queue limitations.

## Implementation checklist

### Step 1 - Baseline and regression contracts

- [x] Inspect git state and preserve all unrelated changes. Run affected tests
  before edits. Add tests in `tests/packaging/test_plan_release.py` and extend
  `test_delivery_workflows.py` for listener events, permission isolation and
  exact artifact handoff. Record expected pre-change failures.

### Step 2 - Extract reusable read-only proof checks

- [x] Refactor `scripts/release_manifest.py` to expose existing source-proof
  verification without changing its semantics. Keep CLI verify/stage/finalize
  behavior and both PR 102 corrections. Add regressions for direct and helper
  invocation in `test_release_manifest.py`.

### Step 3 - Resolve automatic source identity

- [x] Implement `scripts/plan_release.py` event/API validation and bounded
  artifact discovery. Require exact repository, main push, workflow path,
  successful full-profile evidence and unique original build artifact.
  Support retained build artifacts across failed-job retries without accepting
  stale failed validation or ambiguously choosing a rebuilt artifact.

### Step 4 - Classify release state without mutation

- [x] Implement new/completed/blocked decisions and original completion
  verification using existing manifest, download and index helpers. Reuse
  deadlines, strict schemas and hash checks. Verify original GitHub assets,
  tag/source and both index states. Produce explicit recovery diagnostics.

### Step 5 - Wire the existing publisher

- [x] Update `.github/workflows/ci-publish.yml` trigger, request outputs and
  conditions for automatic/new/resume paths. Derive version from verified
  source evidence rather than listener HEAD. Keep main read-only and current
  publishers pinned; change `ci-main.yml` only if contract wiring requires it.
  Audit every source environment variable and skipped-job dependency path.

### Step 6 - Generated and behavioral verification

- [x] Extend `tests/strategies/delivery.py` only where reusable strategies are
  needed. Test decisions against an independent state table with generated
  manifests and evidence; test duplicate checks, conflict introduction and
  interruption sequences using real helper logic with fake API boundaries.
  Explicit cases cover later same-version commits, wrong workflow/SHA/repo,
  API failure versus absence, expired/duplicate artifacts and legacy 0.6.10.

### Step 7 - Documentation and traceability

- [x] Update `docs/developer/releasing.rst`, affected developer docs and
  `tests/packaging/test_docs_consistency.py`. Distinguish automatic new releases,
  no-ops, blocked versions and manual recovery. Add implementation summary in
  `aidlc-docs/construction/automatic-release/code/implementation-summary.md`.
  Update active requirements/design references without rewriting historical
  approvals. Explicitly document the expiry consequence for completion proof.

### Step 8 - Quality gates and handoff

- [x] Run focused tests and all packaging tests with Hypothesis seed 20260926,
  warning-strict Sphinx smoke, Ruff, changed-file format checks, actionlint
  with ShellCheck, YAML parsing and `git diff --check`. Preserve all validation
  lanes and the 95-percent coverage gate; broader application tests are needed
  only if scope unexpectedly touches application behavior.
- [x] Update plan/state and build/test evidence; report hosted OIDC/scheduling
  verification as pending. Do not trigger a real release to test the workflow.

## PBT compliance mapping

PBT-01: authorization, immutability and completion invariants defined above.
PBT-02: existing manifest round trips retained in Step 2.
PBT-03: generated proof/classification invariants in Steps 3, 4 and 6.
PBT-04: duplicate completed checks are read-only and idempotent, Step 6.
PBT-05: independent decision-table oracle, Step 6.
PBT-06: interruption/duplicate/conflict sequences, Step 6.
PBT-07: bounded structured domain strategies, Step 6.
PBT-08: seed 20260926, shrinking enabled, ordinary CI selection, Step 8.
PBT-09: existing pytest/Hypothesis dependencies unchanged.
PBT-10: concrete regressions complement generated checks, Steps 1 and 6.
All planning obligations covered; implementation evidence remains pending.
Disabled Security and Resiliency extensions remain skipped.

## Approval

A) Approve & Begin Code Generation, including the standalone listener amendment.

B) Request Changes to the plan.

X) Other (describe the requested direction).

[Answer]: A (user replied `Approved`).
