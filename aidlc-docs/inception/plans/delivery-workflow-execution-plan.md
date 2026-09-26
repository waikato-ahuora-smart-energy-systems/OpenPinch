# Delivery workflow execution plan

## Scope, dependencies, and risk

One cohesive delivery unit spans `.github/workflows`, `scripts`, packaging
tests, and developer documentation. It does not change application APIs,
domain models, or runtime dependencies. Existing reverse-engineering artifacts
plus the fresh delivery investigation provide sufficient context.

Risk is high because an incorrect gate can admit unvalidated changes and a
publishing error can create immutable remote state. Local rollback is a
normal reviewed revert; remote releases and tags are not rollback targets.
Testing complexity is high at event, permission, artifact, and recovery
boundaries, despite modest application impact.

Dependencies: shared lane policy drives validation and proof checking;
immutable candidate/version preparation feeds artifact creation; validated
artifact identity feeds publishing and recovery; the observed aggregate
check enables the final repository-settings migration.

## Adaptive stage sequence

The numbered sequence is the text workflow visualization; no diagram parser
is required.

1. Workspace detection and requirements: completed.
2. Workflow planning: approved by the user with `go]`.
3. Functional Design: execute to define event/state transitions, candidate
   identity, proof rules, and testable properties.
4. NFR Requirements and NFR Design: execute at focused depth for bounded
   polling, least privilege, observability, runtime cost, and recovery.
5. Infrastructure Design: execute at focused depth for GitHub job permissions,
   publishing environments, triggers, concurrency, and check migration.
6. Code Generation: prepare an explicit checkbox plan, obtain approval, then
   implement the dependency-ordered slices below.
7. Build and Test: execute local verification and report hosted evidence
   separately; provide operational rollout and recovery instructions.

Skip a new Reverse Engineering stage: the application architecture is
unchanged and delivery differences were investigated directly. Skip User
Stories per user direction. Skip separate Application Design and Units
Generation: this is one delivery unit inside existing workflow/script/test
owners; its state model belongs in Functional Design. Operations remains a
placeholder, not permission to publish or change remote settings.

## Dependency-ordered implementation slices

- [x] Define lane/event policy and executable regression contracts first.
- [x] Correct visibility polling and diagnostics with fake-time regressions.
- [x] Consolidate validation and complete evidence reuse, retaining coverage
  and all specialized lanes; add the stable fail-closed aggregate gate.
- [x] Separate version preparation from read-only PR validation and permit
  ordinary main merges without a new version or automatic publication.
- [x] Implement explicit release initiation and same-artifact recovery with
  exact identity checks and finalization after verified publication.
- [x] Add workflow linting, behavior tests, retained test/duration reports,
  developer guidance, and the 0.6.10 recovery runbook without executing it.
- [x] Run focused contracts, workflow validation, then the complete affected
  validation profiles; compare test selection and coverage with the baseline.
- [x] Prepare a staged remote migration checklist and record what remains
  unverified until independent hosted checks run.

These slices are implemented and locally verified under the approved code plan.
No speculative wall-clock estimate is assigned; hosted solver and external
index timings dominate and must be measured.

## Verification and rollout

Local tests must prove the event matrix, fail-closed gates, complete reuse
evidence, immutable candidates/artifacts, deadline behavior, and idempotent
recovery. Use Hypothesis for identified properties with deterministic seeds,
shrinking, realistic release/job inputs, and complementary example tests.
Never publish real packages as a regression test.

Keep existing required-check compatibility during migration. After the new
aggregate check is deployed and observed on a PR, request specific authority
to update branch protection while preserving review restrictions. Verify the
resulting remote rules. Do not claim end-to-end enforcement before that step.
Any workflow filename/environment changes affecting trusted publishing must
have a documented configuration migration before activation.

Success means all D01-D09 requirements have evidence, unchanged numerical
quality/coverage, no unowned test lanes, and a clear distinction between local
completion and pending hosted/configuration verification.

## Planning progress and extension compliance

- [x] Load prior findings, user decision, and dependency context.
- [x] Assess scope, impacts, dependencies, risks, and conditional stages.
- [x] Record implementation order, verification, and rollout boundaries.
- [x] User approves execution plan or requests changes.

PBT-01 through PBT-10 are N/A to workflow planning itself; property assessment
and generated tests are explicitly planned for construction. Security and
Resiliency extensions are skipped because disabled. Plain Markdown content
and table structure were checked before creation; no embedded executable
code or complex diagram requires parsing.
