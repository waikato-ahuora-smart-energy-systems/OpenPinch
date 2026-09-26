# Delivery workflow overhaul requirements

## Intent and approval

The user requested correction of a failing check and an overhaul of testing,
merging, and PR workflows. This is a high-risk delivery refactor, not a change
to the OpenPinch application API or thermodynamic algorithms.

The user selected A in `delivery-workflow-questions.md` and replied `approved`.
Explicit release initiation and the seven improvement areas are approved as
the requirements baseline. Develop/main remain the integration branches.
The investigation contains the evidence and distinguishes the observed
TestPyPI visibility failure from successful application tests.

## Functional requirements

| ID | Requirement | Acceptance evidence |
|---|---|---|
| D01 | Ordinary merges to main validate without publishing or bumping versions. Releases require deliberate initiation. | Event matrix tests distinguish validation, preparation, publication, and recovery. |
| D02 | Commit the selected version and lockfile changes before validating the release candidate. PR validation is read-only. | No source-branch-writing job in PR validation; candidate SHA/tree recorded. |
| D03 | Use shared lane definitions for develop, PR, and release validation. | All current benchmark cases and specialized lanes remain selected; drift/partition contracts pass. |
| D04 | Enforce a stable, uniquely named aggregate merge gate. | Mandatory lane failure, cancellation, absence, or unapproved skip fails the gate. Remote required-check configuration is verified separately. |
| D05 | Reuse results only with complete, current, exact-tree evidence. | Negative tests for differing trees, missing performance evidence, stale runs, skipped jobs, cancellation, and failed latest attempts. |
| D06 | Bound release polling by a documented configurable deadline and report progress. | Fake-clock tests cover delayed/partial visibility, transport errors, deadline exhaustion, and permanent mismatches. |
| D07 | Build and identify release artifacts once; promote and recover using the same exact bytes. | Artifact identity/digest, version, source, and per-file hash checks reject substitution and conflicting releases. |
| D08 | Report release completion only after required package-index verification and GitHub release finalization. | State-transition tests cover every failed boundary and safe resumption without rebuilding or overwriting. |
| D09 | Provide actionable test and release reports and a documented recovery procedure. | Retained failures/durations, source identities, polling diagnostics, and concrete recovery commands. |

## Non-functional requirements

- Preserve 95-percent ordinary branch coverage, HPR/MVR feasibility and
  convergence assertions, Hypothesis shrinking and reproducibility, optional
  dependency smoke tests, and cross-platform artifact verification.
- Do not hide flaky tests with blanket retries or decrease numerical quality
  assertions to improve CI time. Retry only classified transient delivery I/O.
- Eliminate unnecessary description-only validation runs. Measure pipeline
  wall time and lane durations against a recorded hosted baseline; do not
  promise a numerical improvement without measurement.
- Use bounded network operations, deterministic local workflow simulations,
  fake clocks, and no publishing side effects in automated regression tests.
- Preserve least-privilege permissions, trusted publishing, immutable tags,
  and current review protections. Untrusted PR content must not gain release
  credentials or publishing authority.
- A missing proof must fail closed or fall back to full validation; it must
  never silently certify a candidate.

## Scope boundaries

In scope: repository workflow YAML, delivery scripts, packaging/workflow tests,
developer documentation, and a staged required-check migration procedure.
Out of scope: user stories, application redesign, a branch-model migration,
merge-queue adoption, automatic merge, publishing a new release, or recovery
of the already staged 0.6.10 release without separate authorization.

Repository-settings activation is a separate handoff: deploy and observe the
new aggregate check first, then authorize and verify the required-check
change. Local YAML changes alone do not satisfy D04 operationally.

## Extension configuration

Property-Based Testing remains enabled. Design must identify applicable
release-state and proof-validation properties; implementation must complement
example regression tests with generated invariants. PBT-01 through PBT-10
are N/A to this requirements-only artifact, with stage applicability assessed
during construction. Security and Resiliency extensions remain disabled;
ordinary delivery security and recovery requirements above still apply.
