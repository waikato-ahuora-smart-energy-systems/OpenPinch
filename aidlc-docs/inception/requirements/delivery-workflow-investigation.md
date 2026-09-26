# Testing, merge, and release workflow investigation

## Scope and status

2026-09-26: Follow-up to the completed HPR/MVR reliability and test-runtime work.
Brownfield Python/uv library; existing architecture artifacts remain sufficient
for this delivery-only investigation. No application change is proposed.
Requirements clarification is pending. No workflows or remote settings changed.

## Confirmed failure

- Main run 36132464168 at commit 2e7435668f0123d61b7c4098a22336c9db495a19
  passed ordinary tests, solver tests, TESPy tests, performance tests, docs,
  optional installations, build, and cross-platform artifact smoke tests.
- Job 108068264873 failed with: `Published release is absent; expected a
  complete exact release.`
- Upload job 108068159223 received HTTP 200 for both 0.6.10 distributions.
- The verifier makes six visibility checks separated by ten seconds: about
  fifty seconds of waiting, despite the ten-minute job timeout.
- At investigation time the TestPyPI JSON endpoint exposes both files with
  hashes identical to the upload log:
  wheel `270deba7482c00836e73a42b355c46c71d05d5dc5993fd7e5f38f38d2249c98a`;
  sdist `be5f6b770f94cec605ceb2d4bea9e55a17bda89ead3d91f96413ee86a545f8ff`.
- This supports delayed index visibility as the cause of the observed failure;
  the exact propagation duration and upstream caching cause are unknown.
- The verifier correctly stopped promotion. Do not remove exact hash checks.

Sources: [failed verification](https://github.com/waikato-ahuora-smart-energy-systems/OpenPinch/actions/runs/36132464168/job/108068264873),
[upload log](https://github.com/waikato-ahuora-smart-energy-systems/OpenPinch/actions/runs/36132464168/job/108068159223),
[TestPyPI release metadata](https://test.pypi.org/pypi/OpenPinch/0.6.10/json).

## Seven improvement areas

1. **Release visibility and diagnostics.** Replace the short fixed polling
   window with a configurable bounded deadline, report each observation and
   endpoint, and preserve immediate failure for mismatched or unexpected files.
   Test delayed visibility, partial uploads, exhausted retries, HTTP failures,
   and hash conflicts with fake time rather than real sleeps.
2. **Enforced merge gate.** Public main rules require `test`, not `pr-gate`.
   Require a uniquely named aggregate gate that fails on failed, cancelled,
   missing, or unjustifiably skipped mandatory lanes. Preserve current review
   protections. Coordinate the remote required-check migration separately;
   changing YAML alone does not change branch protection.
3. **Immutable version preparation.** PR validation currently includes a
   source-branch-writing version job and mutable head-ref checkouts. Move
   version preparation before validation and validate the exact candidate
   commit/tree. Never push a version change while certifying another revision.
4. **Shared validation definitions.** Develop, PR, and publishing workflows
   repeat dependency setup and test commands. Use reusable validation owners
   with explicit profiles and one test-selection contract. Retain all benchmark
   cases, the 95-percent coverage gate, and specialized integration lanes.
5. **Complete reuse evidence.** The develop-reuse required-job set does not
   explicitly include performance-tests although PR reuse skips that lane.
   Overall run success protects against a failed job, but not an omitted or
   skipped lane. Derive required evidence from one lane manifest and test
   missing, skipped, stale, cancelled, and differing-tree cases.
6. **Recoverable release orchestration.** Keep immutable artifact identity and
   exact-byte promotion. Separate validation from publishing and support
   resuming failed verification without rebuilding, moving tags, or replacing
   published files. Currently GitHub publication precedes production PyPI
   verification; define completion only after all required destinations pass.
7. **Workflow behavior tests and useful reports.** Test the event/job/gate
   matrix, artifact handoff and recovery paths, not only helper functions or
   YAML text. Add workflow linting and retained test/duration summaries. Avoid
   rerunning expensive validation for description-only edits. Assess merge
   queue support only if adopted; it is not currently configured.

## Proposed acceptance criteria

- A failed mandatory lane blocks merging through the actual required check.
- The validated revision is the merged candidate; versioning cannot invalidate
  successful evidence silently.
- Every selected test belongs to a documented lane; no HPR/MVR convergence or
  feasibility assertion is weakened, and no flaky test is hidden by retries.
- Delayed index visibility succeeds within a documented deadline; corrupt or
  unexpected artifacts never promote.
- Release recovery uses the same verified bytes and cannot overwrite a tag or
  package version. Recovery is tested without publishing test packages.
- Shared definitions remove command drift; reuse requires complete evidence.
- Local focused tests, workflow validation, and independent hosted CI are
  distinguished in the completion report.

## Boundaries and extension compliance

No merge, package publication, tag modification, or repository-settings change
has been performed. Recovery of 0.6.10 is separate from implementing the
overhaul. Preserve develop/main unless the user requests a branching change.
User stories remain excluded by prior user direction.

Property-Based Testing remains enabled. PBT-01 through PBT-10 are N/A to this
read-only investigation; assess them during design and implementation of
release-state and evidence-validation helpers. Security and Resiliency
extensions remain disabled under the recorded configuration.
