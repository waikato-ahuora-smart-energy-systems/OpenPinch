# GitHub CI HEN Solver-Isolation Code Generation Plan

## Plan Authority

This checklist is the single source of truth for the GitHub CI HEN
solver-isolation Code Generation stage. Execute the generation steps in order
and mark each checkbox complete in the same interaction as the work.

## Unit Context

- **Unit**: HEN design owner-boundary test isolation.
- **Workspace**: `/Users/timothyw/Github_Local/OpenPinch`.
- **Project type**: Brownfield Python library.
- **Approved requirements**:
  `aidlc-docs/inception/requirements/github-ci-hen-solver-isolation-requirements.md`.
- **Approved workflow**:
  `aidlc-docs/inception/plans/github-ci-hen-solver-isolation-execution-plan.md`.
- **User stories**: N/A; skipped as an isolated internal CI test fix.
- **Production dependencies**: None changed.
- **Existing test dependency**: `_use_fake_default_executor` installs
  `FakeSynthesisExecutor` at all HEN workflow construction sites.
- **Expected contract**: Invalid option objects remain rejected, valid
  call-local HEN configuration produces a design view, and stored configuration
  remains unchanged.
- **Database, API, frontend, deployment, and infrastructure layers**: N/A.

## Exact File Scope

- Modify in place:
  `tests/analysis/heat_exchanger_networks/test_design_workflow.py`.
- Create:
  `aidlc-docs/construction/github-ci-hen-solver-isolation/code/code-generation-summary.md`.
- Do not modify production code, dependency manifests, lockfiles, fixtures,
  pytest markers, or GitHub Actions workflows.

## Requirements Traceability

- **FR-CI-HEN-01 / FR-CI-HEN-04**: Step 1 installs the existing fake executor
  for the affected test.
- **FR-CI-HEN-02**: Step 1 preserves both invalid-input assertions unchanged.
- **FR-CI-HEN-03**: Step 1 preserves the successful manifest and configuration
  restoration assertions unchanged.
- **NFR-CI-HEN-01**: Step 1 removes external-solver execution from the test.
- **NFR-CI-HEN-02 / NFR-CI-HEN-03**: The exact file scope prohibits production
  and CI workflow changes.
- **NFR-CI-HEN-04**: The subsequent Build and Test stage runs the exact node and
  complete containing module under the CI marker and seed.

## Part 1 Planning Status

- [x] Read the approved requirements and workflow plan.
- [x] Read the brownfield code structure and confirm the workspace root.
- [x] Inspect the target test, fake executor, and isolation helper.
- [x] Confirm the target test file has no pre-existing local diff.
- [x] Identify dependencies, preserved assertions, and excluded layers.
- [x] Prepare and content-validate this executable checklist.
- [x] Obtain explicit approval for the complete Code Generation plan.

## Part 2 Generation Steps

### Step 1 - Isolate the owner-boundary test

- [x] Modify the existing test signature to accept pytest's `monkeypatch`
  fixture.
- [x] Call `_use_fake_default_executor(monkeypatch)` before constructing the
  public example problem.
- [x] Preserve the two invalid-options assertions, successful manifest
  assertion, and configuration-restoration assertion without semantic changes.
- [x] Do not add a `solver` marker or alter solver configuration.

### Step 2 - Review the scoped generated change

- [x] Confirm the test uses the same helper pattern as adjacent public HEN
  workflow tests.
- [x] Confirm no production, dependency, fixture, marker, or workflow file was
  changed by this unit.
- [x] Run whitespace and syntax-oriented patch checks on the scoped diff.
- [x] Confirm no duplicate or alternate test file was created.

### Step 3 - Record Code Generation output

- [x] Create
  `aidlc-docs/construction/github-ci-hen-solver-isolation/code/code-generation-summary.md`.
- [x] Record the modified file, preserved contracts, requirements traceability,
  and solver-isolation rationale.
- [x] State that executable tests belong to the subsequent Build and Test stage.

### Step 4 - Close Code Generation

- [x] Mark every completed generation checkbox in this plan.
- [x] Update the focused progress section in `aidlc-state.md`.
- [x] Append Code Generation completion evidence to `audit.md`.
- [x] Present the standardized generated-code review gate.

## Subsequent Build and Test Scope

After generated-code approval, Build and Test will run:

1. The exact GitHub Actions failure.
2. The complete `test_design_workflow.py` module with
   `--hypothesis-seed=20260715 -m "not solver"`.
3. Ruff lint and format checks for the modified test file.
4. Patch-hygiene and scoped-diff verification.

## Property-Based Testing Compliance

- **PBT-02**: N/A; no logical inverse or round trip changes.
- **PBT-03**: N/A; no general data transformation or business invariant is
  added.
- **PBT-07**: N/A; no generated input or strategy is added.
- **PBT-08**: Compliant in plan; the existing CI fixed seed remains in the
  Build and Test command.
- **PBT-09**: Compliant; Hypothesis remains selected and unchanged.

No blocking Partial PBT finding exists.

## Other Extension Compliance

- **Security Baseline**: Disabled; skipped.
- **Resiliency Baseline**: Disabled; skipped.
