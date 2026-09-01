# GitHub Actions Artifact Smoke Correction Plan

## Scope and authorization

The user explicitly authorized correction, verification, and commit of the
installed-wheel smoke failure diagnosed in CI Develop run 33441220938 and PR
Validation run 33441224711.

## Testable property

- PBT-03 invariant: a package imported from the checkout's source package is
  rejected, while a package imported from any valid environment outside that
  source directory is accepted, including a virtual environment nested inside
  the checkout.
- PBT-10 complementarity: retain a concrete regression for the exact
  checkout-local `.venv/site-packages/OpenPinch` path reported by GitHub.

## Execution plan

- [x] Step 1: Retrieve both failed runs and identify the shared root cause and
  non-blocking artifact-action runtime warning.
- [x] Step 2: Add failing example and property regressions for source-import
  classification, including the exact checkout-local virtual-environment case.
- [x] Step 3: Narrow checkout-import detection to the actual source package
  directory without weakening installed-wheel validation.
- [x] Step 4: Update upload-artifact and download-artifact to current immutable
  Node.js 24-compatible commit pins.
- [x] Step 5: Run the focused path, packaging, workflow YAML, and real built-wheel
  smoke gates.
- [x] Step 6: Run the complete configured-solver suite, Ruff, RTD documentation,
  distribution build, and patch hygiene checks.
- [x] Step 7: Review scope, update AI-DLC evidence and checkboxes, and commit the
  verified correction to `develop`.

## PBT compliance target

- PBT-03: applicable and blocking.
- PBT-07: applicable; generated paths remain structured beneath generated
  absolute checkout and environment roots.
- PBT-08: applicable and blocking; Hypothesis shrinking and fixed seed remain
  enabled.
- PBT-10: applicable and blocking.
- PBT-02, PBT-04, PBT-05, and PBT-06: N/A for this pure path classifier.

## Completion evidence

- RED: collection failed because the new classifier did not exist; the action-
  version regression then failed against the obsolete pins.
- GREEN: 27 focused packaging and workflow tests pass after formatting.
- Integration: a fresh wheel installed into an isolated checkout-local virtual
  environment passes the complete artifact smoke workflow.
- Regression: 2,518 tests pass with 4 expected skips under the complete
  configured-solver environment and fixed Hypothesis seed.
- Quality: Ruff lint and changed-file format checks, YAML parsing, patch hygiene,
  warning-strict 54-source RTD build, and fresh wheel/source builds pass.
- PBT-03, PBT-07, PBT-08, and PBT-10 are compliant. PBT-02, PBT-04, PBT-05,
  and PBT-06 are N/A because the correction is a pure path-boundary classifier.
