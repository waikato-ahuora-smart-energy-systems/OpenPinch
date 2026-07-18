# Notebook Presentation Build and Test Plan

- [x] Confirm Code Generation and all profile-specific notebook gates are complete.
- [x] Run the complete fixed-seed non-solver repository suite.
- [x] Run repository-wide Ruff lint and formatting checks.
- [x] Build Sphinx documentation from a clean temporary destination with
  warnings treated as errors.
- [x] Build the OpenPinch wheel and source distribution in a clean temporary
  destination.
- [x] Install the built wheel outside the checkout and run workflow, resource,
  root-API, retired-package, notebook-resource, and CLI smoke checks.
- [x] Revalidate generated notebook determinism, source-only output state,
  optional profile evidence, and tutorial coverage.
- [x] Review the complete patch for whitespace, temporary outputs, dependency
  changes, duplicate files, private imports, and unrelated modifications.
- [x] Generate all required Build and Test instruction and summary artifacts.
- [x] Update AI-DLC state and audit records and close Operations as N/A.

## Extension Compliance Plan

- Security Baseline: N/A; disabled.
- Resiliency Baseline: N/A; disabled.
- PBT-08: run the repository suite with seed `20260715` and shrinking enabled.
