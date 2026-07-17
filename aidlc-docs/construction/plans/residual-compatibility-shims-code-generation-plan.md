# Residual Compatibility Shims Code Generation Plan

## Approved Scope

Remove the four audited behavioural compatibility mechanisms while preserving
`OpenPinch.main.pinch_analysis_service`, canonical configuration values,
numerical behaviour, and current-version serialization. This is an intentional
clean break for unsupported internal APIs, with no aliases, migration loaders,
or compatibility warnings.

## Execution Checklist

- [x] Step 1: Verify the four findings, affected owners, tests, documentation,
  protected contract, and dependency boundaries against the current workspace.
- [x] Step 2: Record this detailed Code Generation plan and the user's explicit
  implementation approval in the AI-DLC state and audit trail.
- [x] Step 3: Remove obsolete HPR helper-signature retries and replace their
  compatibility test with period-forwarding and single-call error propagation
  regressions.
- [x] Step 4: Make `HPRParsedState` and `HPRBackendResult` attribute-only,
  migrate behavioural tests, and add an architecture regression for the absent
  dictionary-emulation surface.
- [x] Step 5: Restrict HPR optimisation method resolution to exact canonical
  identifiers and enum values; remove the public normalisation helper and update
  all internal and test inputs.
- [x] Step 6: Remove legacy `StreamCollection` pickle-state repair while
  preserving and testing complete current-version round trips, callable-sort
  fallback, and restored-instance independence.
- [x] Step 7: Update HPR reference documentation, release notes, implementation
  evidence, and stale-surface searches.
- [x] Step 8: Run focused HPR contract, targeting, optimisation, configuration,
  stream-collection, architecture, and protected-main tests.
- [x] Step 9: Run the complete non-solver suite, affected available HPR tests,
  Ruff lint and format checks, warning-free Sphinx, stale-path checks, and
  `git diff --check`.
- [x] Step 10: Finalize AI-DLC implementation summary, Build and Test evidence,
  state, audit, extension compliance, and generated-code review handoff.

## Validation Policy

- Security extension: disabled, therefore N/A.
- Resiliency extension: disabled, therefore N/A.
- Partial Property-Based Testing: N/A because this cleanup changes no numerical
  algorithm and adds direct behavioural regressions for each changed contract.
- Content validation: this plan uses plain Markdown only, with no Mermaid or
  ASCII diagrams requiring separate syntax validation.
