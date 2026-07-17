# Residual Compatibility Shim Removal Implementation Summary

## Scope

This unit removes four audited behavioural compatibility mechanisms from
unsupported internal APIs while preserving
`OpenPinch.main.pinch_analysis_service`, canonical input values, numerical
behaviour, and current-version serialization.

## Implemented Changes

- HPR cascade processing calls each grid, process-cascade, and utility-cascade
  helper once with an explicit `period_idx`, including `None`. Internal
  `TypeError` exceptions propagate unchanged.
- `HPRParsedState` and `HPRBackendResult` use attributes and Pydantic
  serialization only; dictionary-emulation methods were removed.
- HPR optimisation accepts only `dual_annealing`, `cmaes`, `bo`, and
  `rbf_surrogate`, corresponding enum values, or `None` for the existing
  default. The alias table and public normalisation helper were removed.
- `StreamCollection` no longer repairs missing fields from old pickle state.
  Current-version round trips retain period context and numeric operation, and
  replace unpicklable callable sorting with deterministic supply-temperature
  sorting.

## Contract Impact

The protected main service is unchanged. Unsupported direct callers lose HPR
mapping access, historical optimiser spellings, obsolete helper-signature
retries, and old `StreamCollection` pickle-state repair. No alias, migration
loader, deprecation warning, or version bump is included.

## Verification Status

Focused behavioural tests pass for all four implementation units. The combined
focused gate passed 277 tests, affected cycle coverage passed 37 tests, and the
complete non-solver suite passed 2,063 tests with four solver-tagged tests
deselected. Ruff lint and formatting, warning-free Sphinx, isolated wheel/sdist
builds, stale-surface scans, and patch hygiene also pass. Complete evidence is
recorded in `aidlc-docs/construction/build-and-test/build-and-test-summary.md`.

## Extension Compliance

- Security: N/A because the extension is disabled.
- Resiliency: N/A because the extension is disabled.
- Partial Property-Based Testing: N/A because no numerical algorithm changed;
  direct behavioural regressions cover every changed contract.

## Post-Implementation Import and Type Corrections

The generated-code review subsequently corrected two unresolved type-only
configuration imports, removed a `Zone` self-import, fixed total-site
`period_idx` forwarding, and added explicit positive-integer validation for
crossflow row counts. The final focused gate passed 96 tests, all 301 package
modules imported, targeted Pylint and Ruff checks passed, and the complete
non-solver suite passed 2,067 tests with four solver-tagged tests deselected.
