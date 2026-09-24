# Unit 3 Code Generation Summary

Unit: Direct Process-MVR Hardening and Integrated Proof

Status: Complete and approved under the user's authorization through completion.

## Outcome

Direct process-MVR now rejects non-finite or unsupported compression requests
before stage iteration, converts expected required-state failures into bounded
contextual errors, preserves unexpected defects, and records every permitted
optional fallback. Packaged CoolProp tutorials now distinguish typed physical
infeasibility from service defects and assert their successful primary paths.

## Implemented Changes

- Added finite closed bounds of 200 degC per requested stage lift and 20 per
  requested stage pressure ratio, plus early efficiency validation.
- Added `DirectGasMVRStageError` with stream, period, stage, fluid, stable reason
  code, bounded public text, and original exception chaining.
- Added frozen `DirectGasMVRFallbackDiagnostic` values and attached immutable
  diagnostic tuples to direct-MVR stage results.
- Named and exposed the only two optional fallbacks: `dry_stage` for unavailable
  liquid-injection saturation data and `reduced_profile` for unavailable
  optional profile saturation breakpoints.
- Narrowed fallback handling to expected property `ValueError` failures so
  unexpected runtime defects propagate unchanged.
- Hardened notebooks 09 and 11 to catch typed HPR physical failures and assert
  successful primary CoolProp and direct process-MVR paths.
- Corrected scalar 1-by-1 vapour-compression topology identity so accepted
  CoolProp targets remain compatible with performance-map generation.

## Traceability

| Obligation | Evidence |
|---|---|
| U3-BR-001 through U3-BR-004 | finite lift/ratio validation, closed-bound examples, and seeded above-bound properties |
| U3-BR-005 through U3-BR-008 | contextual chained stage error and atomic failure regressions |
| U3-BR-009 through U3-BR-012 | closed immutable diagnostics, dry-stage/reduced-profile metamorphic checks, and fatal propagation regression |
| U3-BR-013 | real packaged process-MVR execution and broad heat-pump regression gate |
| U3-BR-014 | generated notebook source contracts plus real notebook 09 and 11 execution |
| U3-P1 through U3-P12 | deterministic properties, boundary examples, model assertions, public workflows, and notebook execution |

## Verification

- Expanded HPR, contracts, application, and notebook gate: `975 passed, 3 skipped`.
- Direct process-MVR, VC+MVR, and application reliability gate: `85 passed`.
- Scalar/cascade topology and map compatibility gate: `87 passed`.
- Packaged notebook 09: successful CoolProp target and three-point performance map.
- Packaged notebook 11: successful direct process-MVR plus serial/parallel parity.
- Ruff, selected-file formatting, compilation, generator idempotence, and patch
  whitespace: clean.
- Dependencies and infrastructure: unchanged.

## Extension Compliance

- Property-Based Testing: compliant. Boundary generators use fixed repository
  seeds and retain normal shrinking behavior.
- Security and Resiliency: disabled and not applicable to this unit.

## Remaining Work

No unit implementation remains. The workflow proceeds to aggregate Build and
Test for source distributions, wheels, the full repository suite, installed
artifact smoke, and final documentation.
