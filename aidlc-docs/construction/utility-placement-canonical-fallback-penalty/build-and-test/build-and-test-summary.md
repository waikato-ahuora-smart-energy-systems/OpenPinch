# Canonical Fallback Penalty Build and Test Summary

## Results

- RED: eight expected failures and 19 unaffected passes proved the former result
  was exactly one tenth of the canonical squared penalty.
- GREEN: 27 focused example, integration, and property checks passed.
- Broad gate: 283 tests passed before the user-modified Notebook 19 base-profile
  execution was manually stopped after 26 minutes; two known notebook
  inventory/generator assertions failed because that notebook differs from its
  generated source.
- Scoped notebook contract gate excluding the three user-notebook execution and
  regeneration checks: eight passed and three optional-profile tests skipped.
- Ruff: passed.
- `git diff --check`: passed.

## Assessment

The production penalty correction and all directly dependent scientific and
application tests pass. Notebook 19 and `.gitignore` were already modified by
the user and were not changed by this correction.

## Extension Compliance

Property-Based Testing rules PBT-03, PBT-05, PBT-07, PBT-08, PBT-09, and PBT-10
are compliant. PBT-02, PBT-04, and PBT-06 are N/A. Security and Resiliency
extensions remain disabled.
