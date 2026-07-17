# Tutorial Suite Code Summary

## Outcome

The packaged tutorial series now contains eighteen focused process-engineer
notebooks generated from one reviewed source and governed by the canonical
`docs/_data/tutorial-coverage.csv` manifest. The series uses only package-root
workflow imports and demonstrates the explicit prepare, execute, observe
lifecycle across core targeting, multiperiod studies, HPR, cogeneration,
energy transfer, HEN design, plots, reports, and exports.

## Implementation

- Added `scripts/generate_tutorial_notebooks.py` and regenerated the exact
  eighteen-notebook package inventory without stored outputs.
- Added stable notebook cell identifiers and execution-profile metadata.
- Replaced stale API substring assertions with manifest parity, nbformat,
  AST import, compilation, operation-coverage, and base-profile execution
  contracts.
- Aligned `OpenPinch.resources` metadata with the manifest inventory.
- Hardened total-site refresh orchestration so missing direct targets are
  prepared before indirect targeting.
- Corrected multiperiod cogeneration to use the explicit period index when a
  target does not carry a period-name lookup.

## Verification

- All eighteen notebooks pass warning-as-error nbformat validation.
- Extracted code from all eighteen notebooks passes Ruff.
- The ten base-profile notebooks execute cell-by-cell from a clean temporary
  directory.
- Tutorial and service-orchestration tests pass: 51 tests.
- Ruff lint and format checks pass for all changed Unit 4 Python sources.

## Extension Compliance

- Security: disabled; N/A because no enabled security rule applies.
- Resiliency: disabled; N/A because no enabled resiliency rule applies.
- Partial PBT: N/A for generated notebook documents; deterministic manifest
  and executable contract tests provide the applicable drift protection.
