# Remove Compatibility Facades Code Generation Plan

## Part 1 - Approved Planning

- [x] Inventory explicit compatibility-only modules and intentional API barrels.
- [x] Define the clean-break boundary and canonical concrete schema owners.
- [x] Obtain implementation approval from the user's explicit request.

## Part 2 - Generation

- [x] Remove synthesis compatibility modules and re-exporting package barrel.
- [x] Route lazy public exports directly to concrete schema modules.
- [x] Update implementation, tests, and documentation to canonical imports.
- [x] Remove legacy pickle/import compatibility assertions.
- [x] Add structural regressions proving retired paths stay absent.
- [x] Run focused synthesis, import, API, and packaging tests.
- [x] Run complete non-solver and solver tests, Ruff, Sphinx, notebooks,
  distributions, stale-path checks, and patch hygiene.
- [x] Update release notes, implementation evidence, state, audit, and all
  checkboxes.
