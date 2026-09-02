# Heat-Recovery Approach Documentation Amendment Summary

## Delivered teaching workflow

Generated notebook 02 now demonstrates a selected-period inverse recovery
request with explicit kW input, canonical period selection, JSON-ready result
fields, and an observation proving the ordinary cached target is preserved.
Notebook 06 continues to own scalar broadcast and exact period-mapping examples.

Read the Docs now includes a dedicated task guide and consistent discovery from
the top-level index, getting started, capability and workflow maps,
fundamentals, problem and workspace APIs, contributor service/reference pages,
the notebook catalog, and release notes. The guide covers supported scopes,
units, statuses, thermodynamic and zero-recovery boundaries, plateau selection,
all-period and case-batch calls, validation failures, non-mutation, and the HRAT
versus exchanger EMAT distinction.

## TDD and verification evidence

- Red notebook and RTD contract tests failed before the teaching surfaces were
  expanded and passed after implementation.
- The complete focused documentation gate passed 46 tests with 3 expected
  optional-profile skips.
- Notebook 02 executed successfully and the generator drift guard passed.
- Warning-strict Sphinx completed through the documentation build test.
- Ruff lint and formatting checks passed for every edited Python file.
- Git patch-hygiene checks passed.

## Extension compliance

PBT-01 through PBT-10 are N/A because the amendment changes tutorial and
documentation presentation only. Security and Resiliency remain disabled by
project configuration.
