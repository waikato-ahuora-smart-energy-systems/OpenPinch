# Build and Test Summary

## Results

- Focused parent-owned record and API suite: 11 passed.
- Focused problem and multiperiod orchestration suite: 105 passed.
- Complete non-solver suite: 2,011 passed, 4 solver tests deselected.
- Coverage: 98% total; extracted targeting execution reached 100%.
- Solver suite: 3 passed, 1 intentional skip.
- Ruff lint: passed.
- Ruff formatting for all changed Python files: passed.
- Sphinx documentation with warnings treated as errors: passed with external
  intersphinx access.
- Notebook JSON parsing: passed.
- Wheel and source distribution: passed.
- Public API, retired-path, cold-import, and `git diff --check` gates: passed.

The first sandboxed full run reported only environmental Chrome and remote
intersphinx access failures. The same gates passed with the required external
permissions; no product-code workaround was introduced.

## Extension Compliance

- Security: N/A (disabled).
- Resiliency: N/A (disabled).
- Property-Based Testing Partial: compliant for all enabled rules applicable
  to this refactor.

