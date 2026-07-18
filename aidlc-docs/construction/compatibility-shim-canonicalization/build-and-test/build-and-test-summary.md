# Build and Test Summary

## Results

- Complete suite: 2,089 passed and 4 optional tests skipped in 227.04 seconds.
- Fixed-seed non-solver suite: 2,086 passed, 3 skipped, and 4 solver tests
  deselected in 161.70 seconds.
- Slow-HPR tutorial profile: passed in 215.37 seconds.
- HEN solver tutorial profile: passed in 164.37 seconds.
- Ruff lint and formatting: passed across 454 Python files.
- Warning-free Sphinx/RTD HTML build: passed for 54 source pages.
- Wheel and source distribution: built successfully.
- Isolated installed-wheel smoke: passed outside the source checkout.
- Patch whitespace and stale-symbol checks: passed after canonical CSV line-ending
  generation.

## Scope Notes

Seven orphan transition pages were deleted. All 18 tutorial notebooks and the
operation-level RTD coverage manifest were regenerated. Intentional unit, fluid,
value-like, optional-dependency, and solver normalization remains unchanged.

## Extension Compliance

- Security Baseline: N/A, disabled.
- Resiliency Baseline: N/A, disabled.
- Partial Property-Based Testing: compliant with fixed seed `20260715`, shrinking,
  centralized strategies, and CI pytest execution.
