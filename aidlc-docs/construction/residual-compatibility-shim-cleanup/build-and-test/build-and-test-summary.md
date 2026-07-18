# Build and Test Summary

## Build Status

- Build tool: Hatchling through the repository build helper and uv environment.
- Status: success.
- Artifacts: `openpinch-0.5.2-py3-none-any.whl` and `openpinch-0.5.2.tar.gz`.
- Isolated Python 3.14 wheel installation and smoke: passed outside the checkout.
- Warning-as-error Sphinx HTML build: passed across 53 source pages.

## Test Execution Summary

| Gate | Result | Status |
|---|---:|---|
| Initial focused baseline | 183 passed | Pass |
| Expanded affected suite | 275 passed | Pass |
| Complete fixed-seed non-solver suite | 2,108 passed, 3 skipped, 4 deselected | Pass |
| Real HEN solver profile | 3 passed, 1 skipped | Pass |
| Solver-regression artifact checks | 6 passed, 3 deselected | Pass |
| Repository Ruff lint | All checks passed | Pass |
| Repository Ruff format | 460 files formatted | Pass |
| Sphinx warnings-as-errors | 53 sources | Pass |
| Wheel and source distribution | 2 artifacts | Pass |
| Isolated installed-wheel smoke | Site-packages import | Pass |
| Stale-symbol and AST guards | No active findings | Pass |
| Patch whitespace | No findings | Pass |

## Verification Notes

The first real-solver profile produced the correct objective and design but failed an exact weighted-job assertion at 1,199 versus 1,210. An isolated rerun produced 1,177, proving the branch count was nondeterministic. The regression was corrected to require 95 through 100 successful ESM branches because ESM tasks are conditional on successful TDM parents. The complete real-solver profile then passed in 223.96 seconds.

## Extension Compliance

- Security Baseline: disabled; N/A.
- Resiliency Baseline: disabled; N/A. Existing Couenne resilience remains tested and unchanged.
- Partial Property-Based Testing: compliant for the enabled penalty and unit-group invariants with Hypothesis seed `20260715` and shrinking enabled.
- Serialization property coverage: unchanged and green in the complete non-solver suite.

## Overall Status

- Build: success.
- Required tests: pass.
- Documentation: pass.
- Distribution: pass.
- Performance testing: N/A with proportional timing evidence recorded.
- Operations: N/A because no deployment or production environment change was requested.
