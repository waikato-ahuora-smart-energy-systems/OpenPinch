# Indirect Profile Precision Build and Test Summary

## Results

| Gate | Result | Status |
|---|---|---|
| Regression-first precision tests | 2 expected failures before fix; 2 passed after fix | Pass |
| Expanded targeting and downstream matrix | 158 passed | Pass |
| Complete fixed-seed non-solver suite | 2,197 passed; 3 skipped; 4 deselected in 173.04 s | Pass |
| Notebook 2 base-profile execution | 1 passed; 9 unrelated profiles deselected | Pass |
| Packaging and notebook resources | 84 passed; 3 optional-profile skips | Pass |
| Repository Ruff lint and format | All checks passed; 461 files formatted | Pass |
| Fresh wheel and source distribution | OpenPinch 0.5.3 wheel and sdist built | Pass |
| Isolated wheel precision smoke | Exact tables, close interval, and duties verified | Pass |
| Patch hygiene | Clean | Pass |

## Completion

All requirements are satisfied. Graph presentation remains rounded to four
decimal places without mutating the underlying cascade. Indirect profile
reconstruction retains full temperature and enthalpy precision. Security and
Resiliency extensions are disabled and N/A; partial PBT requirements are
compliant. Operations is N/A because no deployment change was requested.
