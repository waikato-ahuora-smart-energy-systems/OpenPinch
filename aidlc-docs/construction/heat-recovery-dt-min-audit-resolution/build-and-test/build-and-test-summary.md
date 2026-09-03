# Build and Test Summary

## Result

All feature, regression, architecture, notebook, RTD, coverage, distribution,
and installed-wheel gates are green.

| Gate | Result |
|---|---|
| Feature analysis/application/contract/PBT | 104 passed |
| Ordinary cascade and problem-table regression | 122 passed |
| Architecture, units, and tutorial ownership | 71 passed |
| Packaged notebooks | 23 passed, 3 expected skips |
| Warning-strict Sphinx | 55 sources, succeeded |
| Full configured suite | 2,641 passed, 4 expected skips |
| CI non-solver branch coverage | 2,638 passed, 3 expected skips, 4 deselected; 96% |
| Ruff lint | passed |
| Changed-file Ruff formatting | passed |
| Generated-notebook drift | no changes |
| Wheel and source distribution | built successfully |
| Isolated Python 3.14 wheel smoke | passed |
| Patch hygiene | passed |

The repository-wide format command reports 14 already-committed files outside
this correction. They were not reformatted because they are unrelated to the
approved feature scope; every changed Python file is formatted.

## Numerical evidence

The inverse-only precise cascade preserves the approved `1e-6 delta_degC`
boundary contract while ordinary targeting retains its established grid. The
packaged Bleaching threshold's exact discontinuity is approximately
`58.34505012947355 delta_degC`; the solver returns its greatest feasible
bracket side within tolerance. Exact zero, positive micro duties, zero-limit
problems, interior and maximum plateaus, and unstable final evaluations all
have permanent regressions.

## Extension compliance

Property-Based Testing is compliant for all applicable PBT-01 through PBT-05
and PBT-07 through PBT-10 rules. PBT-06 is N/A because no persistent mutable
state is introduced. Disabled Security and Resiliency extensions were not
enforced.
