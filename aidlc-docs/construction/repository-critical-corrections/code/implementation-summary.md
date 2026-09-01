# Repository Critical Corrections Implementation Summary

## Delivered corrections

- Problem source replacement now snapshots all input, prepared-zone, result,
  period-result, and source-identity fields before applying a replacement. Any
  validation or preparation exception restores the prior coherent state.
- `set_dt_cont_multiplier(...)` now updates the canonical zone tree and rebuilds
  transactionally, so problem JSON, workspace bundles, scenarios, and restored
  cases retain the multiplier.
- Returned utility-placement cases always serialize selected-period maximum
  duties with explicit identities, including one-period and equal-value limits.
  Candidate replay remains a current-period scalar operation.
- Utility load profiles explicitly require matching one-dimensional temperature
  and heat vectors plus finite, tolerance-non-negative heat values.
- Segmented utility targeting scales authoritative child duties in their last
  non-zero proportions. Zero assignment and later retargeting are supported,
  including sub-tolerance child duties, without permitting ordinary invalid
  segment edits.
- Reusable optimizer results are filtered for bounds and SciPy-compatible
  equality, inequality, linear, and nonlinear constraints before ranking. A run
  with no finite feasible candidate fails explicitly.

## TDD evidence

The RED gate reproduced all six defects with eleven failures. Fixed examples
and generated properties now cover atomic rollback, canonical multiplier and
period identity round trips, malformed load profiles, segmented-duty
conservation and idempotence, and optimizer feasibility. The complete suite
passes with shrinking enabled.

## Scope review

The change retains existing public workflow names, utility-placement decision
coordinates, notebook 19, plotting behavior, monetary-optimization exclusion,
and CLI exclusion. RTD changes document only the corrected atomic lifecycle and
segmented-utility targeting behavior.

## Extension compliance

- Property-Based Testing: compliant; generated cap, segmented-duty, and
  optimizer-constraint properties pass with shrinking enabled.
- Security Baseline: disabled and N/A; no new external boundary was added.
- Resiliency Baseline: disabled and N/A; no infrastructure or deployment
  behavior changed.
