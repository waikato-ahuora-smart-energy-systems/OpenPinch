# PinchProblem Targeting Business Rules

- A target accessor is not callable; method selection is always explicit.
- Configuration may supply omitted numerical assumptions but never selects a
  core workflow.
- `direct_heat_integration()` runs direct recovery only.
- `indirect_heat_integration()` and `total_site_heat_integration()` share one
  focused Total Site implementation and signature.
- `all_heat_integration()` runs direct prerequisites before aggregate Total Site
  analysis in one zone-tree traversal.
- Exactly one effective HPR load form is allowed: `load_fraction`, `load_duty`,
  or `period_loads`.
- HPR method names select the physical model; no cycle or placement string is
  accepted on the normal public path.
- Utility placement and cascade topology booleans are accepted only where both
  values are physically supported.
- Multiperiod Brayton targeting is not exposed until the backend supports it.
- Isentropic cogeneration requires an explicit efficiency in its valid range.
- A supplied `base_target` must belong to the same problem execution state.
- Removed targeting selector keys are rejected by persistent option updates.
- Explicit invocation values do not mutate stored configuration.
- Ordered period results follow canonical `period_ids` exactly.
- Observation and export operations never solve implicitly.
- Retired names and compatibility aliases are absent.
