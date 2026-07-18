# Compatibility Shim Canonicalization Code Summary

## Outcome

The approved clean break is implemented. Runtime streams expose only descriptive
engineering names while compact private storage and JSON fields remain stable.
Enum identity aliases, `Value.values`, summary format dispatch, graph-string
normalization, dynamic HEN result forwarding, view serialization, permissive
unknown-field handling, optional workspace bundle versions, and seven transition
pages are removed.

## Contract Evidence

- Root exports remain exactly `PinchProblem` and `PinchWorkspace`.
- `TargetInput` and nested inputs reject unknown fields; known plant profile data
  now has an explicit strict schema instead of relying on ignored extras.
- Workspace bundles require schema version `3`.
- `HeatExchangerNetworkDesignView` exposes only its explicit view operations and
  serializes through `design.result.model_dump(...)`.
- Fixed-seed property tests retain seed `20260715` and cover stream mutation,
  compact-wire round trips, period ordering, and serialized HEN transport.

## Extension Compliance

- Security Baseline: N/A; disabled and no security boundary changed.
- Resiliency Baseline: N/A; disabled and no deployed service changed.
- Property-Based Testing: compliant for enabled partial rules PBT-02, PBT-03,
  PBT-07, PBT-08, and PBT-09.
