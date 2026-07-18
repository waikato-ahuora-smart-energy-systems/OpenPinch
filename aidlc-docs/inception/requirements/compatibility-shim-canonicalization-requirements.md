# Compatibility Shim Canonicalization Requirements

## Intent Analysis

- **Request type**: Breaking public-API refactor and contract hardening.
- **Scope**: System-wide Python, schema, tutorial, documentation, and test changes.
- **Complexity**: High because runtime stream naming and enum identities affect most
  analysis packages.
- **Compatibility policy**: Immediate pre-1.0 clean break with no aliases,
  deprecation warnings, migration loaders, or transition pages.
- **Persona**: Process engineers using `PinchProblem` and `PinchWorkspace`.

## Functional Requirements

1. Summary APIs accept only `detailed`, `include_periods`, and
   `include_weighted_average`; numeric comparison remains private.
2. Runtime `Stream` construction, properties, setters, and string-based helpers use
   descriptive names only. Compact private storage and established JSON keys remain.
3. `set_dt_cont_multiplier()` remains the deliberate problem/workspace engineering
   shorthand.
4. `Value.period_values` is the sole public period-array accessor; serialized data
   continues to use the `values` key.
5. Full enum class names are canonical. All module, local, and synthesis enum aliases
   are removed.
6. `HeatExchangerNetworkDesignView` is a closed explicit view. Serialization uses
   `design.result.model_dump(...)`.
7. External input schemas reject unknown fields, and workspace bundles require an
   explicit current schema version.
8. Named plot methods and canonical `GraphType`/`StreamLoc` values replace all graph
   and composite-location string aliases.
9. Seven legacy RTD transition pages and their preservation tests are deleted.
10. The root continues to export exactly `PinchProblem` and `PinchWorkspace`.

## Deliberately Retained Behavior

- Compact JSON field names and configuration keys.
- Unit-group overrides, case-insensitive fluid phases, `vapor`/`vapour`, Pint and
  value-like coercion, optional dependency guards, and solver fallbacks.
- The segmented-stream parent-axis solver placeholder required by current equations.
- Both `selected_network` and ranked `network(...)` on the explicit design view.

## Non-Functional Requirements

- Preserve numerical results, result ordering, serialization mappings, and optional
  dependency boundaries.
- Maintain root-only tutorial imports and 100 percent tutorial coverage of the live
  process-engineer surface.
- Preserve unrelated working-tree changes.
- Use Hypothesis with seed `20260715`, shrinking, and reusable domain strategies for
  applicable round-trip and invariant properties.

## Acceptance Criteria

- Static and runtime checks prove every retired alias, forwarding surface, format
  selector, and transition page is absent.
- Compact input JSON constructs descriptive runtime streams and round-trips unchanged.
- Unknown fields and missing/unsupported workspace versions fail validation.
- All focused, non-solver, HPR/HEN profile, Ruff, Sphinx, packaging, and isolated-wheel
  gates pass.

## Extension Compliance

- **Security Baseline**: Disabled; N/A.
- **Resiliency Baseline**: Disabled; N/A.
- **Partial Property-Based Testing**: PBT-02, PBT-03, PBT-07, PBT-08, and PBT-09 apply
  to input round trips, stream mutations, and public-contract invariants.

