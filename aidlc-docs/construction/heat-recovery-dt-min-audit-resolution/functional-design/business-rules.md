# Heat-Recovery DT_MIN Audit Resolution Business Rules

## Input rules

- Plain numeric inputs include Python and NumPy real scalar numbers plus finite
  standard-library decimal or fraction scalars, excluding complex numbers and
  every Boolean representation.
- OpenPinch `Value` and Pint quantities must contain exactly one magnitude.
- Serialized scalar mappings contain exactly `value` and `unit`; the magnitude
  is a supported non-Boolean numeric scalar and the unit is non-empty text.
- Lists, tuples, arrays, strings, bytes, arbitrary mappings, and scalar-like
  objects outside the approved representations are invalid.
- Numerical contributor `dt_min` values obey the same numeric predicate.
  `period_idx` is a non-Boolean integer and is range-checked by the stream
  collection.

## Numerical rules

- Recovery is evaluated on detached process streams without utilities or
  configured heterogeneous contributions.
- Exact finite shifted temperature levels are deduplicated only by exact
  equality for inverse evaluation.
- A stream segment participates in an interval only when it strictly overlaps
  that interval. No fixed temperature epsilon is subtracted from this test.
- The thermodynamic limit is recovery at global `dt_min = 0`.
- A request greater than the exact calculated limit remains invalid.
- `zero_recovery_boundary` requires an exactly zero request.
- Positive requests at or below `1e-6 kW` require `achieved >= requested`
  without the general absolute feasibility allowance.
- Other positive requests may use the documented absolute and relative
  recovery comparison tolerance.
- The final feasible/infeasible temperature bracket is at most
  `1e-6 delta_degC` wide and is explicitly verified before a result is emitted.

## Zone and period rules

- A `Zone` is a selector, not a transferable stream container. Only its address
  crosses the application boundary.
- Every case resolves that address within its own root and configuration.
- Exact canonical period-key matching takes precedence over recognizing a
  scalar value-with-unit mapping.
- Scalar mapping recognition occurs only for the exact `value` and `unit`
  shape after canonical-key matching fails.

## Contract rules

- `dt_min` is non-negative and convertible to `delta_degC`.
- Requested, achieved, and limit values are non-negative and convertible to
  kW. Residual is signed and convertible to kW.
- Comparisons occur after conversion to canonical units.
- Requested and achieved recovery cannot exceed the thermodynamic limit beyond
  tolerance.
- Achieved recovery satisfies the applicable positive-request predicate.
- Residual equals achieved minus requested within recovery tolerance.
- Status-specific predicates match the numerical termination condition.
- JSON numerical tokens remain valid; numeric strings and Boolean tokens do
  not.

## Error compatibility

Existing exception categories and complete above-limit context are retained.
New validation errors identify the invalid field or input shape without
exposing unrelated internal state.
