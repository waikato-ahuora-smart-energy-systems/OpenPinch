# Heat-Recovery DT_MIN Audit Resolution Business Logic Model

## Processing flow

1. Classify `heat_recovery` before general unit coercion. Accept only an
   approved scalar shape and recursively reject Boolean or nonscalar
   magnitudes.
2. Resolve a string zone normally. For a `Zone` object, retain only its address
   and resolve that address against the current problem's root.
3. Resolve the canonical period and convert the request to canonical kW.
4. Deep-copy the selected process streams and assign half the trial global
   `dt_min` to each side with a contribution multiplier of one.
5. Evaluate recovery through an inverse-only precision mode of the existing
   vectorized cascade. Precision mode uses exact finite shifted temperature
   levels and strict interval-overlap predicates. Ordinary targeting retains
   its current canonical grid and overlap tolerance.
6. Establish recovery at zero and at the finite no-overlap upper bound. Reject
   above-limit requests and invalid brackets.
7. For exactly zero recovery, bisect the transition from positive to zero and
   return the smallest zero-recovery boundary. For a positive request, bisect
   the feasible-to-infeasible transition and return the greatest feasible
   boundary. Positive requests no larger than the absolute recovery tolerance
   use a strict achieved-at-least-requested predicate.
8. Post-verify the returned side of the final bracket and the opposite side.
   The bracket width must be no greater than `1e-6 delta_degC`, its recovery
   predicates must differ, and all values must remain finite and bounded.
9. Convert result quantities to configured output units and construct the
   frozen result contract, which revalidates dimensions and cross-field
   relationships.

## Precision-mode isolation

The cascade implementation will expose precision behavior only through a
private analysis path. Shared public contributors and ordinary targeting keep
their existing signatures and defaults. The precise path reuses the same
problem-table columns and cascade equations, changing only temperature-level
canonicalization and interval-overlap tolerances that caused boundary aliasing.

## Failure flow

- Unsupported scalar shapes fail before unit conversion.
- Invalid or foreign zone addresses fail before stream selection.
- Negative, non-finite, or above-limit requests fail before bisection.
- Non-finite recovery, non-monotone bracket predicates, invalid endpoints, and
  failed post-verification raise a closed numerical error.
- Contract dimensional or relational inconsistencies fail Pydantic validation.
