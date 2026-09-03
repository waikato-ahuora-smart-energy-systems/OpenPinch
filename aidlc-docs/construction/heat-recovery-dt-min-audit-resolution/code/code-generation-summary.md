# Heat-Recovery `dt_min` Audit Resolution Code Generation Summary

## Outcome

All six approved audit findings are resolved without changing the public
method signatures or ordinary targeting behavior.

- Recovery input normalization now accepts only supported finite real scalar
  forms and rejects implicit Boolean, string, sequence, array, and arbitrary
  mapping coercions.
- The inverse solver uses a private precision-preserving cascade path with
  exact shifted-temperature levels and strict interval overlap. The public
  ordinary cascade retains its existing canonical grid and tolerances.
- Every supplied `Zone` contributes only an address, resolved against the
  current problem or batch case.
- Exact zero alone selects the zero-recovery path. Positive micro-duty requests
  use strict achieved-at-least-requested feasibility, including positive
  thermodynamic-limit plateaus.
- `HeatRecoveryDtMinResult` validates finite numeric types, dimensions,
  non-negativity, limit ordering, feasibility, residual arithmetic, and status
  relationships in common units.
- Exact canonical all-period keys take precedence over the serialized
  `value`/`unit` scalar-map shape.

Final bisection brackets are re-evaluated and verified before return. The
internal bracket is narrowed to half the public `1e-6 delta_degC` tolerance so
the feasible endpoint remains within the published boundary accuracy.

## TDD evidence

The Red gate produced 44 expected failures with 46 existing controls passing.
The Green feature gate passes 104 tests. It includes analytical and packaged
examples, strict contract and application cases, workspace/batch isolation,
foreign-zone resolution, canonical `value`/`unit` period IDs, micro duties,
threshold/no-overlap boundaries, fail-closed bracket checks, and seeded
Hypothesis properties. Shrunk forward-oracle cases at `45.5` and `1.828125`
`delta_degC` are permanent explicit examples.

The precision-preserving Bleaching boundary is approximately
`58.34505012947355 delta_degC`; the solver returns the greatest feasible side
within `1e-6 delta_degC`.

## PBT extension compliance

- PBT-01 through PBT-05: reusable bounded strategies, independent invariants,
  shrinking, fixed seeds, and deterministic examples are present.
- PBT-07 through PBT-10: finite/unit-aware data, canonical ordering,
  sequential/parallel equivalence, JSON round trips, and permanent regressions
  are covered.
- PBT-06: N/A because the solver owns no persistent mutable state; generated
  snapshots still prove source non-mutation.
- Security and Resiliency extensions remain disabled for this workflow.

## User-facing artifacts

The dedicated RTD guide, fundamentals, problem/workspace APIs, contributor
service reference, and release notes document the strict shapes, local zone
semantics, mapping precedence, micro-duty behavior, result invariants, and
inverse-only precision path. Generated notebooks remain in sync; no notebook
source or output change was required for these contract corrections.
