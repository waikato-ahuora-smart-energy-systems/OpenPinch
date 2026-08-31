# Utility Placement Supply-Order Interleaving Implementation Summary

## Outcome

Generated utility physics now follows optimized supply temperature rather than
the isothermal-then-sensible vector schema. Isothermal and sensible levels may
interleave, while same-kind hot representatives retain stable ordinal identity.
Independent candidate verification sorts hot supplies and derived cold supplies
and enforces the physical adjacent gap.

The default minimum separation is `0.01 delta_degC`, equal to the default
isothermal temperature difference. Residual `HU` and `CU` supplies are explicit
exact-target inputs at 50 K beyond the context-wide maximum and minimum real
process temperatures. They are not optimization coordinates, target after named
utilities, and retain the physical entropy plus squared fallback-duty penalty
when active.

The final sensible structured start now uses its feasible lower supply and span
edge. This makes the close-profile, zero-fallback branch available before the
black-box search. On the notebook Process case, default CMA-ES returns:

- objective: `0.03524514018747493 kW/K`;
- fallback penalty: `0.0`;
- active isothermal level: `173.03 -> 173.02 degC`;
- active sensible level: `173.02 -> 64.2624 degC`.

The standard Process GCC ends at `173.00 degC`; the Utility GCC begins at
`173.03 degC`, replacing the former visible top gap while preserving the
non-crossing constraint and exact ordinary-retarget replay.

## TDD Evidence

- RED proved the old 1 K default, declaration-order rejection, inaccessible
  lower-entropy isothermal point, missing 50 K fallback margin, and inadequate
  zero-fallback structured starts.
- GREEN covers default equality, cross-kind example and Hypothesis invariants,
  optimizer transform round trips, exact Process entropy improvement, fallback
  supply temperatures, and the notebook-derived deterministic start.
- All focused placement and application suites pass.
- The temporary canonical notebook executes both Process GCC and Site TSP
  workflows without assertions or source mutation; Process fallback is zero,
  evidence matches retargeting, and the baseline remains unchanged.
- The complete solver-enabled repository gate has 2,451 passing tests, 4
  expected skips, and 4 deliberate checks excluded for the user's unexecuted,
  locally modified notebook. Two sandbox-only Chrome/process-pool failures pass
  when rerun with the required local permissions.

## Extension Compliance

- Property-Based Testing: compliant; generated cross-kind physical ordering is
  exercised over arbitrary domain-valid sensible supplies.
- Security Baseline: disabled, not applicable.
- Resiliency Baseline: disabled, not applicable.
