# Unit 2 Code Generation Summary

## Outcome

Unit 2 provides a detached, immutable, solver-neutral, thermodynamic-only
utility-placement service using balanced-composite entropy generation.

## Production ownership

- Extended the shared contracts with explicit optimizer and temperature-policy options.
- Added immutable context, allocation, thermodynamic, penalty, evaluation,
  optimisation, and service modules under
  `OpenPinch/analysis/utility_placement/`.
- Reused existing utility targeting, optimiser, unit conversion, and Unit 1
  model owners without private backend imports.
- Prepared each utility-independent process load profile once per period and
  target zone. Candidate replay deep-copies the prepared problem tables,
  inserts shifted and real utility endpoints with the existing interval engine,
  and continues through the existing utility-target and balanced-composite
  methods.

## TDD evidence

- Integrated correction-focused suite: 227 passed with 3 guarded optional-profile skips.
- PBT: fixed-seed generated numerical examples plus pickle/process-state,
  logarithmic heat-load scaling, and default-penalty permutation properties.
- Oracle gates: hand-calculable `CP * ln(T_out / T_in)` and `Q/T` entropy
  cases plus closer-temperature ranking passed.
- Real backend: one fixed-seed, bounded dual-annealing regression passed.
- Performance: 40-level by 100-period entropy batch p95 below 50 ms; cold replay below 1 second; exact memo hit below 1 ms.
- Branch coverage: the corrected thermodynamic and penalty kernels are both at
  100 percent; the established Unit 2 ownership gate remains above threshold.
- Ruff: passed.
- Existing utility-targeting, turbine, optimisation, architecture, API-boundary, and package-entrypoint regressions passed.
- Prepared-versus-fresh oracle coverage compares complete shifted and real
  problem tables, utility duties, target totals, Process snapshots, and Total
  Site snapshots across explicit and generated utility temperature sets.

## Distribution evidence

An isolated no-network build produced `openpinch-0.5.4.tar.gz` and
`openpinch-0.5.4-py3-none-any.whl`. Both contain all 12 specialist package files
and the placement contract. A target-installed wheel imported the specialist
service and constructed the default thermodynamic request successfully.

## Numerical and failure contracts

- Every candidate replays every period and must cover hot and cold residual duty.
- Thermodynamic placement minimizes physical entropy generation from
  candidate-local real-temperature balanced composite curves. Sensible
  intervals use `CP * ln(T_out / T_in)` and isothermal intervals use signed
  `Q/T`; materially unbalanced curves are rejected.
- Placement retains the deterministic cross-product of five supply-distribution
  and five sensible-span starts. Count-generated paired models prepend
  profile-spanning, low-gap, and endpoint starts, then retain only candidates
  that pass independent pair, bound, Kelvin, and ordering verification.
- Application-owned physical bounds cover the temperature intervals where the
  residual hot and cold profiles change; sensible span bounds cover the full
  target temperature range. Generated-envelope properties verify containment,
  ordering, finiteness, and feasible deterministic starts.
- Thermodynamic results retain utility and process terms, total physical
  entropy generation, ambient temperature, and exergy destruction.
- Positive allocated duty on generated fallback `HU` or `CU` utilities receives
  a deterministic infeasible penalty before objective ranking.
- Monetary and placement-specific cogeneration contracts and evaluators are
  intentionally absent.
- Feasible and infeasible solver scalars occupy disjoint intervals even at binary64 saturation.
- Exact process-local memoization, bounded diagnostic representatives, parent canonical replay, deterministic physical ranking, and typed exhaustion are implemented.

## Compatibility and handoff

Root exports and legacy result schemas are unchanged; no dependency or CLI was
added. Unit 3 can consume `optimise_utility_placement`, the frozen context, and
the result contract through the specialist package.

PBT-01 through PBT-10 are compliant. Security and Resiliency are disabled and
N/A for this in-process numerical unit.
