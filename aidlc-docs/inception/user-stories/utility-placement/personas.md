# Utility Placement Personas

## Persona Selection

The approved methodology defines four personas. The process engineer owns the
primary journey; the integrator, numerical reviewer, and maintainer represent
distinct acceptance viewpoints rather than separate implementation units.

## P-01: Process Integration Engineer

### Profile

A process or energy engineer who understands pinch analysis, utility systems,
and plant operating periods. They use OpenPinch to identify utility
temperatures and duties that satisfy process demand with lower thermodynamic or
monetary cost.

### Goals

- Configure separate isothermal and sensible utility counts.
- Find a placement that covers all residual heating and cooling demand.
- Start with entropy minimisation and optionally evaluate monetary trade-offs.
- Understand why the selected placement is feasible and preferable.
- Use the same workflow for direct and Total Site studies.

### Behaviors and Needs

- Supplies plant stream, period, utility-family, price, and turbine assumptions.
- Compares the best placement with ordered alternatives.
- Reviews per-period duties and objective decompositions before recommending a
  utility-system change.
- Needs explicit units, scope, bounds, feasibility, and warning information.

### Pain Points and Constraints

- A low scalar objective is untrustworthy if demand is not fully covered.
- Utility levels can be physically meaningless when temperatures, spans, or
  ordering violate plant constraints.
- Total Site and multiperiod results are difficult to interpret without
  per-period evidence.
- Monetary conclusions are misleading when cogeneration credit is hidden.

### Success Signals

- Every period has complete heating and cooling coverage within tolerance.
- The best and alternative placements are explainable from their decomposed
  thermodynamic or monetary terms.
- Direct and Total Site calls use a consistent public workflow.
- Invalid or infeasible studies provide actionable diagnostics.

## P-02: Python Library Integrator

### Profile

A Python developer or technical analyst who embeds OpenPinch in notebooks,
internal study tools, automated case pipelines, or downstream reporting.

### Goals

- Invoke utility placement through the established `problem.target` surface.
- Construct valid typed templates and options without private imports.
- Serialize results and compare or report them downstream.
- Run ordered workspace batches while isolating case failures.
- Trust that an analysis call does not mutate the source study.

### Behaviors and Needs

- Reads API signatures, contracts, examples, and error types.
- Uses JSON round-trips and case batches in automated workflows.
- Overrides configuration with explicit method arguments when needed.
- Needs stable names, deterministic ordering, and backward compatibility.

### Pain Points and Constraints

- Hidden mutation can corrupt later comparisons or cached cases.
- Backend-private result objects cannot cross process or persistence boundaries.
- Generic exceptions make automated recovery unreliable.
- Package-root growth and new mandatory dependencies complicate adoption.

### Success Signals

- Public calls, specialist imports, result serialization, and batches are typed
  and documented.
- Repeated fixed-seed calls return equivalently ordered candidates.
- Existing root imports and analysis workflows remain unchanged.
- Failures are isolated and programmatically distinguishable.

## P-03: Numerical Assurance Reviewer

### Profile

A senior process-integration specialist, numerical methods reviewer, or
research engineer responsible for validating the equations, constraints,
oracles, tolerances, and reproducibility of optimisation results.

### Goals

- Verify entropy and exergy equations against analytical cases.
- Prove complete heating and cooling demand conservation in every period.
- Check variable encoding, bounds, ordering, penalties, and feasibility.
- Compare small optimisation results against a transparent reference oracle.
- Reproduce any failure from its seed and diagnostics.

### Behaviors and Needs

- Reviews absolute-temperature handling and near-isothermal limits.
- Inspects per-period and aggregate objective identities.
- Uses example-based and property-based evidence together.
- Challenges apparently good solutions with boundary and adversarial cases.

### Pain Points and Constraints

- Black-box minima can conceal infeasible or numerically unstable candidates.
- Entropy formulas are vulnerable to sign, unit, and cancellation errors.
- Weighted aggregates can hide a failed period.
- Non-deterministic test failures are expensive to diagnose.

### Success Signals

- Analytical and property-test invariants agree within named tolerances.
- Feasible candidates always outrank penalised infeasible candidates.
- Structured-grid or brute-force oracles confirm small solved cases.
- Seeds, evaluation limits, and minimal counterexamples are reproducible.

## P-04: OpenPinch Maintainer

### Profile

A package maintainer responsible for architecture boundaries, code quality,
tests, documentation, compatibility, and sustainable feature ownership.

### Goals

- Add the feature through existing application, analysis, contract, and
  optimisation owners.
- Keep numerical kernels pure and accessors orchestration-only.
- Implement every production slice with red-green-refactor TDD.
- Enforce all enabled PBT rules and preserve CI reproducibility.
- Avoid regressions, new mandatory dependencies, and package-root expansion.

### Behaviors and Needs

- Uses architecture tests, focused test modules, fixed Hypothesis seeds, Ruff,
  and the complete non-solver suite.
- Converts shrunk property-test failures into permanent regressions.
- Reviews typed diagnostics instead of broad exception suppression.
- Documents public behavior and concrete specialist imports.

### Pain Points and Constraints

- Cross-layer numerical logic can create ownership and import cycles.
- Large integration changes are risky without thin TDD slices.
- Duplicated test fixtures make scientific invariants drift.
- Solver or optimiser options can create unbounded or flaky execution.

### Success Signals

- All story, requirement, TDD, and PBT traceability is complete.
- Architecture, focused, regression, packaging, and fixed-seed gates pass.
- The feature has no blocking PBT finding.
- The public contract is maintainable and backward compatible.

## Persona-to-Story Map

| Persona | Primary stories | Supporting stories |
|---|---|---|
| P-01 Process Integration Engineer | UPO-01 through UPO-08 | UPO-10, UPO-11 |
| P-02 Python Library Integrator | UPO-01, UPO-02, UPO-05, UPO-07 through UPO-10 | UPO-11, UPO-12 |
| P-03 Numerical Assurance Reviewer | UPO-03 through UPO-06, UPO-08, UPO-11 | UPO-07, UPO-12 |
| P-04 OpenPinch Maintainer | UPO-09 through UPO-12 | UPO-01 through UPO-08 |

## Extension Compliance

- **PBT-01 through PBT-10**: N/A at User Stories. P-03 and P-04 retain the
  assurance needs that become blocking in designated later stages.
- **Security Baseline**: N/A; disabled.
- **Resiliency Baseline**: N/A; disabled.
