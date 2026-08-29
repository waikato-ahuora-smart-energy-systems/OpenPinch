# Build and Test Summary

## Build Status

- **Build tool**: Hatchling through the locked uv environment.
- **Build status**: Success.
- **Artifacts**: Fresh `openpinch-0.5.4.tar.gz` and
  `openpinch-0.5.4-py3-none-any.whl` in an isolated temporary directory.
- **Archive contents**: Both archives contain the cap-aware targeting,
  placement contracts/service, public integration, runtime stream metadata,
  and executed notebook 19.
- **Installed-wheel smoke**: Success from an isolated import target outside
  the source checkout.

## Test Execution Summary

### Maximum-Duty TDD and Integration

- RED produced 12 expected failures and 21 unaffected passes before the API,
  runtime metadata, targeting cap, and penalty kernel existed.
- The expanded focused GREEN gate passed 81 tests covering generated and
  inferred names, scalar/unit/period inputs, zero and omitted limits,
  independent side caps, batch forwarding, named-before-fallback allocation,
  common cross-period fallback temperatures, typed evidence, and detached-case
  replay.
- The broad utility-placement, direct-targeting, input-contract, and notebook
  gate passed 298 tests with 3 guarded skips.
- The numerical property suite proves squared-penalty identities, scale
  invariance, monotonic bounded ranking, raw period-weight aggregation, cap
  compliance, and exact duty coverage.

### Complete Repository Suite

- **Collected**: 2,442 tests.
- **Passed**: 2,438.
- **Skipped**: 4 expected environment/profile-specific tests.
- **Failed**: 0.
- **Solver coverage**: solver-marked tests used the configured local IPOPT,
  CBC, Bonmin, Couenne, APOPT, MojoPSE, and AMPL function libraries.
- **Duration**: 276.71 seconds for the final reviewed-tree run.

### Quality, Documentation, and Performance

- Ruff: pass.
- `git diff --check`: pass.
- Sphinx warnings-as-errors build: 54 source pages, pass.
- Representative performance gates: 5 tests, pass.
- Notebook generator/source contract: pass.
- Notebook 19 project-kernel execution: pass with refreshed outputs.
- A standalone Python 3.14 `coverage run` attempt was unavailable because
  NumPy raised its duplicate-module import guard under instrumentation. The
  same 25 penalty tests pass normally, including Hypothesis and every explicit
  valid/invalid branch; the empty coverage file was removed.

## Installed-Wheel Notebook Evidence

The isolated wheel imported from the temporary target and executed notebook
19's setup, capped Process/GCC workflow, and uncapped Site/TSP workflow. The
Process result reports `0.05984815679388933 kW/K` entropy and
`0.33603087814600113` dimensionless fallback penalty. Four named hot utilities
carry independent 20 kW caps, residual `HU` remains visible, and ordinary
direct targeting respects the stored limits. The uncapped Site result remains
`0.6863313374698734 kW/K`. Both standard figures are constructed and the
baseline case remains unchanged.

## Delivered Scope

Utility placement remains thermodynamic-only and Python-API-only. The optional
`maximum_duties` mapping is keyed by final generated or inferred names and
accepts scalar, explicit-unit, or period-resolved values. Omitted entries are
unbounded and zero disables one named utility. Reserved `HU` and `CU` are not
placement options; they cover only residual duty after capped named utilities
are exhausted. Their balanced-composite entropy remains physical, while
`g[p] = (Q_HU[p]/Q_heat[p])^2 + (Q_CU[p]/Q_cool[p])^2` is reported and
raw-weighted across all periods as a separate dimensionless ranking term.

The returned detached case retains caps and required fallback definitions, so
normal Process GCC and Site TSP targeting/plots show the constrained utility
system. Monetary placement, cogeneration economics, CLI integration, new plot
APIs, dependencies, and infrastructure remain excluded.

## Extension Compliance

- **Property-Based Testing**: compliant; all applicable example, generated,
  scaling, ordering, determinism, and conservation properties pass.
- **Security Baseline**: disabled; N/A.
- **Resiliency Baseline**: disabled; N/A.

## Overall Status

- **Build**: Success.
- **All required tests**: Pass.
- **Operations**: N/A; no deployment or publication was requested.

## Utility Targeting Profile Non-Crossing Refresh

The shared sensible-duty maximizer now retains the tightest interior Process
GCC breakpoint limit. RED captured the notebook-derived 97 degC crossing and a
shrunk hot/cold property counterexample. The final focused gates pass 67 shared
targeting tests, 248 analytical/placement tests, 7 application exact-replay
workflows, and 20 notebook/documentation checks. The complete solver-enabled
suite passes 2,450 tests with 4 expected skips in 526.67 seconds. Detailed
evidence is recorded in
`aidlc-docs/construction/utility-targeting-profile-non-crossing/build-and-test/build-and-test-summary.md`.

## Utility Placement Uncapped Notebook Refresh

Notebook 19 and its generator no longer pass `maximum_duties` to the Process or
Site utility-placement workflows. The deliberately small `search_options`
remain so the tutorial is executable, while the optional maximum-duty API and
its specialist tests remain available outside the notebook. RTD presents the
uncapped workflow first and documents capacity limits separately.

The regenerated notebook executes successfully. Its Process result has zero
fallback penalty, exact optimizer-to-retarget duty replay, and a feasible
standard Utility GCC that does not cross the Process GCC. The focused Ruff,
notebook contract/execution, documentation consistency, and warnings-as-errors
RTD gate passes 21 tests. The complete solver-enabled repository suite passes
2,450 tests with 4 expected skips in 501.53 seconds.

## Utility Placement CMA-ES Default Refresh

`UtilityPlacementOptions.method` now defaults to `cmaes`; exact per-call
selection of `dual_annealing`, `cmaes`, `bo`, and `rbf_surrogate` remains
unchanged. HPR selection and the reusable optimizer service's generic default
are unchanged. Requirements, functional design, RTD, default-contract tests,
coordinator evidence, and the public detached-case integration test are aligned.

RED produced the two expected default-selection failures. GREEN passes 27
focused contract/coordinator/application tests, 19 RTD and notebook-source
checks, and the broader solver-enabled suite with 2,446 passed, 4 expected
skips, and 4 deliberate notebook deselections in 622.87 seconds. The deselected
checks depend on the user's locally modified notebook 19, which has unexecuted
cells and differs from the generator. A temporary canonical notebook 19 was
generated and executed successfully under the CMA-ES default without modifying
the user's file; its Process result retained zero fallback penalty and exact
optimizer-to-retarget duty replay.
