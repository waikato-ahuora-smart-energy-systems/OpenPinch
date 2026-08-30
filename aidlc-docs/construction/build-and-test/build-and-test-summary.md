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
the normalized hot/cold fallback residual vector is passed to canonical
`g_ineq_penalty(..., form=PenaltyForm.SQUARE)` with default `rho=10`; its result
is reported and raw-weighted across all periods as a separate dimensionless
ranking term.

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

## Utility Placement Supply-Order Interleaving Refresh

Generated isothermal and sensible levels may now interleave by optimized supply
temperature. Stable same-kind hot identities remain structurally ordered, while
independent verification sorts physical hot and derived cold supplies and uses
the `0.01 delta_degC` default isothermal temperature difference as the adjacent
minimum gap. Residual `HU` and `CU` are explicit exact-target utilities with
supplies 50 K beyond the context-wide real process-temperature extremes.

The final sensible structured seed reaches the close-profile, zero-fallback
branch before CMA-ES. The executed canonical notebook Process workflow returns
`0.03524514018747493 kW/K`, zero fallback, an active `173.03 -> 173.02 degC`
isothermal level, and a sensible level beginning at `173.02 degC`. The standard
GCC ends at `173.00 degC`; exact optimizer-to-retarget duty replay holds. The
Site TSP workflow also executes and the baseline remains unchanged.

All focused placement, property, application, RTD, generator, and canonical
notebook gates pass. The broad solver-enabled gate records 2,449 immediate
passes, 4 expected skips, and 4 deliberate local-notebook deselections. Its two
sandbox-denied Chrome/process-pool checks pass on their permitted rerun, giving
2,451 passing tests for the complete applicable gate. The user's `.gitignore`
and locally modified notebook 19 remain outside this amendment.

## Utility Placement Balanced-Composite Duty-Conservation Refresh

The uncapped notebook 19 Site workflow exposed a balanced allocation whose
thermodynamic reconstruction lost approximately `2.625 kW`. The active cold
isothermal utility spanned `22.878892` to `22.888892 degC`; a rounded process
breakpoint lay about `0.00000795 K` inside that span. The shared problem-table
kernel treats intervals narrower than `0.00001 K` as inactive. That temperature
noise rule is normally harmless, but the near-isothermal utility's very large
heat-capacity flow made the discarded interval carry material duty.

Balanced-composite thermodynamics now constructs hot and cold utility profiles
analytically from exact allocated duty and clipped temperature fractions. Every
profile therefore reaches its complete endpoint duty on any subdivided grid.
The candidate-failure containment remains as defense, but the formerly failing
candidate is now balanced and evaluated normally rather than rejected.

The exact current notebook cells execute without rewriting the user's file.
Process and Site finish in 355.17 seconds with objectives
`0.03524514018747493 kW/K` and `0.3452285101943513 kW/K`; Process fallback is
zero, no invalid-balanced-composite diagnostic occurs, and the baseline remains
unchanged. The focused utility-placement and shared numerical gate passes 240
tests, including deterministic and property duty-conservation coverage. The
solver-enabled broad gate passes 2,456 tests with 4 expected skips and 3
deliberate deselections for execution-count/generator assertions affected only
by the user's locally modified notebook. Ruff and patch hygiene pass.

## Utility Placement Prepared Target Replay Refresh

The exact-target adapter now constructs and preprocesses one detached
`PinchProblem`, then caches its utility-independent shifted and real direct load
profiles once per period and target zone. Each candidate receives deep-copied
problem tables with candidate utility temperature endpoints inserted through
the existing interval engine. Existing utility targeting, balanced-composite,
direct targeting, and Total Site aggregation remain the calculation owners.

The fresh ordinary target is the independent oracle. Explicit edge cases and
fixed-seed generated utility sets match complete shifted/real problem tables,
utility names and duties, target totals, Process snapshots, and Total Site
snapshots. Construction-count, repeated-replay, multiperiod, pickle, and source-
isolation regressions pass.

With three warmups and ten measurements, median Process candidate replay
improves from 0.256325 to 0.028981 seconds (8.84 times), and Total Site improves
from 0.537118 to 0.324896 seconds (1.65 times). The equivalent three-generation
end-to-end runs improve from 27.861657 to 3.258408 seconds for Process and from
49.468948 to 30.589541 seconds for Total Site.

The final focused gate passes 328 tests. The complete configured-solver gate
passes 2,460 tests with 4 expected optional-profile skips and 3 deliberate
deselections tied only to the user's locally modified, long-running notebook
19. Ruff, formatting, patch hygiene, and the warnings-as-errors Sphinx build
pass. The notebook and `.gitignore` are unchanged by and excluded from this
refresh.

## Utility Placement Total Site Uniform-Tolerance Refresh

Temperature-grid construction, interval insertion, interpolation safeguards,
grid comparison, and Total Site profile assignment now use one canonical
absolute `1e-6 K` tolerance. The comparison normalizes machine-level
subtraction noise six decimal places below the domain resolution and no longer
uses the former scale-dependent relative tolerance.

The fixed 70/71-row failure, generated adjacent-utility profiles, a negative-
temperature boundary case, and a high-temperature absolute-only guard pass.
All 50 problem-table tests and the 296-test targeting and utility-placement
gate pass. A deterministic four-isothermal Total Site placement solves and its
detached case completes ordinary Total Site retargeting with 71 rows.

The complete applicable configured-solver gate passes 2,462 tests with 4
expected skips and 5 deliberate deselections tied only to the user's locally
modified notebook 19. Ruff, formatting, patch hygiene, and the warnings-as-
errors Sphinx build pass. The notebook and `.gitignore` remain unchanged by and
excluded from this correction.

## Utility Placement Cached Process-Profile and SUGCC Refresh

Aggregate candidate evaluation now completes every immediate process net-load
profile once per period, then targets each utility set by endpoint interpolation
against those cached profiles. Numeric duties are accumulated by utility level,
one candidate SUGCC is built, and the existing same-temperature cancellation
returns net site utility use. No aggregate candidate constructs a problem tree,
reruns child Direct targeting, reruns Indirect targeting, or regenerates graph
and report data.

Fresh hierarchy replay remains the independent oracle. Five structured
candidates match every named and fallback duty and both target totals exactly;
their balanced-composite entropy objectives agree within `1e-4 kW/K`. Fixed and
generated child-profile oracles, lifecycle counts, Process, Site, Community,
Region, multiperiod, pickle, and source-isolation regressions pass.

Ten representative Total Site candidates improve from `3.263975833 s` through
full hierarchy replay to `0.294159917 s` through cached targeting, an
`11.0959` times speed-up with identical duties and required targets. The focused
gate passes 372 tests. The complete configured-solver gate passes 2,467 tests
with 4 expected skips and 4 deliberate deselections tied only to the user's
modified notebook 19. Repository-wide Ruff lint, changed-file Ruff formatting,
patch hygiene, and the warnings-as-errors 54-source Sphinx build pass. Ten
untouched files remain outside the current repository-wide formatter baseline;
they were not cosmetically changed. The notebook and `.gitignore` remain
unchanged by and excluded from this refresh.
