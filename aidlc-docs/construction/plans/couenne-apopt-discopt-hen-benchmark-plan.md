# Couenne vs APOPT vs Discopt v0.6.0 HEN Benchmark Plan

## Status

- **Stage**: Code Generation, Part 1 - Planning
- **Approval**: Approved with environment amendment on 2026-07-15
- **Single source of truth**: This checklist controls implementation and execution.
- **Application code location**: Workspace root, primarily `scripts/` and `tests/`.

## Objective

Create and run a reproducible comparison of Couenne, APOPT, and Discopt v0.6.0
against OpenPinch's maintained OpenHENS synthesis fixtures. The comparison must
separate runtime, solution quality, feasibility, and optimality evidence, and
must not imply that three different solver stacks provide equivalent global
optimality guarantees.

## Confirmed Feasibility Findings

- OpenPinch 0.4.5 currently targets Python 3.14.2 or newer.
- Couenne and IPOPT are locally available from `/Users/timothyw/.idaes/bin`.
- APOPT is available through GEKKO 1.3.2.
- Couenne currently receives the GEKKO-generated HEN model through GEKKO's
  Pyomo converter.
- Discopt v0.6.0 provides a registered Pyomo solver through
  `discopt.pyomo` and `SolverFactory("discopt")`.
- The same GEKKO source equations can therefore feed Couenne and Discopt
  through the same Pyomo conversion boundary; APOPT receives the same source
  equations through GEKKO's native backend.
- Discopt v0.6.0 has no Apple-silicon Python 3.14 wheel, so it must be built
  from the exact v0.6.0 source distribution for the supported OpenPinch runtime.
- The user selected a shared Python 3.14 benchmark environment and authorized
  installation of a Rust toolchain for that source build.
- The primary comparison will use one isolated Python 3.14 environment for all
  three stacks. The report will capture the Rust version, source-distribution
  hash, built-wheel hash, and complete resolved Python package set.

## Comparison Contract

### Solver stacks

| Label | Algebraic source | Translation | Solver |
|---|---|---|---|
| Couenne | OpenPinch GEKKO HEN equations | GEKKO to Pyomo | Couenne global MINLP |
| APOPT | OpenPinch GEKKO HEN equations | GEKKO native | APOPT MINLP |
| Discopt 0.6.0 | OpenPinch GEKKO HEN equations | GEKKO to Pyomo to NL | Discopt spatial branch-and-bound |

The report will call these **backend-and-solver stacks**. It will not present
the timing comparison as a solver-only microbenchmark because APOPT does not
use the Pyomo translation layer.

### Fixture matrix

Use the seven maintained, non-reordered OpenHENS fixtures containing at most
nine process streams:

1. Five-stream Bogataj and Kravanja 2012
2. Five-stream Kim et al. 2017
3. Four-stream Escobar and Trierweiler 2013
4. Four-stream Yee and Grossmann 1990
5. Nine-stream Linnhoff and Ahmad 1999
6. Six-stream Spray Dryer 2025
7. Six-stream Yee and Grossmann 1990

Run a three-case smoke matrix first: Four-stream Yee and Grossmann,
Five-stream Bogataj and Kravanja, and Six-stream Yee and Grossmann. Advance to
the full matrix only when every solver can construct and attempt the same model
class successfully.

### Controlled inputs

- Use one quality tier, approach-temperature set, derivative threshold, stage
  selection, objective definition, feasibility tolerance, and external timeout
  per case.
- Set the PDM, TDM, and EVM solver consistently for each solver-stack run.
- Keep OpenPinch source revision, fixture revision, Python environment,
  dependency lock, hardware, process count, and thread limits fixed.
- Use deterministic case and solver order for the canonical run, and record it.
- Treat timeout, infeasibility, unsupported expression, verification failure,
  and solver exception as benchmark results rather than dropping those runs.

### Run protocol

- Perform one cold run in a fresh process for every solver and case.
- Perform at least two repeated runs in the same environment and classify them
  separately from the cold run so Discopt's JAX compilation cost remains
  visible rather than being silently amortized.
- Use process-level timeouts and preserve partial solver-call records.
- Do not run solver cases concurrently for the canonical timing comparison.
- Run a separate optional throughput experiment only after the serial canonical
  comparison is complete.

### Recorded metrics

- Environment and package versions, operating system, architecture, and CPU.
- Model dimensions when exposed: variables, integer variables, constraints,
  stages, and solver-call count.
- End-to-end time, solver-boundary time, and solver-reported time.
- Completion status, termination condition, failure category, and timeout.
- Objective or total annual cost, hot and cold utilities, exchanger count, and
  recovered heat.
- Independent OpenPinch network-verification result and maximum violation.
- Best bound, absolute gap, relative gap, and node count when exposed.
- Repetition variability and cold-versus-repeated timing.

## Implementation Checklist

### Step 1 - Reproducible environment

- [x] Install a minimal Rust toolchain capable of building Discopt v0.6.0 and
      record the exact `rustc`, Cargo, and rustup versions.
- [x] Create an isolated benchmark environment using Python 3.14.
- [x] Build and install `discopt[pyomo]==0.6.0` from its exact source
      distribution, recording source and wheel hashes and the complete resolved
      package set without changing OpenPinch's supported runtime dependencies.
- [x] Make Couenne and IPOPT available through an explicit benchmark PATH.
- [x] Run tiny Couenne, APOPT, and Discopt smoke solves and capture versions and
      termination metadata.
- [x] Verify that the environment is genuinely using Python 3.14 and that no
      incompatible prebuilt Discopt wheel was substituted.

Environment evidence: rustc 1.97.0, Cargo 1.97.0, and rustup 1.29.0 were
installed with the minimal profile for `aarch64-apple-darwin`; shell startup
files were not modified. Discopt was built as a native CPython 3.14 arm64 wheel
using PyO3's forward-compatibility mode after its declared Python 3.13 version
gate rejected the first build. Source, wheel, package, platform, and smoke-test
evidence is recorded in `results/hen_solver_benchmark/environment.json`.

### Step 2 - Benchmark-only Discopt adapter

- [x] Add Discopt registration and availability checks at the narrowest solver
      boundary needed by the benchmark.
- [x] Preserve existing public solver defaults and do not make Discopt a
      mandatory OpenPinch dependency.
- [x] Confirm that Couenne and Discopt consume the same converted Pyomo model and
      that APOPT consumes the same GEKKO source model.
- [x] Normalize Discopt status, objective, timing, bound, gap, and node metadata
      without discarding solver-specific fields.

### Step 3 - Solver-comparison harness

- [x] Add a dedicated solver-comparison entry point instead of overloading the
      existing quality-tier benchmark semantics.
- [x] Reuse existing case discovery, task tracing, timeout, partial-result, and
      failure-taxonomy helpers where their contracts match.
- [x] Add CLI selection for solvers, fixtures, repetitions, timeout, output
      paths, cold/repeated mode, and dry-run environment validation.
- [x] Emit an incremental raw JSON document after every attempted run.
- [x] Emit a machine-readable environment manifest with the raw results.

### Step 4 - Verification and aggregation

- [x] Independently verify every returned HEN before classifying it as feasible.
- [x] Compare objectives only among independently verified runs.
- [x] Calculate within-case objective deltas relative to the best verified result
      and never average raw objective values across different fixtures.
- [x] Summarize success rate, verified-feasible rate, median time, timing spread,
      objective delta, and optimality evidence per solver stack.
- [x] Clearly distinguish `optimal`, `locally optimal`, `feasible`, timed out,
      failed verification, and uncertified outcomes.

### Step 5 - Tests

- [x] Add example-based tests for solver registration, missing optional
      dependency behavior, CLI validation, timeout capture, partial writes,
      status normalization, and fixture selection.
- [x] Add fake-solver integration tests proving all three labels receive the
      same benchmark settings and algebraic-source case.
- [x] Add regression tests that objective aggregation excludes unverified runs
      and preserves solver failures as data.
- [x] Add Hypothesis invariant tests for generated result matrices: status
      counts sum to attempted runs, per-case deltas use only verified results,
      timing summaries remain non-negative, and serialization round-trips
      preserve solver and case identities.
- [x] Retain shrinking and reproducible seed reporting through the repository's
      existing Hypothesis configuration.

### Step 6 - Smoke benchmark

- [x] Run environment preflight and the three-case smoke matrix.
- [x] Inspect raw solver logs for unsupported Pyomo/NL expressions, status
      mistranslation, false feasibility, or inconsistent model dimensions.
- [x] Correct only benchmark integration defects; do not tune one solver on the
      basis of another solver's result.
- [x] Re-run the smoke matrix after any correction and preserve failed earlier
      attempts in the development record.

Smoke outcome: APOPT returned three OpenPinch-verified networks. Couenne and
Discopt each constructed and attempted all three cases but reached the common
60-second case limit. The Discopt logs identify unbounded auxiliaries and
unsupported nonconstant-division relaxations; these are retained as solver
capability evidence rather than addressed through solver-specific model tuning.

### Step 7 - Full canonical benchmark

- [x] Run all seven fixtures serially with identical external limits.
- [x] Run the configured repetitions and preserve cold and repeated results.
- [x] Generate a concise Markdown comparison from the raw JSON.
- [x] Include per-case tables before any cross-case aggregate.
- [x] State all unsupported cases, timeouts, verification failures, and missing
      optimality certificates explicitly.

Canonical outcome: 63 unique attempts completed across seven fixtures, three
solvers, and three fresh-process repetitions. APOPT returned 15 verified
networks across five fixtures; Couenne and Discopt returned no verified network
within the common limit. Couenne and Discopt each recorded 18 timeouts and
three common Spray Dryer decomposition failures. APOPT also recorded three
nine-stream decomposition failures and the three common Spray Dryer failures.

### Step 8 - Engineering verification

- [x] Run focused benchmark and solver-backend tests.
- [x] Run Ruff format and lint checks for all changed Python files.
- [x] Run the CI-equivalent non-solver test suite with coverage and confirm the
      repository remains above the enforced 95% coverage threshold.
- [x] Run relevant solver-marked HEN tests with Couenne/APOPT available.
- [x] Validate the raw JSON and generated Markdown parse cleanly.
- [x] Review the patch for optional-dependency isolation, deterministic output,
      bounded resource use, and absence of hard-coded local paths.

### Step 9 - Documentation and handoff

- [x] Document the benchmark command, environment constraint, result schema,
      solver-stack distinction, and interpretation limits.
- [x] Link the final report to the exact raw JSON and environment manifest.
- [x] Record whether Discopt should remain benchmark-only or proceed to a
      separately approved public solver integration.
- [x] Update this checklist and `aidlc-state.md` in the same interaction as each
      completed step.

Final artifacts:

- [Markdown comparison](../../../results/hen_solver_benchmark/full_comparison.md)
- [Raw canonical JSON](../../../results/hen_solver_benchmark/full_raw.json)
- [Derived summary JSON](../../../results/hen_solver_benchmark/full_summary.json)
- [Environment manifest](../../../results/hen_solver_benchmark/environment.json)

Discopt remains a private, optional, benchmark-only bridge. It is absent from
package dependency metadata and production defaults. Any supported public
solver contract, formulation changes for finite bounds or relaxation coverage,
and production documentation require a separately approved change.

## Acceptance Criteria

- Every comparison row comes from the same source revision, fixture, controlled
  settings, machine, and Python environment.
- Every reported objective is independently verified by OpenPinch.
- Cold-start and repeated timings are distinguishable.
- Solver failures and timeouts remain visible in the result set.
- The report distinguishes backend overhead and optimality guarantees.
- Discopt is pinned exactly to v0.6.0.
- No mandatory runtime dependency or public solver default changes are made.
- Focused tests, solver-marked tests, linting, and the CI-equivalent test suite
  pass, with coverage at or above 95%.

## Property-Based Testing Compliance

- **PBT-02**: Applicable to raw-result JSON round trips; planned.
- **PBT-03**: Applicable to aggregation invariants; planned.
- **PBT-07**: Domain-specific generated benchmark result records; planned.
- **PBT-08**: Existing Hypothesis shrinking and seed reproduction retained;
  planned.
- **PBT-09**: Hypothesis is already selected and available in the development
  environment; compliant.
- Security Baseline: disabled, not enforced.
- Resiliency Baseline: disabled, not enforced.

## Text Execution Sequence

Environment preflight, then benchmark adapter, then comparison harness, then
tests, then the smoke matrix, then the full serial matrix, then CI and coverage
verification, and finally documentation and result interpretation.
