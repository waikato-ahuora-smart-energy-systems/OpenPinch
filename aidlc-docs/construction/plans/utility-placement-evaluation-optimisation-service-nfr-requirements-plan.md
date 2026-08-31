# Unit 2 Placement Evaluation and Optimisation Service NFR Requirements Plan

## Purpose

Define measurable capacity, performance, numerical, reproducibility,
reliability, compatibility, observability, maintainability, and test-stack
requirements for Unit 2 before NFR Design and TDD implementation.

## Plan Progress

- [x] Load the Unit 2 Functional Design, approved project NFRs, Unit 1 NFR
  baseline, NFR Requirements stage rules, and enabled PBT rules.
- [x] Inspect the current Python/runtime dependencies, optimizer backend
  behavior, fixed-seed CI configuration, coverage gate, and test markers.
- [x] Evaluate scalability, performance, availability, security, technology,
  reliability, maintainability, and usability categories for applicability.
- [x] Identify the unresolved measurable NFR decisions below.
- [x] Create these questions using the mandatory question-file format.
- [x] Collect and validate every answer; create clarification questions if any
  response is missing, invalid, contradictory, or ambiguous.
- [x] Generate `nfr-requirements.md` with measurable acceptance criteria and
  explicit N/A rationale for service-process availability and infrastructure.
- [x] Generate `tech-stack-decisions.md` with existing-stack reuse, optimizer
  adapter, numerical precision, test organization, CI profiles, and PBT-09
  evidence.
- [x] Validate Markdown syntax, NFR/story traceability, PBT-09 compliance,
  compatibility with Unit 1, and consistency with disabled Security and
  Resiliency extensions.
- [x] Present the standardized NFR Requirements completion checkpoint and wait
  for explicit approval.

## Questions

Please answer every question by entering one option letter after its
`[Answer]:` tag. Choose the final Other option only when the listed choices do
not express the required behavior, and add the intended behavior after the tag.

## Question 1

Which performance policy should govern Unit 2, given that complete optimization
time depends on the selected backend and caller evaluation budget?

A) Use tiered deterministic budgets: pure entropy/monetary kernels at p95 no
more than 50 ms for 40 levels across 100 periods; one cold candidate replay at
p95 no more than 1 second for 16 levels across 12 periods on fixture data; an
exact memo hit at p95 no more than 1 ms; and no universal full-solve wall-clock
promise beyond honoring evaluation/iteration limits (recommended)

B) Require every representative complete optimization to return within 5
seconds regardless of method or evaluation budget

C) Set no latency targets and verify only that iteration/evaluation limits are
honored

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 2

Which numerical coverage criterion should determine whether allocated utility
duty matches a period's residual demand?

A) Use a combined criterion `abs(residual) <= coverage_absolute +
relative_tolerance * max(required_duty, 1 kW)`, defaulting to the existing
`1e-6` absolute and `1e-9` relative tolerances (recommended)

B) Use only the existing absolute `1e-6 kW` coverage tolerance

C) Use a fixed relative tolerance of `1e-6` with no absolute floor

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 3

How should exact-coordinate memoization behave when an existing optimizer uses
multiple worker processes or concurrent callbacks?

A) Keep one concurrency-safe memo per execution process/session; require the
objective payload to be pickle-safe, allow duplicate physical evaluation
across isolated workers, and perform exact deduplication plus canonical
re-evaluation in the parent before result assembly (recommended)

B) Force all optimizer evaluations through one serialized parent-process memo,
even when the backend normally evaluates runs in parallel

C) Reject `run_count > 1` for utility placement so only one memo can exist

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 4

What real-optimizer coverage should be required in routine CI?

A) Exercise all service logic with injected deterministic optimizers and the
structured-grid oracle, add one small fixed-seed real dual-annealing regression
to the routine non-solver suite, and retain broader backend behavior in the
existing optimizer tests (recommended)

B) Run complete utility-placement regressions against all four optimizer
methods in every routine CI job

C) Use injected optimizers only and defer every real optimizer invocation to
manual testing

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 5

Which numerical reference should verify entropy formulas and limiting behavior?

A) Use ordinary binary64 `math`/NumPy calculations in production and a
high-precision standard-library `decimal` oracle in tests for generated small
branches and near-isothermal limits (recommended)

B) Use NumPy `longdouble` for both production and tests where the platform
supports it

C) Use high-precision decimal arithmetic in production for every candidate
evaluation

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 6

How much rejected-candidate diagnostic information should one solve retain?

A) Keep aggregate counts by stable failure code plus at most ten representative
diagnostics in deterministic severity/period/side/template order; never retain
an unbounded callback trace (recommended)

B) Retain every rejected-candidate diagnostic for complete forensic detail

C) Retain only the final rejected candidate's first diagnostic

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 7

What optional dependency policy should Unit 2 follow?

A) Add no dependency: reuse existing Python, Pydantic/Pint, NumPy/SciPy,
CoolProp steam-turbine, pytest, Hypothesis, coverage, Ruff, and Hatchling
declarations and preserve current import boundaries (recommended)

B) Add a dedicated nonlinear optimization package for utility placement

C) Add a high-precision numerical package for production entropy calculations

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Preliminary Applicability Assessment

| Category | Assessment before answers |
|---|---|
| Scalability/capacity | Applicable to period-by-level replay, evaluation budgets, and bounded diagnostic/memo growth |
| Performance | Applicable to pure kernels, cold candidate replay, memo hits, and budget honoring; backend-wide wall time requires a user policy |
| Numerical correctness | Applicable to coverage, entropy limits, aggregation, penalty separation, units, and finite-value handling |
| Reproducibility | Applicable to seeds, exact memo keys, process isolation, canonical parent re-evaluation, and candidate ordering |
| Reliability | Applicable to complete-or-typed-failure behavior, no retries, and recoverable/non-recoverable adapter failures |
| Availability/continuity | N/A for uptime/failover because Unit 2 is an in-process stateless library service with no persistent store |
| Security | No extension enforcement because Security remains disabled; the no-new-network/file/credential/code-execution boundary remains mandatory |
| Maintainability | Applicable to owner boundaries, injected adapters, typed modules, architecture/Ruff/coverage gates, and bounded diagnostics |
| Usability/observability | Applicable to stable error codes, reproducibility context, units, and deterministic result metadata; UI/accessibility is N/A |
| Technology/PBT-09 | Applicable; the existing Python/pytest/Hypothesis stack appears sufficient and must be formally verified |
