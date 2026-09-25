# Test-Suite Runtime Reduction Tech Stack Decisions

## Retained Stack

| Concern | Decision | Rationale |
|---|---|---|
| Runtime | Python 3.14 and `uv` | Matches the profiled and CI environment |
| Test runner | pytest | Existing markers, fixtures, durations, and selection support are sufficient |
| Property testing | Hypothesis | Existing domain strategies, shrinking, and seeded reproducibility must remain |
| Coverage | Coverage.py through `coverage run` | Retains the existing branch-aware 95 percent gate |
| Profiling | pytest duration reports and JUnit timing | Requires no new dependency and matches the audit baseline |
| CI | GitHub Actions | Existing pull-request, develop, and publish workflows own all validation |
| Documentation | Sphinx through the existing pytest smoke | Preserves warnings-as-errors output validation while avoiding a second build |
| Tutorial generation | Existing Python notebook generator | Preserves canonical source ownership and drift enforcement |
| Thermodynamics | CoolProp and optional TESPy | Required real-engine evidence remains unchanged |
| Optimisation | Existing OpenPinch and SciPy-backed services | No production solver or backend replacement is in scope |

## Decisions

### No parallel-test dependency

`pytest-xdist` will not be added in this unit. The measured issues are repeated
work and lane duplication; correcting those first provides deterministic,
portable savings without introducing process-scheduling and coverage-combine
complexity. Parallel execution can be evaluated separately after the serial
baseline is optimized.

### No benchmark plugin

The repository will continue to use pytest durations and recorded wall-clock
commands. Adding `pytest-benchmark` is unnecessary for the bounded end-to-end
comparisons and would change the dependency surface.

### Marker-based lane ownership

Central pytest markers are the stable contract between tests and CI. CI jobs
shall select marker categories rather than maintain test-path allowlists.

### Coverage strategy

The ordinary lane remains the primary coverage gate. Dedicated TESPy,
performance, and documentation lanes prioritize real behavior and clear
failure attribution. Coverage artifact combination will be introduced only if
measurement proves the specialized exclusions make 95 percent unattainable
without duplicate expensive execution.

### Fixture strategy

pytest module-scoped construction plus immutable serialization or deep-copy
isolation is preferred for repeated solved evidence. Global mutable caches and
session-scoped live `PinchProblem` instances are prohibited.

## Extension Compliance

Hypothesis remains the selected Property-Based Testing framework and stays in
the development dependency group. No framework, generator, shrinking, or seed
behavior changes. Security and Resiliency extensions are disabled and N/A.
