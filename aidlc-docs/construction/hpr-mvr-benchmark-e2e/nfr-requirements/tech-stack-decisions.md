# Technology Stack Decisions

## Selected stack

| Concern | Selection | Rationale |
|---|---|---|
| Test runner | pytest 9 or newer | Existing repository runner, parameter IDs, markers and CI integration. |
| Property testing | Hypothesis 6.135 or newer | Existing configured dependency with domain strategies, shrinking and seed replay. |
| Thermodynamics | CoolProp 7-compatible project constraint | The services under test use CoolProp; replacing it would invalidate the e2e boundary. |
| Numerical runtime | NumPy and SciPy from the project environment | Existing HPR service dependencies; no alternate numerical stack required. |
| Contracts | Pydantic from the project environment | Public result and diagnostic serialization boundary. |
| Fixtures | JSON via `pathlib` and existing repository paths | Same inputs and loading behavior as the standard e2e suite. |
| Quality gates | Ruff, pytest, compileall, and git diff checks | Existing repository quality toolchain and CI behavior. |

## PBT framework decision

Hypothesis remains the sole property-based framework. It supports custom domain
strategies, automatic shrinking, reproducible seeds and native pytest discovery.
It is already present in `pyproject.toml`; no package or lockfile change is
permitted for this unit.

The complete fixed 54-case matrix is example-based e2e coverage. Hypothesis is
reserved for genuine general properties such as generated assignment indices or
state transitions when the generated oracle is independent and non-tautological.
Existing HPR diagnostic, serialization, budget and stateful-cache properties are
rerun as complementary evidence.

PBT-09 is compliant.

## Rejected alternatives

- **Separate benchmark script**: rejected because the user requested blocking
  e2e tests and the existing corpus owner is pytest-based.
- **Mocked CoolProp/optimizer**: rejected because it would not verify the public
  thermodynamic transaction.
- **Synthetic convergence results**: rejected. A test wrapper may observe and
  delegate to the real search evaluator but may not replace or alter it.
- **Full Cartesian matrix**: rejected by the approved bounded representative
  scope and normal-CI cost.
- **External solver marker**: rejected because HPR CoolProp execution does not
  require an external mathematical-programming binary.
- **New snapshot framework**: rejected because stable public invariants and JSON
  contracts are sufficient and more maintainable than volatile numerical files.
- **Timing assertions**: rejected as machine-dependent and contrary to the
  requirements.
- **Global-optimum assertion**: rejected because the standard corpus has no
  certified HPR optimum. The oracle is material incumbent improvement plus
  selection of the best viable point actually observed within the hard budget.

## Compatibility constraints

- Production public APIs and contracts remain unchanged unless the benchmark
  proves a valid-input defect.
- Test helpers stay under `tests/e2e` and may depend on repository support paths;
  production packages must not depend on tests.
- The suite must pass the same `not solver` CI selection used today.
- No environment variable, network service or generated fixture is required.
