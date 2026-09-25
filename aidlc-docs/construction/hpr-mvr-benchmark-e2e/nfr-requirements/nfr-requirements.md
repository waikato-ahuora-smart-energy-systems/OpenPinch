# NFR Requirements

## Performance and bounded execution

| ID | Requirement | Verification |
|---|---|---|
| NFR-BE-01 | Every HPR invocation uses one restart and explicit finite iteration and evaluation limits. | Instrumented diagnostic assertion for each matrix case. |
| NFR-BE-02 | The selected normal-CI evaluation allowance must be the smallest stable value found by exploratory execution and must not exceed 50 distinct search evaluations per call. | Matrix configuration inspection plus observed diagnostic counts. |
| NFR-BE-03 | Exact-coordinate cache hits remain free under the documented search budget. | Existing stateful cache/property regressions. |
| NFR-BE-04 | The suite performs no automatic retries, sleeps, network access, or external solver startup. | Code inspection and ordinary non-solver CI execution. |
| NFR-BE-05 | No wall-clock assertion is used as a correctness criterion. Runtime is recorded diagnostically during exploration only. | Test-source inspection. |

The matrix adds 54 bounded real-CoolProp calls plus one direct process-MVR
workflow. If exploratory evidence shows that this cannot fit normal CI without
unreasonable cost, the design must be revised and re-approved rather than
silently skipping problems or weakening thermodynamic execution.

## Reliability and correctness

| ID | Requirement | Verification |
|---|---|---|
| NFR-RL-01 | Only solved, no-op, and typed `HPRTargetingError` outcomes are accepted. | Complete 54-case parameterized matrix. |
| NFR-RL-01A | Every profile has at least one named case that must satisfy the strict solved contract. | Six profile-level success sentinels selected from exploratory evidence. |
| NFR-RL-01B | Every sentinel shows material bounded objective convergence and selects the best viable objective observed. | Observation-only ordered search trace and independent tolerance oracle. |
| NFR-RL-02 | Unexpected exceptions retain their original type and traceback and fail the named case/profile. | No broad catch in the e2e harness; focused defect regressions. |
| NFR-RL-03 | Failure and no-op are transaction-atomic; success commits exactly one target. | Pre/post public JSON and target-count assertions. |
| NFR-RL-04 | Successful results and diagnostics are detached, copyable, bounded, and serializable. | Deep-copy, model serialization and existing pickle regressions. |
| NFR-RL-05 | Each case uses fresh problem and engine state. | Fixture construction inside each parameterized invocation. |
| NFR-RL-06 | Direct process MVR exercises a real successful property workflow and downstream targeting. | Dedicated public e2e test. |

## Determinism and reproducibility

- Problem paths are sorted by filename through one shared discovery owner.
- Profile assignment depends only on sorted ordinal modulo six.
- Fluids, stages, load fraction, topology, placement mode and search controls are
  explicit; environment-dependent defaults are not part of the oracle.
- Hypothesis continues to use the CI seed `20260715`, with ordinary shrinking
  and failure examples.
- The matrix does not assert a global optimum, exact candidate coordinate,
  exact elapsed time, or universal feasibility. It asserts best-observed
  progress and final-selection consistency under the configured budget.

## Scalability

- Discovery and assignment are O(n) in the number of standard problems and use
  O(n) immutable pytest parameters.
- Runtime scales linearly with corpus size times the explicit evaluation cap.
- Current acceptance is 54 problems and six evenly represented profiles.
- A larger corpus remains deterministically assigned. A material runtime growth
  requires explicit test-tier review rather than hidden sampling.

## Maintainability and diagnostics

- Shared discovery prevents drift between direct-target and HPR e2e suites.
- Profile configuration is declarative, immutable and test-owned.
- Parameter IDs include filename and profile.
- Assertion messages distinguish corpus, profile, outcome and diagnostic facts.
- E2E tests assert public behavior; private numerical details remain in focused
  analysis tests.
- Investigation evidence records per-profile outcome counts and every confirmed
  defect/correction.

## Portability and CI compatibility

- Tests run under the repository's ordinary Python, pytest, NumPy, Pydantic,
  SciPy and CoolProp environment.
- They remain in the non-solver marker selection on supported CI platforms.
- No IDAES, Pyomo, GEKKO, TESPy binary, browser, network or platform-specific
  resource is required.
- Paths use `pathlib` and repository support constants.

## Availability and operations

This is a local library test suite with no deployed service, persistence daemon,
uptime target, failover, disaster recovery, or operational alerting surface.
Availability requirements are N/A.

## Security and privacy

Security Baseline is disabled by user decision. The workflow processes only
repository fixtures and creates no credentials, network calls, personal data,
or external artifacts. General repository safety and dependency policies remain
applicable; no additional security control is required for this unit.

## Property-Based Testing compliance

- PBT-08: deterministic seed, shrinking and normal CI discovery remain enabled.
- PBT-09: pytest plus Hypothesis is the selected supported Python stack.
- The framework is already declared in the development dependency group and
  requires no dependency change.
- Remaining PBT rules are addressed by functional design and the later code plan.
- No blocking PBT finding exists at NFR Requirements.

## Extension compliance

- Property-Based Testing: enabled and compliant for applicable NFR rules.
- Security Baseline: disabled and skipped.
- Resiliency Baseline: disabled and skipped.
