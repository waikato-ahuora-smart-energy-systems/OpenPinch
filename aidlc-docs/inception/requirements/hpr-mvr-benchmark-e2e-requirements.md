# HPR and MVR Benchmark End-to-End Reliability Requirements

## Intent analysis

- **User request**: Investigate HPR and MVR service robustness and add
  end-to-end tests using problems from the standard targeting suite.
- **Request type**: Reliability investigation and test-suite enhancement.
- **Scope**: Public `PinchProblem` HPR targeting services, optimized MVR
  targeting, direct process MVR, result contracts, and atomic problem state.
- **Complexity**: Moderate. The code change is test-focused, but the real
  CoolProp matrix spans 54 thermodynamically diverse process problems and must
  remain bounded in normal CI.
- **Requirements depth**: Standard.

## Confirmed decisions

1. Reuse all 54 problems discovered by `tests/e2e/test_main.py` from
   `examples/stream_data/p_*.json`.
2. Assign one deterministic bounded HPR profile to every standard problem,
   rotating heat pumping, refrigeration, optimized MVR, direct/utility, and
   cascade/parallel variants rather than running the full Cartesian product.
3. Add packaged `process_mvr.json` as the dedicated direct process-MVR case,
   because the standard targeting inputs do not contain pressure-qualified gas
   streams.
4. Run the representative real-CoolProp matrix in normal CI. Broader stress
   coverage, if useful, remains explicit slow or release validation.
5. Preserve disabled Security and Resiliency extensions. Preserve full
   Property-Based Testing enforcement.

## Functional requirements

### Corpus and matrix

- **FR-01**: The HPR benchmark must derive its problem list from the same
  `_example_problem_filepaths()` owner used by the existing end-to-end targeting
  suite; it must not duplicate a hand-maintained filename list.
- **FR-02**: Every discovered standard problem must receive exactly one profile
  from a stable six-profile rotation:
  1. direct cascade vapour-compression heat pump;
  2. utility parallel vapour-compression heat pump;
  3. direct cascade vapour-compression refrigeration;
  4. utility parallel vapour-compression refrigeration;
  5. direct optimized VC+MVR heat pump;
  6. utility optimized VC+MVR heat pump.
- **FR-03**: With the current 54-case corpus, each profile must cover nine
  distinct problems. Corpus growth must remain discoverable and deterministically
  assigned without changing production code.
- **FR-04**: Real CoolProp thermodynamics and the public `PinchProblem` surface
  must be exercised. The tests must not replace the optimizer, property calls,
  objective, transaction, or serialization boundary with mocks.
- **FR-05**: Each HPR call must use explicit finite search controls, one restart,
  deterministic fluids and bounded stage counts. Construction-time probing must
  select the smallest evaluation allowance that exercises real search while
  keeping the full representative matrix suitable for normal CI; it must not
  exceed 50 distinct search evaluations per call without a documented revision.

### Robust outcome contract

- **FR-06**: A benchmark invocation may finish in one of three documented
  states: a valid solved target, a legitimate no-op where no applicable positive
  target exists, or `HPRTargetingError` with structured diagnostics for bounded
  physical/search infeasibility.
- **FR-07**: No standard problem may escape with an untyped thermodynamic,
  numerical, serialization, lifecycle, or indexing exception.
- **FR-08**: A successful result must contain finite objective/accounting
  values, no live CoolProp model, a detached CoolProp target simulation record,
  positive useful duty, internally consistent loop duty, and JSON-serializable
  public results.
- **FR-08A**: The exploratory matrix must identify at least one named expected-
  success case for each of the six profiles. Those six sentinels must satisfy the
  full success contract; a typed failure or no-op is not a successful solve and
  cannot satisfy this profile-level oracle.
- **FR-08B**: Each profile sentinel must include a bounded convergence witness
  from the real search objective: at least two distinct viable search points,
  at least one strict incumbent-objective improvement exceeding
  `1e-8 * max(1, abs(first_viable_objective))`, and a selected final objective
  equal to the best viable objective observed within that tolerance.
- **FR-09**: A typed failure must contain bounded structured diagnostics,
  a nonnegative evaluation count within the configured allowance for the search,
  a recognized failure category, and serializable/copyable public evidence.
- **FR-10**: A no-op or typed failure must leave the pre-call public problem
  results unchanged. Success must commit exactly one new HPR target for the
  invocation.
- **FR-11**: The benchmark report produced by pytest failures must identify the
  problem filename and assigned service profile without requiring debug logs.

### Direct process MVR

- **FR-12**: An end-to-end test must load packaged `process_mvr.json`, create a
  process-MVR component through the public component accessor, and run targeting
  on the transformed problem.
- **FR-13**: The direct-MVR test must verify positive finite stage work and duty,
  pressure increase, replacement-stream insertion, component/result detachment,
  and successful downstream target serialization.
- **FR-14**: The existing invalid-fluid and second-stage atomicity tests remain
  complementary regression coverage; the new direct-MVR test must exercise a
  successful public workflow without mocking CoolProp.

### Investigation and regression handling

- **FR-15**: Run the proposed matrix as an investigation before freezing
  assertions. Classify every unexpected exception by root cause and correct
  production defects exposed by valid benchmark inputs rather than weakening
  the oracle.
- **FR-16**: If a benchmark exposes an intentionally unsupported or physically
  infeasible case, encode the documented public outcome contract rather than a
  private exception message or unstable optimizer objective.
- **FR-17**: Any corrected production defect must receive a minimal focused
  regression in addition to the end-to-end case that exposed it.

## Non-functional requirements

- **NFR-01 Determinism**: Problem discovery, profile assignment, explicit
  fluids, budgets, stage counts, and Hypothesis seed behavior must be stable.
- **NFR-02 Bounded execution**: Each optimization has a hard evaluation cap;
  the normal-CI test has no unbounded retries and no wall-clock assertion that
  would make slower supported machines flaky.
- **NFR-03 Isolation**: Each parameterized case creates a fresh `PinchProblem`.
  Tests must not share mutable targets, streams, optimizer caches, or engine
  objects across cases.
- **NFR-04 Diagnostics**: Failure output must retain the pytest parameter ID,
  service profile, typed diagnostic category counts, and bounded representative
  reasons.
- **NFR-05 Maintainability**: The e2e corpus owner remains singular. Matrix
  configuration is declarative and located with the e2e tests, not production
  service code.
- **NFR-06 CI compatibility**: The representative matrix runs under the existing
  non-solver pytest selection and requires no external solver binary, network,
  new dependency, or workflow change.
- **NFR-07 Public-boundary focus**: Tests assert stable public contracts and
  thermodynamic invariants, not exact optimizer coordinates, timing, or private
  implementation details.

## Verification requirements

- **VR-01**: The existing end-to-end suite still solves every shipped example
  for direct heat integration.
- **VR-02**: Corpus/profile coverage proves all discovered paths are assigned
  once and all six profiles are represented evenly for the current corpus.
- **VR-03**: The complete bounded HPR matrix finishes with only the three allowed
  public outcomes and no unexpected exception.
- **VR-03A**: At least one stable sentinel per profile returns a non-null target
  with `hpr_success` and `hpr_details.success` true and passes every FR-08
  invariant.
- **VR-03B**: Observation-only tracing of the unmodified search records each
  sentinel's distinct viable objective sequence, incumbent improvement, final
  selection gap and evaluation count. All six sentinels pass FR-08B.
- **VR-04**: Successful targets pass detachment, duty, finite-value, copy, and
  JSON round-trip assertions.
- **VR-05**: Typed failures pass diagnostic bounds, copy, serialization, and
  state-atomicity assertions.
- **VR-06**: The direct process-MVR public workflow succeeds and its downstream
  target remains serializable.
- **VR-07**: Focused HPR/MVR, contracts, application, e2e, formatting, lint and
  patch-hygiene gates pass.
- **VR-08**: The investigation records per-profile success, no-op and typed-
  failure counts, plus every production correction required by the corpus.

## Property-Based Testing applicability

- **PBT-01/PBT-03**: The fixed benchmark corpus introduces no production
  transformation, but corpus-to-profile assignment has coverage and determinism
  invariants that must be tested. Functional design must decide whether a small
  generated-index property adds value beyond exact 54-case coverage.
- **PBT-02**: Existing generated HPR diagnostic/result JSON round trips remain
  applicable; the new e2e suite adds concrete real-engine round trips.
- **PBT-04**: No idempotent production operation is added; N/A unless a reusable
  normalizer is introduced.
- **PBT-05**: The existing e2e problem list is the corpus oracle. The new suite
  must prove it consumes the same discovered paths.
- **PBT-06**: Problem transaction state is mutable; explicit success/failure
  transition assertions complement the existing stateful cache properties.
- **PBT-07/PBT-08/PBT-09**: Reuse domain-constrained strategies, Hypothesis
  shrinking, seed `20260715` in CI, and the existing pytest/Hypothesis stack.
- **PBT-10**: Properties supplement, not replace, per-problem real CoolProp
  examples and focused regressions.

## Out of scope

- Exhaustive full-Cartesian HPR combinations across all 54 problems.
- Global-optimum claims or universal thermodynamic feasibility.
- Treating bounded best-observed convergence as proof of a global optimum.
- New public APIs, optimizer algorithms, dependency changes, external solvers,
  deployment, or infrastructure.
- Treating runtime duration as a service-level agreement.
