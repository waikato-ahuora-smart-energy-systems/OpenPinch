# Business Logic Model

## Standard corpus discovery

1. Resolve the repository `examples/stream_data` directory.
2. Discover files matching `p_*.json`.
3. Sort by filename.
4. Return an immutable sequence used by both the existing direct-target test and
   the new HPR benchmark.
5. Fail the corpus coverage test if either consumer introduces a separate list.

Discovery remains test-owned. Production code does not learn about benchmark
fixtures.

## Profile assignment

Maintain an ordered immutable tuple of six profile specifications. Enumerate the
sorted corpus and select `profiles[index % len(profiles)]` for each problem.
Build pytest parameter IDs from `<filename>::<profile_id>`.

The current 54-case corpus gives nine cases per profile. Coverage assertions pin
that fact while the modulo rule permits deterministic corpus growth.

## Invocation preparation

For each assignment:

1. load a fresh `PinchProblem` from JSON;
2. capture its public result snapshot and target count;
3. resolve the named public target service;
4. supply explicit load fraction, Water-based working fluids, port/stage counts,
   one restart, bounded iterations, bounded evaluations, and direct/utility and
   topology flags from the profile;
5. invoke the real public service without thermodynamic or optimizer mocks.

Construction-time exploration selects the smallest stable evaluation limit that
exercises real search, capped at 50 per call by the approved requirement.

## Outcome classification

### Solved

Classify as solved only when a non-null target is returned and both
`target.hpr_success` and `target.hpr_details.success` are true. Validate:

- exactly one target was added;
- target scope and HPR details are public and detached;
- objective/accounting values are finite;
- `target_simulation_record` exists, identifies CoolProp, has positive useful
  duty, and has internally consistent finite loop/stage values;
- the live model is absent;
- deep copy and JSON serialization succeed.

After exploration, select at least one stable solved problem for each of the six
profiles as a named success sentinel. A no-op or typed failure remains an
acceptable robustness outcome for an arbitrary matrix assignment but cannot
satisfy a profile's success sentinel.

### Bounded convergence witness

For strict-success sentinels, wrap the shared search-evaluation boundary with an
observation-only delegate. The wrapper must call the original function exactly
once with unchanged arguments and record only search-mode results. It does not
replace the optimizer, objective, CoolProp or transaction behavior.

Deduplicate by the exact design-point tuple and retain the ordered viable
objective values. The sentinel demonstrates progress when:

1. at least two distinct search points are viable;
2. a later viable objective is smaller than the first viable objective by more
   than `1e-8 * max(1, abs(first_viable_objective))`;
3. the incumbent-best sequence is non-increasing;
4. the selected final target objective equals the minimum viable objective
   observed within the same tolerance; and
5. the distinct evaluation count stays within the configured allowance.

This proves material progress toward the best solution observed during the
bounded run. It does not prove global optimality, and the tests must say so.

### No-op

Classify as no-op when the public method returns `None` without exception.
Require byte-for-byte equality of the pre-call and post-call public result JSON
and unchanged target count.

### Typed failure

Catch only `HPRTargetingError`. Validate:

- public state is unchanged;
- diagnostics are present and bounded;
- evaluation counts are nonnegative and consistent with the configured search
  allowance and documented final reevaluations;
- category counts and representative failures satisfy contract bounds;
- deep copy, pickle where supported, and JSON-compatible model serialization
  succeed.

Any other exception fails the case and starts a focused reproduction. A defect
is fixed in its owning production component, then pinned with a focused test and
the original e2e case.

## Direct process-MVR workflow

1. Load a fresh `PinchProblem("process_mvr.json")`.
2. Capture source-stream and result state.
3. Call `problem.components.add_process_mvr` for `Evaporator vapour` using
   explicit bounded compression settings.
4. Validate returned stage work/duty, pressure lift, replacement streams,
   registered inventory and absence of a retained live property engine.
5. Run `problem.target.direct_heat_integration()`.
6. Validate finite targets and JSON serialization of the resulting public state.

This is the concrete successful complement to existing focused invalid-fluid,
transport, fallback and second-stage atomicity tests.

## Investigation output

The exploratory run records counts by profile and outcome. It reports every
unexpected exception with case/profile identity. Counts describe the tested
environment and are not frozen as universal thermodynamic feasibility claims.

## Testable properties

- Complete fixed-corpus enumeration is the primary oracle.
- A generated index property may verify range and repeatability of profile
  selection across corpus-growth indices, but only if it is independent of the
  production expression under test.
- Existing HPR round-trip and stateful cache properties remain mandatory and are
  rerun; the e2e matrix supplies concrete real-engine examples under PBT-10.
- A generated convergence-trace oracle may verify incumbent monotonicity and
  tolerance handling independently of CoolProp; real sentinels prove the same
  rule through the public service.
