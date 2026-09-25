# Business Rules

## Corpus and assignment rules

- **BR-01**: There is one shared sorted owner for standard e2e problem paths.
- **BR-02**: Every standard problem is assigned exactly once.
- **BR-03**: Assignment is determined only by sorted ordinal modulo six.
- **BR-04**: Profile definitions are immutable, explicit and descriptive.
- **BR-05**: The current 54-case matrix contains exactly nine assignments for
  each profile.

## Invocation rules

- **BR-06**: Every case creates a fresh `PinchProblem`.
- **BR-07**: Real public accessors, CoolProp, objective and transaction logic are
  used; monkeypatching these boundaries is prohibited in the e2e test.
- **BR-08**: Fluids, load, stages, topology, direct/utility mode and all search
  budgets are explicit.
- **BR-09**: Maximum evaluations never exceed 50 per search in the normal-CI
  matrix without approved requirements revision.
- **BR-10**: The matrix has no retry-on-failure behavior.

## Outcome rules

- **BR-11**: Solved, no-op and typed failure are the only accepted outcomes.
- **BR-12**: Unexpected exceptions are defects, not infeasible-case aliases.
- **BR-13**: Success adds exactly one HPR target and exposes a detached CoolProp
  record with positive useful duty; both public success flags are true.
- **BR-14**: No-op and typed failure preserve public result state exactly.
- **BR-15**: A typed failure must expose structured, bounded, transportable
  diagnostics.
- **BR-16**: Assertions must use public contracts and thermodynamic invariants,
  not exact optimizer coordinates or runtime duration.
- **BR-17**: Pytest IDs contain both the problem filename and profile identifier.
- **BR-17A**: Every profile has at least one named strict-success sentinel. No-op
  and typed failure demonstrate robust handling but are not successful solves.
- **BR-17B**: Every sentinel has at least two distinct viable search evaluations,
  a material strict incumbent improvement, and a final selected objective equal
  to the best viable objective observed within tolerance.
- **BR-17C**: Convergence means progress toward the bounded best-observed
  objective. It must not be described as proof of global optimality.

## Direct process-MVR rules

- **BR-18**: Direct MVR uses the packaged pressure-qualified sample rather than
  inventing pressure data for the standard targeting corpus.
- **BR-19**: Stage work, duty and outlet pressure are finite and positive; each
  replacement stream is attached through the public component transaction.
- **BR-20**: Downstream direct targeting and public serialization succeed after
  component insertion.
- **BR-21**: No live CoolProp engine is retained in public component or result
  state.

## Defect-handling rules

- **BR-22**: Each unexpected matrix failure is reproduced as the smallest
  independent case before editing production code.
- **BR-23**: Only valid-input production defects are corrected; expected
  physical infeasibility remains a typed public failure.
- **BR-24**: Every correction has a focused regression plus the original e2e
  coverage.
- **BR-25**: Failure assertions do not pin private exception strings unless the
  text is an explicit public contract.
- **BR-26**: Convergence instrumentation is observation-only: it delegates once
  to the real evaluation boundary with unchanged inputs and cannot synthesize,
  suppress or modify candidates or results.

## Property-Based Testing compliance

- **PBT-01**: Compliant; properties and categories are documented in all three
  functional artifacts.
- **PBT-02/PBT-03/PBT-05/PBT-06/PBT-10**: Applicable through round trips,
  assignment/corpus invariants, the corpus oracle, transaction states and
  complementary concrete examples.
- **PBT-04**: N/A; no new idempotent operation is designed.
- **PBT-07/PBT-08/PBT-09**: Existing domain strategies, shrinking, deterministic
  CI seed and Hypothesis framework remain applicable.
- No blocking PBT finding exists at Functional Design.
