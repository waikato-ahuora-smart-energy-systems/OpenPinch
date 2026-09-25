# Logical Components

All components are test-owned unless an exploratory failure proves a production
defect. Names below describe responsibilities; Code Generation may use small
functions or frozen data classes rather than creating unnecessary classes.

## Proposed file ownership

| File | Responsibility |
|---|---|
| `tests/e2e/cases.py` | Single sorted owner for standard `p_*.json` discovery. |
| `tests/e2e/hpr_benchmark.py` | Frozen profiles, assignments, observations, outcome validation and pure convergence analysis. |
| `tests/e2e/test_hpr_mvr.py` | The 54-case public HPR matrix, six strict sentinels and direct process-MVR workflow. |
| `tests/e2e/test_hpr_benchmark_helpers.py` | Focused example and Hypothesis tests for pure assignment and convergence helpers, if separation keeps the e2e module concise. |
| `tests/e2e/test_main.py` | Existing direct-target suite, changed only to import shared corpus discovery. |

If helper scope remains small, the two helper files may be combined. Corpus
ownership must remain separate and shared; production code must not depend on
any of these modules.

## Benchmark corpus

**Inputs**: repository root and `examples/stream_data`.

**Output**: immutable sorted paths for every `p_*.json` problem.

**Responsibilities**:

- perform dynamic discovery once per test collection;
- reject duplicates or non-files;
- provide the identical sequence to direct-target and HPR tests;
- expose enough stable naming for pytest IDs without loading problem models.

## HPR profile catalogue

**Inputs**: none beyond declared constants.

**Output**: six ordered immutable profile specifications.

**Responsibilities**:

- own explicit topology, placement, fluid, load, stage/port and search controls;
- prevent environment defaults from influencing the oracle;
- provide stable IDs and public service selection;
- keep one restart and the chosen hard budget visible to assertions.

## Assignment builder

**Inputs**: sorted corpus and ordered profile catalogue.

**Output**: immutable benchmark assignments.

**Responsibilities**:

- apply ordinal-modulo assignment;
- build readable parameter IDs;
- prove complete one-to-one corpus coverage;
- prove all six profiles are represented and currently receive nine cases.

## Problem factory and state snapshot

**Inputs**: one assignment.

**Output**: a fresh public problem and immutable pre-call snapshot.

**Responsibilities**:

- load inside the parameter invocation, never at module collection;
- derive any stable project name from the filename;
- capture public result JSON and target count;
- prevent cross-case mutable state and cache reuse.

## Public HPR runner

**Inputs**: fresh problem and profile.

**Output**: solved, no-op or typed-failure observation.

**Responsibilities**:

- invoke the selected real public target method with explicit arguments;
- catch only `HPRTargetingError`;
- preserve unexpected exception types and tracebacks;
- expose the returned target or typed diagnostic without rewriting it;
- never retry or mutate an outcome after the call.

## Outcome assertions

**Inputs**: pre-state, post-state and observed outcome.

**Output**: assertion success or a case/profile-specific pytest failure.

**Responsibilities**:

- validate atomic no-op/failure and exactly-one-target success;
- validate strict success, finite accounting, detached CoolProp records and
  serialization;
- validate bounded typed diagnostics and budget accounting;
- avoid private optimizer-coordinate and global-feasibility assertions.

## Search observer

**Inputs**: the original candidate evaluator and one sentinel invocation.

**Output**: ordered detached search observations.

**Responsibilities**:

- install and restore a temporary wrapper using pytest cleanup guarantees;
- delegate exactly once with unchanged arguments;
- record only search-mode evaluations after the real result exists;
- retain exact design-point tuples plus finite viable objectives;
- make no candidate, cache, objective or result-selection decision.

Sequential execution is part of this component's contract because the wrapped
boundary is shared process state.

## Convergence analyzer

**Inputs**: ordered observations, selected objective and configured allowance.

**Output**: immutable convergence witness or an assertion explaining the first
failed criterion.

**Responsibilities**:

- deduplicate exact points in first-seen order;
- derive viable values and non-increasing incumbent bests;
- apply the scale-aware `1e-8` improvement tolerance;
- compare the final target with the best viable objective observed;
- enforce at least two viable points and the hard distinct-evaluation budget;
- describe the result as bounded best-observed progress, never global proof.

This pure component is the primary target for generated convergence properties.

## Strict-sentinel registry

**Inputs**: construction-time exploratory evidence.

**Output**: one explicit stable problem identifier for each profile.

**Responsibilities**:

- make six required successes reviewable and deterministic;
- fail rather than skip if a sentinel no longer solves or converges;
- prevent arbitrary solved matrix cases from masking profile regression.

The registry is committed test data, not learned dynamically during a test run.

## Direct process-MVR scenario

**Inputs**: packaged `process_mvr.json` and explicit compression settings.

**Output**: validated MVR component, replacement-stream inventory and downstream
direct-integration result.

**Responsibilities**:

- use the public component and targeting accessors;
- validate finite positive physical outputs and pressure lift;
- validate atomic registration and engine detachment;
- validate downstream target completion and serialization.

## Investigation evidence

**Inputs**: full exploratory matrix outcomes.

**Output**: development evidence for profile counts, sentinels, selected budget
and any defects.

**Responsibilities**:

- record evidence in the Code Generation summary rather than creating volatile
  runtime snapshots;
- distinguish solved, no-op, typed failure and unexpected failure;
- route unexpected failures to focused regressions and production owners;
- avoid freezing environment-specific feasibility counts as product promises.

## Interaction sequence

1. Corpus discovery and the profile catalogue feed the assignment builder.
2. Each assignment asks the factory for a fresh problem and snapshot.
3. The public runner performs one bounded call.
4. Outcome assertions validate public state and contracts.
5. For six named sentinels only, the search observer supplies evidence to the
   convergence analyzer before strict-success assertions complete.
6. The separate process-MVR scenario exercises its public component workflow
   and downstream target.
7. Unexpected failures leave the harness and are corrected at their production
   owner with focused regression coverage.

## Performance and failure allocation

| Concern | Owner | Enforcement |
|---|---|---|
| Discovery/assignment complexity | Corpus and assignment builder | O(n), one immutable parameter per input. |
| Search cost | Profile catalogue and public service | One restart, smallest stable allowance, maximum 50 distinct evaluations. |
| Cache accounting | Search observer and existing focused properties | Exact duplicate points are excluded from distinct-count budget. |
| Cross-test isolation | Problem factory and pytest execution policy | Fresh instances and sequential module execution. |
| Expected infeasibility | Public runner and outcome assertions | Only bounded `HPRTargetingError` is accepted. |
| Unexpected implementation failure | pytest plus production owner | Original exception propagates; focused reproduction precedes correction. |
| Convergence | Search observer and convergence analyzer | Real trace plus independent best-observed oracle. |
| Flakiness | Declarative inputs and no retries | Deterministic ordering, explicit options, no timing oracle. |

## Infrastructure decision

No queue, database, service, worker pool, distributed cache, external solver,
network endpoint, deployment resource or new CI job is applicable. The existing
ordinary non-solver pytest job is the execution environment. Infrastructure
Design remains skipped.

## Property-Based Testing allocation

| Rule area | Design disposition |
|---|---|
| Property identification | Assignment, convergence, state-transition, cache, budget and serialization properties are explicit. |
| Generators and constraints | Generate finite objectives, exact-point duplicates, selected-objective offsets and corpus sizes; exclude NaN/inf unless testing rejection. |
| Oracles | Pure reference folds compute deduplication, incumbent minima and tolerance decisions independently. |
| Stateful behavior | Existing HPR cache/search model properties remain; e2e cases use fresh state. |
| Shrinking and reproducibility | Hypothesis defaults plus seed `20260715`; no suppression or disabled shrinking. |
| Complementary examples | All 54 real assignments, six real convergence sentinels and one real direct-MVR scenario. |

PBT requirements are fully allocated with no blocking finding. Security and
Resiliency extensions remain disabled and are N/A at this stage.
