# HPR and MVR Notebook Reliability NFR Requirements

## Scope

These requirements apply to generated notebooks 08 through 11, their generator,
tutorial metadata and guidance, and the tests that execute or inspect them.
They do not introduce a new production HPR/MVR API or optimization algorithm.

## Performance and Work Boundedness

### NFR-PERF-001: Explicit optimizer caps

Every optimizer-backed HPR or MVR call in the four notebooks shall pass public
work-limit arguments explicitly. Required Carnot, CoolProp VC, refrigeration,
and VC+MVR examples shall use:

- `maximum_restarts=1`;
- `maximum_iterations` no greater than 20; and
- `maximum_evaluations` no greater than 50.

An optional advanced screen may use smaller limits. No notebook call may inherit
the 300-iteration or 1,000,000-evaluation defaults.

### NFR-PERF-002: Explicit topology bounds

Every required simulated VC or VC+MVR call shall explicitly request one
condenser, one evaporator, and one MVR stage where applicable. Its fluid list
shall also be explicit. Tutorial success shall not depend on inherited cascade
size or refrigerant configuration.

### NFR-PERF-003: Algorithmic limit is the blocking runtime gate

Optimizer budgets and per-notebook completion are blocking. Wall-clock elapsed
time shall be recorded for the four real notebook executions, but shall not be
asserted as a hard cross-platform threshold. On the reference development
environment, each notebook should complete within 60 seconds and the four-
notebook profile should complete within 180 seconds; an exceedance requires
investigation but is not by itself a test failure when all hard work limits and
results are correct.

### NFR-PERF-004: No repeated shared optimization

Each Notebook 10 technology shall invoke one scalar targeting method once. The
notebook shall not wrap a shared-design call in `target.all_periods`, a loop over
periods, or any other construct that starts the same shared optimization more
than once per technology.

## Reliability and Correctness

### NFR-REL-001: Required paths fail loudly

Required examples shall not be protected by a catch-and-continue helper.
Notebook 08's two Carnot targets, Notebook 09's CoolProp heat-pump and
refrigeration targets, Notebook 10's five shared-design targets, and Notebook
11's direct process-MVR and optimized VC+MVR paths shall assert their required
success evidence. A failure in any of these paths shall fail that notebook's
execution test.

### NFR-REL-002: Shared-design completeness

Every Notebook 10 main result shall expose:

- a non-empty finite shared design vector;
- exactly the period identifiers `turndown`, `base`, and `peak` as a set;
- one successful period output for every identifier;
- period weights aligned with the ordered outputs; and
- a finite weighted result or objective.

The five technologies remain five independent optimizations; no ranking is to
be presented as a joint technology-selection optimization.

### NFR-REL-003: Notebook-specific proof

- Notebook 08 shall preserve valid downstream residual targeting, utility
  placement, summaries, and plots from the successful heat-pump result.
- Notebook 09 shall preserve a detached target simulation record and a valid
  JSON-mode performance map, and its required refrigeration target shall
  succeed.
- Notebook 11 shall expose non-empty direct process-MVR stage results, finite
  compressor work, equivalent serial and parallel multiperiod results, and
  non-empty detached VC+MVR loop records.

### NFR-REL-004: Optional outcomes remain distinct

Notebook 09 shall distinguish at least:

- `feasible`;
- `optional dependency unavailable`;
- `method unavailable`; and
- `typed infeasible`.

Notebook 10's optional advanced screen may use `typed infeasible`. Optional
outcomes shall not be reported as required successes and shall not trigger a
silent backend or technology fallback.

### NFR-REL-005: Unexpected defects propagate

Notebook helpers may catch only the declared typed HPR failure and, where
applicable, the explicit optional dependency or unavailable-method exceptions.
Serialization, programming, contract, and lifecycle errors shall propagate and
fail the notebook execution.

## Diagnostics and Output Boundedness

### NFR-DIAG-001: Plain detached diagnostics

Typed HPR failures shall be converted with the existing Pydantic
`model_dump(mode="json")` contract. The resulting structure shall round-trip
through standard JSON and shall contain no live CoolProp/TESPy object, NumPy
array, exception instance, stream collection, or target object.

### NFR-DIAG-002: Existing schema bounds are preserved

Failure presentation shall preserve the public diagnostic bounds:

- no more than 16 representative failures;
- reason codes no longer than 128 characters;
- summaries no longer than 512 characters and restricted to one line; and
- fluid and period identifiers no longer than 256 characters.

The notebook shall not append an unbounded raw candidate log.

### NFR-DIAG-003: Compact successful output

Review cells shall display plain compact summaries, not `TargetOutput` objects
or dictionaries containing them. A success summary shall contain only the
fields needed to prove status, topology/backend, selected and achieved duty,
objective or weighted result, period identifiers, and a bounded design-vector
or stage/loop summary.

## Determinism and Reproducibility

### NFR-DET-001: Generator authority and idempotence

`scripts/generate_tutorial_notebooks.py` remains authoritative. Two consecutive
generation passes with the same repository state shall produce byte-identical
notebooks. Checked-in notebooks 08 through 11 shall match the generator.

### NFR-DET-002: Source-only notebooks

Every generated code cell shall have `execution_count` set to null and an empty
`outputs` list. Regeneration shall remove Notebook 10's current local execution
metadata and saved output as explicitly approved.

### NFR-DET-003: Reproducible randomized tests

New Hypothesis-based tests shall use the repository's existing profile and the
project seed `20260715` wherever explicit seeding is supported. Strategies shall
generate only contract-valid, size-bounded diagnostic and period structures.

## Test Attribution and Maintainability

### NFR-TEST-001: One execution case per notebook

The slow-HPR execution test shall expose notebooks 08, 09, 10, and 11 as four
distinct pytest cases. A stall or failure in one case shall not erase the
completed result or identity of another.

### NFR-TEST-002: Layered verification

Verification shall include:

1. static source and argument-contract tests;
2. generator idempotence, source-only, and checked-in drift tests;
3. focused example-based outcome tests;
4. applicable property-based helper/serialization invariants;
5. four independent real notebook executions;
6. the combined slow-HPR profile;
7. related HPR/MVR, packaging, tutorial-coverage, and documentation tests; and
8. broad regression and lint gates defined in Build and Test.

### NFR-MAINT-001: Existing ownership boundaries

Notebook behavior shall be authored in the generator and tested through public
workflow APIs. No private analysis import, hidden notebook execution framework,
or duplicate production targeting service shall be introduced.

### NFR-MAINT-002: Synchronized learning metadata

Notebook metadata, tutorial coverage rows, notebook-series text, and heat-pump
workflow guidance shall describe the actual generated calls. Notebook 10 text
shall explicitly distinguish shared-design optimization from independent
`target.all_periods` replay.

## Scalability, Availability, Security, and Operations

### NFR-SCALE-001: Bounded fixed tutorial workload

The tutorial set is a fixed four-notebook local workload. Scaling is controlled
by explicit optimizer budgets and bounded period/topology sizes; horizontal
scaling, distributed execution, and capacity planning are N/A.

### NFR-AVAIL-001: Availability is N/A

There is no server, persistent daemon, recovery-point objective, recovery-time
objective, or uptime commitment. Failures are local, attributable pytest or
notebook outcomes.

### NFR-SEC-001: Security extension disabled

The Security Baseline is disabled by user decision. The notebooks add no
network access, credentials, authentication, authorization, secret storage, or
untrusted execution surface. Security-specific design is N/A.

### NFR-OPS-001: Operations is N/A

No deployment, publishing, monitoring, alerting, or infrastructure work is in
scope.

## Property-Based Testing Compliance

- **PBT-01**: properties cover plain diagnostics, status vocabulary, bounded
  collections, JSON round trips, and period-set invariants.
- **PBT-02**: generated valid diagnostic summaries round-trip through JSON.
- **PBT-03**: generated inputs vary valid identifiers, messages, counts,
  categories, and ordering within contract bounds.
- **PBT-04**: pure diagnostic normalization is idempotent if introduced;
  otherwise N/A with no normalization helper.
- **PBT-05**: compact summaries are compared with their source result contracts.
- **PBT-06**: N/A unless stateful helper logic is introduced.
- **PBT-07**: strategies remain contract-valid and size-bounded.
- **PBT-08**: reproducibility uses seed `20260715` where supported.
- **PBT-09**: Hypothesis remains the only property-testing framework.
- **PBT-10**: real example tests remain mandatory alongside properties.

## Stage Extension Compliance

- **Property-Based Testing**: compliant. Applicable requirements are explicit
  and blocking; conditional stateful/metamorphic rules include N/A criteria.
- **Security Baseline**: disabled and N/A.
- **Resiliency Baseline**: disabled and N/A.
