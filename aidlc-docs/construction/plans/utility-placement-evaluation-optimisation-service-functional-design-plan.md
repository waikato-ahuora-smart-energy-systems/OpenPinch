# Unit 2 Placement Evaluation and Optimisation Service Functional Design Plan

## Purpose

Define the detailed business logic, domain entities, equations, feasibility
rules, failure boundaries, and testable properties for the detached numerical
service before NFR design or TDD implementation begins.

## Plan Progress

- [x] Load the approved Unit 2 definition, story/requirement map, requirements,
  Application Design, Unit 1 functional design, and implemented contracts.
- [x] Inspect the existing direct/Total Site targeting, utility allocation,
  solver-neutral optimisation, and multi-stage steam-turbine boundaries.
- [x] Evaluate all PBT-01 property categories for the Unit 2 components.
- [x] Identify unresolved functional decisions that materially affect equations,
  domain boundaries, errors, or test oracles.
- [x] Create the questions below using the mandatory question format.
- [x] Collect and validate every answer; create focused clarification questions
  if any response is missing, invalid, contradictory, or ambiguous.
- [x] Generate `business-logic-model.md` with context preparation, candidate
  replay, coverage, entropy, monetary/cogeneration, aggregation, penalty,
  optimisation, alternatives, and result-assembly flows.
- [x] Generate `business-rules.md` with equations, units, precedence,
  tolerances, feasibility, deterministic ordering, and typed failures.
- [x] Generate `domain-entities.md` with detached Unit 2 context, allocation,
  evaluation, memoization, turbine-adapter, and result relationships.
- [x] Add a PBT-01 Testable Properties section assigning round-trip, invariant,
  idempotence, commutativity, oracle, induction, and easy-verification status to
  every Unit 2 component.
- [x] Validate Markdown/code-block syntax, traceability, PBT-01 compliance, and
  consistency with the approved Unit 1 contract and no-CLI/single-notebook
  scope.
- [x] Present the standardized Functional Design completion checkpoint and wait
  for explicit approval.

## Questions

Please answer every question by entering one option letter after its
`[Answer]:` tag. Choose the final Other option only when the listed choices do
not express the required behavior, and add the intended behavior after the tag.

## Question 1

How should Unit 2 resolve the gap between FR-011 and the implemented
`UtilityPlacementOptions`, which currently lacks the designed optimizer method,
run count, local method, clustering tolerance, and backend override fields?

A) Extend the frozen specialist options contract test-first before Unit 2
orchestration, default to the existing dual-annealing service, and validate all
method-specific overrides through the existing optimizer (recommended)

B) Add only an optimizer-method field and map all other execution controls from
the existing OpenPinch configuration

C) Fix Unit 2 to dual annealing for the first release and defer method/backend
overrides, accepting a documented FR-011 scope change

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 2

Which detached data boundary should drive candidate duty allocation in each
period?

A) Copy the selected zone, reuse its existing shifted and real problem tables
plus utility-targeting cascade, then extract immutable allocation intervals and
results (recommended)

B) Extract numerical profile arrays first and reconstruct a standalone
problem-table calculation inside the placement service

C) Rebuild duty allocation as a new pure algorithm without calling the existing
utility-targeting cascade

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 3

When multiple utility levels contribute heat within the same process
temperature interval, how should process-side entropy be attributed?

A) Follow the existing cascade's deterministic utility assignment and split
each level's assigned duty across the real-temperature intervals it actually
serves (recommended)

B) Allocate the interval process entropy among active levels in proportion to
their total period duties

C) Compute one aggregate process-side entropy term per hot/cold side without
attributing it to individual levels

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 4

Which objective breakdowns should a successful period calculate and retain?

A) Always calculate thermodynamic entropy/exergy; additionally calculate the
monetary/cogeneration breakdown only for monetary requests (recommended)

B) Calculate only the selected objective's breakdown

C) Always calculate both breakdowns and therefore require complete economic
inputs even for thermodynamic requests

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 5

How should eligible hot utility levels obtain the non-placement inputs required
by the existing multi-stage steam-turbine calculation?

A) Use explicit placement arguments when supplied and otherwise inherit the
existing problem power/turbine configuration; candidate temperatures and
duties remain the only optimized turbine inputs (recommended)

B) Require every cogeneration-eligible template to carry all inlet pressure,
inlet temperature, turbine model, efficiency, and stage settings

C) Use one fixed first-release turbine configuration independent of the
problem configuration

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 6

How should failures from the turbine calculation be classified after structural
request validation has passed?

A) Treat temperature/duty-dependent turbine incompatibility as ordinary
candidate infeasibility, but raise a typed run-level error for configuration or
adapter failures that no candidate can correct (recommended)

B) Treat every turbine failure as ordinary candidate infeasibility

C) Abort the complete optimisation on the first turbine failure

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 7

What scalar penalty behavior should guide the black-box optimizer when a
candidate violates ordering, coverage, targeting, thermodynamic, or turbine
constraints?

A) Use a deterministic finite base penalty plus normalized violation magnitude,
while independently filtering all infeasible evaluations before final ranking
(recommended)

B) Use one flat deterministic finite penalty for every infeasible candidate

C) Raise from the objective callback and rely on backend error handling

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 8

How should repeated and near-duplicate optimizer evaluations be memoized and
deduplicated within one service run?

A) Memoize evaluations by the exact normalized coordinate tuple; deduplicate
final feasible alternatives using the existing optimizer's clustering result,
then order by objective and exact coordinates (recommended)

B) Quantize coordinates by placement tolerances for both memoization and final
deduplication

C) Do not memoize; re-evaluate every callback and retain every returned
candidate before applying the candidate limit

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 9

Which oracle should be mandatory for bounded analytical optimisation cases?

A) A deterministic structured grid over tiny one-period models, comparing the
service's best feasible objective with the grid minimum within a declared grid
resolution/tolerance; test direct and Total Site adapters separately
(recommended)

B) Exhaustive enumeration only for fixed-coordinate or explicitly discrete
test models

C) Analytical stationary-point formulas only, without a generated grid oracle

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Preliminary PBT-01 Category Assessment

The following categories must be finalized after the answers:

| Component | Applicable categories |
|---|---|
| Context preparation | Invariant, idempotence, commutativity where period ordering is intentionally canonical |
| Candidate replay and coverage | Invariant, easy verification, induction across periods/levels |
| Thermodynamic evaluator | Invariant, oracle, commutativity of independent branch summation |
| Monetary/cogeneration evaluator | Invariant, oracle, commutativity of independent purchase-cost terms |
| Weighted aggregation | Invariant, oracle, induction, commutativity only under explicitly order-insensitive summation comparison |
| Penalty and feasibility separation | Invariant, easy verification |
| Per-run evaluation memo | Idempotence and possibly stateful model testing, depending on Question 8 |
| Optimisation coordinator | Oracle, easy verification, reproducibility invariant |
| Candidate normalization/result assembly | Invariant, round-trip through the existing Unit 1 result contract |

PBT-01 remains pending until the final artifacts name exact properties and mark
non-applicable categories with rationale. PBT-02 through PBT-10 are not
blocking at Functional Design, but their future obligations must be carried
into Code Generation planning.
