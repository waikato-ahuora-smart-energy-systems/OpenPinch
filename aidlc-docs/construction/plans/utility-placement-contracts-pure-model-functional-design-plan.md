# Utility Placement Contracts and Pure Model Functional Design Plan

## Unit Context

Unit 1 establishes the specialist contracts and deterministic pure model used
by the numerical service and public integration units. It owns UPO-01, the
contract portions of UPO-08 and UPO-09, and UPO-12 foundations; FR-002 through
FR-006, typed portions of FR-012 and FR-014; and relevant NFR-001, NFR-002,
NFR-004, NFR-005, and NFR-006 behavior.

This is a brownfield, in-process Python module. It introduces no persistence,
network, UI, independent deployment, package-root export, or optimiser call.
Frontend Components are N/A because the unit is a Python contract and pure-
transformation boundary with no frontend.

## Plan Steps

- [x] Load and reconcile the approved Unit 1 definition, story map,
  requirements, Application Design, and enabled PBT rules.
- [x] Collect and analyze every functional-design answer below; add follow-up
  questions for any vague, combined, missing, or contradictory response.
- [x] Obtain explicit approval of the answered Functional Design plan.
- [x] Generate
  `aidlc-docs/construction/utility-placement-contracts-pure-model/functional-design/business-logic-model.md`.
- [x] Generate
  `aidlc-docs/construction/utility-placement-contracts-pure-model/functional-design/business-rules.md`.
- [x] Generate
  `aidlc-docs/construction/utility-placement-contracts-pure-model/functional-design/domain-entities.md`.
- [x] Document PBT-01 Testable Properties per component, covering every
  applicable property category and explicit N/A rationale.
- [x] Validate artifact consistency, complete Unit 1 requirement/story
  ownership, and PBT-01 compliance.
- [x] Obtain explicit approval before Unit 1 NFR Requirements.

## Planned Functional Design Scope

- contract relationships and lifecycle from public arguments to normalized
  request, template set, placement model, candidate/result, and JSON;
- deterministic generated-template behavior and caller-template validation;
- count, identity, side, kind, units, direction, bounds, price, eligibility,
  span, ordering, and separation rules;
- all-period feasibility-envelope input, bound intersection, starting points,
  and empty-region classification;
- decision-vector dimension, coordinate order, encode/decode, and candidate
  verification;
- typed request/model exceptions versus structured candidate diagnostics;
- PBT-01 round-trip, invariant, idempotence, oracle, and easy-verification
  properties, with commutativity and induction applicability assessed.

## Question 1 - Generated Templates

When callers provide counts but omit templates, how should deterministic
templates be generated?

A) Generate stable hot/cold identities from side, kind, and ordinal; derive
temperature bounds and starting supplies from the feasibility envelope; fix
isothermal spans to 0.01 degrees Celsius by default; give sensible spans valid
derived bounds; default cogeneration eligibility to false; and leave monetary
prices absent so monetary requests still require explicit economics

B) Require every template explicitly in the first release and reject count-only
requests

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 2 - Units at the Contract Boundary

How should temperature, duty, price, entropy, and monetary values be represented?

A) Accept established OpenPinch scalar-or-value-with-unit inputs, normalize to
documented canonical numerical units inside the pure model, and return explicit
unit metadata on result values

B) Accept only bare floats in fixed units for the first release

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 3 - Template Identity and Coordinate Order

What ordering should remain stable through validation and vector round-trips?

A) Preserve caller declaration order within each side/kind family and encode in
the fixed sequence hot isothermal, hot sensible, cold isothermal, cold sensible;
enforce solved physical hot-descending/cold-ascending order as candidate
constraints without silently reordering or merging identities

B) Sort templates automatically by bounds or solved temperature and allow
identities to move during normalization

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 4 - Normalization and Immutability

What contract mutation policy should apply?

A) Use frozen, extra-forbid specialist Pydantic contracts; normalize once into
new detached values; make normalization idempotent; and never retain caller
lists, mappings, mutable streams, callables, or backend objects

B) Use mutable contracts consistent with ordinary Python containers and rely
on callers not to mutate them

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 5 - Feasibility-Envelope Boundary

How should Unit 1 receive process-derived feasible temperature information
without depending on Unit 2?

A) Unit 1 defines a detached `PlacementFeasibilityEnvelope` contract containing
ordered period identities, per-template physical bounds, approach/separation
limits, and units; Unit 2 later populates it from isolated targeting context

B) Unit 1 imports the Unit 2 context builder directly and derives bounds itself

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 6 - Error Taxonomy

How should Unit 1 expose validation and pure-model failures?

A) Define one specialist `UtilityPlacementError` root with typed subclasses for
request validation, template validation, units, and empty feasible bounds;
include stable field/template/period context while preserving conventional
`ValueError` compatibility for invalid caller data where practical

B) Raise only generic `ValueError` messages with no specialist hierarchy

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 7 - Boundary Scenarios

Which boundary behavior should be normative for the pure model?

A) Accept exactly two isothermal and zero sensible levels per side; reject
booleans as counts; preserve unused or zero-duty level identities; require
strictly positive absolute temperatures and separation; and reject an empty
all-period bound intersection before optimisation

B) Relax one or more boundaries and describe the exact alternative after the
`[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Question 8 - Floating-Point Round Trips

What equality rule should apply to normalized and serialized contracts?

A) Require exact equality for enums, identities, collection order, counts, and
unit labels; require finite floats and compare calculated/converted floats
within named absolute and relative tolerances, preserving signed-zero-neutral
semantics

B) Require exact bitwise equality for every floating-point value

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)
