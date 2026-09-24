# Unit 1 Functional Design Plan

Unit: CoolProp HPR and MVR — Candidate Correctness and Detached HPR Results

Status: Functional Design complete; artifacts validated and awaiting review.

## Purpose

Define the detailed, technology-agnostic behavior for penalty normalization,
search-time versus final-time evaluation, accepted-result detachment, generalized
simulation evidence, and the foundational budget/failure contracts consumed by
later units.

## Approved Boundary

Unit 1 owns:

- penalty and finite-objective semantics;
- search-mode and final-mode result responsibilities;
- detached public output and recursive multiperiod detachment;
- generalized `HprTargetSimulationRecord`;
- foundational `HPRSearchBudget`, diagnostic, and
  `HPRTargetingError(ValueError)` contracts.

Unit 1 does not own CoolProp capability preflight, global-search orchestration,
public budget parameters, or direct process-MVR stage behavior.

## Functional-Design Category Assessment

- **Business Logic Modeling**: applicable. Candidate values pass through
  normalization, search evaluation, final evaluation, finalization, and public
  transaction states.
- **Domain Model**: applicable. Budget, failure diagnostic/summary, targeting
  error, generalized simulation record, and evaluation-mode concepts require
  exact relationships.
- **Business Rules**: applicable. Penalty shapes, finite values, success
  equivalence, detachment, and compatibility require closed rules.
- **Data Flow**: applicable for in-memory candidate/result transformation.
  Persistence is N/A because no database, file store, or durable state is added.
- **Integration Points**: applicable at the topology-objective, reusable
  optimizer, multiperiod, HPR service, and application-transaction seams.
  External/network integration is N/A.
- **Error Handling**: applicable. Candidate-local physical failures must remain
  distinct from fatal shape, contract, lifecycle, and detachment defects.
- **Business Scenarios**: applicable for empty/mixed penalties, final
  reevaluation divergence, nested period outputs, and compatibility callers.
- **Frontend Components**: N/A. Unit 1 has no UI or frontend surface.

## Plan Checklist

- [x] Read the approved Unit 1 definition and requirement map.
- [x] Read the approved Application Design and source-owner boundaries.
- [x] Evaluate every functional-design question category.
- [x] Identify six unresolved detailed behavior decisions.
- [x] Create context-specific questions using the required answer format.
- [x] Collect all six answers.
- [x] Analyze every answer for ambiguity, combined choices, or contradiction.
- [x] Add and resolve follow-up questions if required (none were needed).

## Resolved Decisions

1. Accept numeric scalar or rectangular array-like penalty inputs, flatten them
   in stable row-major order, treat empty values as zero terms, and reject
   ragged, non-numeric, NaN, or infinite inputs as fatal contract errors.
2. Generalize the existing simulation record with a topology identifier and
   ordered detached loop/stage records while retaining compatible single-stage
   top-level fields.
3. On final reevaluation, continue only after a newly classified physical
   infeasibility; artifact, contract, lifecycle, and detachment failures remain
   fatal.
4. Enforce recursive engine-object exclusion, canonical simulation evidence,
   `model=None`, and actual deep-copy safety at the transaction boundary.
5. Use a closed failure-category enum, stable reason codes, and bounded
   sanitized summaries without public raw engine messages or exception objects.
6. Require positive non-boolean integer budgets with defaults of 300 iterations
   and 1,000,000 evaluations.

All six answers are explicit A selections. They are mutually consistent and do
not move Unit 2 or Unit 3 behavior into Unit 1.
- [x] Generate
  `aidlc-docs/construction/coolprop-hpr-mvr-unit-1/functional-design/business-logic-model.md`.
- [x] Generate
  `aidlc-docs/construction/coolprop-hpr-mvr-unit-1/functional-design/business-rules.md`.
- [x] Generate
  `aidlc-docs/construction/coolprop-hpr-mvr-unit-1/functional-design/domain-entities.md`.
- [x] Document PBT-01 property categories and downstream PBT-02 through PBT-10
  obligations.
- [x] Validate completeness, cross-artifact consistency, Markdown, and links.
- [x] Present the standardized Functional Design review gate.

## Question 1 — Penalty Normalization

Which input policy should the shared penalty normalizer enforce?

A) Accept numeric scalars and rectangular numeric array-like values, flatten in
stable row-major order, treat an empty value as zero terms, and reject ragged,
non-numeric, NaN, or infinite values as fatal contract errors (recommended)

B) Accept only a scalar or already one-dimensional numeric sequence and reject
all higher-dimensional arrays

C) Preserve nested groups and aggregate each group separately before the shared
penalty calculation

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 2 — Generalized Simulation Evidence

How should the existing simulation record represent cascade, parallel, and
VC+MVR accepted designs?

A) Add an explicit topology identifier and ordered detached loop/stage records,
while retaining current single-stage top-level fields for backward compatibility
(recommended)

B) Keep the schema unchanged and place all topology-specific evidence in the
free-form `assumptions` mapping

C) Introduce separate public simulation-record models for each topology

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 3 — Final Reevaluation Divergence

What should happen when a point was viable in search mode but final evaluation
does not produce a publishable result?

A) Continue to the next ranked candidate only for a newly classified physical
infeasibility; propagate artifact, contract, lifecycle, or detachment failures
as fatal defects (recommended)

B) Treat every final-evaluation failure as candidate-local and try the next point

C) Abort on every final-evaluation failure, including physical infeasibility

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 4 — Detachment Proof

What must the finalizer prove before returning a public HPR output?

A) Enforce a recursive no-engine-object rule, require canonical detached
simulation evidence, set `model=None`, and complete an actual deep-copy check at
the transaction boundary (recommended)

B) Rely only on Pydantic validation and setting `model=None`

C) Require the complete public target to serialize to plain JSON in addition to
deep-copy safety

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 5 — Failure Vocabulary

How should Unit 1 define the diagnostic vocabulary later used by optimized HPR
and direct process-MVR?

A) Use a closed category enum plus stable reason codes and bounded sanitized
summaries; engine-specific raw messages and exception objects remain internal
(recommended)

B) Use free-form category and reason strings to simplify extension

C) Store the original exception object in each diagnostic for debugging

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 6 — Budget Contract

What validation/default policy should the foundational search-budget contract
use?

A) Require positive non-boolean integers, with bounded defaults matching the
current reusable optimizer limits (`maxiter=300`, `maxfun=1,000,000`);
Unit 2 may override them only through validated public inputs (recommended)

B) Require positive integers but leave both defaults undefined until Unit 2

C) Permit zero as a valid value meaning that global search is disabled

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Answer Review Criteria

- Every `[Answer]:` line is populated.
- Expanded answers define an exact rule rather than “depends” or mixed options.
- Decisions remain compatible with the approved Application Design.
- No Unit 2 preflight/search policy or Unit 3 process-MVR behavior is pulled into
  Unit 1.
