# Unit 1 HPR Performance-Map Contract Functional Design Plan

## Scope

Design the technology-independent business rules for the strict alpha schema
`1.0` request, units, point, map, JSON Schema, golden fixtures, validation
errors, and serialization boundary. Thermodynamic simulation and public target
API behavior belong to later units.

## Question Assessment

No unanswered `[Answer]:` questions are required. The approved requirements,
Application Design, corrected OpenUtility consumer, and Units Generation
approval settle every functional-design category:

- **Business logic modeling**: strict construction, cross-field validation,
  deterministic ordering, serialization, and schema/fixture publication.
- **Domain model**: immutable request, units, point, map, JSON value, and typed
  error concepts are fixed.
- **Business rules**: version, units, mode, capacity, COP, energy balance,
  curve topology, ordering, uniqueness, and provenance rules are explicit.
- **Data flow**: validated in-memory values serialize to plain JSON, JSON Schema,
  and checked-in fixture resources; there is no persistence workflow.
- **Integration points**: Unit 2 produces the contract, Unit 3 exposes it, and
  OpenUtility consumes fixture data without Python imports.
- **Error handling**: invalid payloads fail closed before serialization or
  downstream use; partial and coerced payloads are prohibited.
- **Business scenarios**: heat-pump and refrigeration capacity/COP conventions,
  multiple fixed-temperature curves, at least three part-load breakpoints, and
  malformed external payloads are covered.
- **Frontend components**: N/A because this unit has no UI.

## Execution Steps

- [x] Read the approved unit definition, dependency matrix, requirements map,
  Application Design, and corrected OpenUtility field semantics.
- [x] Evaluate all functional-design question categories and confirm that the
  approved inputs leave no ambiguity requiring user answers.
- [x] Create the business logic model for construction, validation,
  deterministic serialization, JSON Schema, and golden-fixture generation.
- [x] Define all field-level, cross-field, curve-level, map-level, provenance,
  and compatibility business rules.
- [x] Define domain entities, value relationships, ownership, and lifecycle.
- [x] Identify testable properties under PBT-01 and assign round-trip,
  invariant, oracle, generator-quality, reproducibility, and complementary
  example obligations.
- [x] Validate Markdown syntax, tables, code fences, traceability, unit
  boundaries, and enabled-extension compliance.
- [x] Obtain explicit approval before Unit 1 NFR Requirements.

## Content Validation

The plan contains no Mermaid, ASCII diagrams, embedded JSON/YAML, or executable
code blocks. Markdown headings, lists, inline code, and checkbox syntax were
checked before file creation.
