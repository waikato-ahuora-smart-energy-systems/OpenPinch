# Unit 2 HPR Point Simulators and Map Generation Functional Design Plan

## Scope

Design the business logic for deterministic HPR map generation through a
default CoolProp point simulator and an explicitly selected optional TESPy
leaf. Unit 2 consumes the approved Unit 1 contract and does not change the
public target accessor, targeting signatures, or return models owned by Unit 3.

## Plan

- [x] Read the Unit 2 definition, requirement map, Application Design, current
  vapour-compression implementation, existing TESPy isolation pattern, and Unit
  1 contract.
- [x] Identify the unresolved physical-model and lifecycle choices that can
  change generated duties, power, COP, and provenance.
- [x] Create the dedicated Functional Design question file with complete
  mutually exclusive choices and answer tags.
- [x] Validate every user response and resolve any ambiguity or contradiction.
- [x] Define the generation context, operating point, normalized simulation,
  diagnostic, simulator protocol, factory, and exception relationships.
- [x] Define deterministic Cartesian traversal, design/offdesign lifecycle,
  duty/power normalization, COP calculation, provenance assembly, and atomic
  map construction.
- [x] Define the default CoolProp simulator behavior and its nominal-result
  comparison oracle.
- [x] Define the optional TESPy topology, characteristic ownership,
  design/offdesign behavior, convergence criteria, extraction, and cleanup.
- [x] Define all validation, unsupported-context, missing-dependency,
  convergence, partial-failure, and cleanup rules.
- [x] Identify complementary examples and PBT properties for traversal size and
  order, physical invariants, repeatability, simulator lifecycle, failure
  atomicity, and the CoolProp comparison oracle.
- [x] Create and validate `business-logic-model.md`, `business-rules.md`, and
  `domain-entities.md` under the Unit 2 Functional Design directory.
- [x] Update plan and stage tracking and request explicit Functional Design
  approval before Unit 2 NFR Requirements.
- [x] Amend the design after review to preserve current single-loop CoolProp
  support for named refrigerant blends and explicit mole-fraction mixtures of
  arbitrary component count without a Unit 2 refrigerant allowlist.
- [x] Define engine-neutral working-fluid normalization, dew/bubble saturation
  anchors, TESPy syntax translation, fail-closed compatibility, provenance, and
  complementary example/PBT coverage for mixtures.
- [x] Require the TESPy adapter to accept the same pure, registered-blend, and
  explicit N-component mixture input domain without categorical mixture or
  component-count rejection, while reporting actual property-wrapper or state
  limitations as typed backend failures.

## Boundaries

- CoolProp remains the installed default; TESPy is imported only by its concrete
  optional adapter.
- Unit 2 produces only Unit 1 plain-data maps and internal typed diagnostics.
- No OpenUtility, Pyomo, HiGHS, investment, dispatch, interpolation constraint,
  public accessor, target mutation, database, UI, or infrastructure behavior is
  designed here.
- Analytic Carnot, existing Brayton, MVR, cascade, parallel multi-port, and
  partial-map behavior remain excluded from the first release.

## Content Validation

This plan contains no Mermaid, ASCII diagram, executable code block, embedded
JSON, or YAML. Headings, lists, paths, and checkbox syntax were validated before
creation.
