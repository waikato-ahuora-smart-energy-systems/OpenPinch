# Unit 3 Current HPR Target API Integration and Publication Functional Design Plan

## Scope

Design the public application behavior that connects the existing
vapour-compression heat-pump and refrigeration targeting methods to the Unit 2
CoolProp-default, TESPy-selectable performance-map generator. Unit 3 also owns
detached target-basis extraction, the explicit map follow-up operation, public
documentation, and final compatibility boundaries.

## Plan

- [x] Read the Unit 3 definition, requirement map, Application Design, current
  target accessor, HPR target/result contracts, and completed Unit 2 interfaces.
- [x] Compare the approved conceptual backend-selection flow with the concrete
  Unit 2 boundary and identify one material ambiguity in ordinary targeting.
- [x] Create a dedicated Functional Design question file with mutually
  exclusive choices and an answer tag.
- [x] Collect and validate the answer, including consistency with the approved
  current-method tie-in and Unit 2 single-stage map-simulation boundary.
- [x] Define selector validation, replay intent, default compatibility, target
  provenance, and supported-target rejection rules.
- [x] Define pure extraction of one detached `HprTargetMapBasis` from successful
  supported targets, including refrigerant mixtures and nominal-point facts.
- [x] Define the explicit `hpr_performance_map` application flow, return value,
  non-mutation guarantees, typed errors, and optional TESPy behavior.
- [x] Define scalar, selected-period, multi-period, all-period, and workspace
  batch behavior without adding a second public workflow.
- [x] Define public documentation, example, schema/fixture publication, optional
  installation, limitations, and OpenUtility-independent consumption rules.
- [x] Identify complementary examples and PBT properties for omitted-selector
  equivalence, explicit dispatch, replay, basis detachment, non-mutation,
  rejection, serialization, and dependency isolation.
- [x] Create and validate `business-logic-model.md`, `business-rules.md`, and
  `domain-entities.md` in the Unit 3 Functional Design directory.
- [x] Update plan and stage tracking and request explicit Functional Design
  approval before Unit 3 NFR Requirements.
- [x] Obtain explicit approval of the Unit 3 Functional Design before NFR
  Requirements begins.

## Fixed Boundaries

- CoolProp remains the default and TESPy remains optional.
- Map generation is explicit and never runs for an omitted follow-up call.
- Schema `1.0` remains fixed-capacity and single-source/single-sink.
- OpenUtility consumes only plain versioned data and never imports OpenPinch.
- OpenPinch imports neither OpenUtility, Pyomo, nor HiGHS.
- No root export, CLI, automatic map attachment, partial map, or multi-port
  flattening is introduced.

## Content Validation

This plan contains no Mermaid diagram, ASCII diagram, executable code block,
JSON, or YAML. Markdown headings, lists, inline code, paths, and checkbox syntax
were checked before creation.
