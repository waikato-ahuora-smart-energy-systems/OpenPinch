# Unit 3 Current HPR Target API Integration and Publication NFR Requirements Plan

## Scope

Define measurable quality requirements and technology choices for the public
CoolProp-default/TESPy-selectable HPR targeting integration, detached target
records, explicit performance-map follow-up, documentation, and release gates.

## Plan

- [x] Read the approved Unit 3 Functional Design and identify the performance,
  determinism, optional-dependency, compatibility, reliability,
  maintainability, and publication risks.
- [x] Carry forward the approved Unit 1/2 technology baseline: Python 3.11 or
  newer, Pydantic contracts, CoolProp 8 or newer, optional TESPy 0.11.2 or
  newer, pytest, Hypothesis, uv, Hatchling, Ruff, and blocking base/TESPy CI
  profiles.
- [x] Confirm availability, disaster recovery, authentication, authorization,
  external persistence, network service, UI, and accessibility requirements are
  not applicable to this in-process scientific library feature.
- [x] Identify unresolved quality choices for duplicate expensive candidate
  evaluation and portable performance gating.
- [x] Create a dedicated NFR question file using complete mutually exclusive
  choices and answer tags.
- [x] Collect and validate every answer and resolve any ambiguity.
- [x] Define measurable backward-compatibility, performance, scalability,
  determinism, numerical, reliability, optional-install, and packaging
  requirements.
- [x] Define the approved technology stack, dependency bounds, lazy-import
  firewall, evaluator injection, cache policy, profiling, and test profiles.
- [x] Define PBT-09 compliance and carry Functional Design PBT properties into
  later Code Generation planning.
- [x] Create and validate `nfr-requirements.md` and
  `tech-stack-decisions.md` in the Unit 3 NFR Requirements directory.
- [x] Update plan and stage tracking and request explicit NFR Requirements
  approval before Unit 3 NFR Design.
- [x] Obtain explicit approval of Unit 3 NFR Requirements before NFR Design.

## Fixed Quality Boundaries

- CoolProp omitted-selector behavior is a blocking compatibility oracle.
- TESPy remains optional and lazily imported only by its concrete leaf.
- Invalid or failed TESPy evaluations never fall back to CoolProp.
- Base and TESPy source and built-artifact profiles remain blocking.
- Security and Resiliency extensions remain disabled; Property-Based Testing
  remains enabled.

## Content Validation

This plan contains no Mermaid diagram, ASCII diagram, executable code block,
JSON, or YAML. Markdown headings, lists, paths, inline code, and checkbox syntax
were checked before creation.
