# OpenPinch RTD Overhaul Plan

This file captures a full documentation overhaul plan for the Read the Docs
site. The goal is not a light refresh. The goal is to turn the current docs
into a technical manual that explains:

- what OpenPinch does
- the thermodynamic grounding behind the package
- the supported workflows and their decision context
- the full practical API surface
- the maturity level of each exposed subsystem

The plan below is based on the current docs and package surface as of
2026-05-11.

## Why This Overhaul Is Needed

The current RTD site has three structural gaps:

1. It is workflow-forward but method-light.
   The docs explain how to run some workflows, but they do not explain the
   technical foundations of direct integration, indirect integration, problem
   tables, utility targeting, graph meaning, and multiscale zone aggregation
   in enough depth.

2. It under-exposes the package power.
   The codebase exposes a much richer front door than the docs make obvious:
   `PinchProblem`, `problem.plot.*`, `problem.target.*`, the CLI, packaged
   sample cases, packaged notebooks, schema-driven service calls, and multiple
   advanced analysis modes.

3. It does not distinguish stable public API from expert-level or partial
   package surface.
   The docs currently expose a broad module reference, but they do not make it
   clear which parts are the intended stable user-facing API and which parts
   are expert, internal, or still maturing.

## Overhaul Objectives

The new RTD site must:

- explain the technical foundations of the package clearly
- show the package capability map before drilling into module internals
- document the real public API by user intent, not just by package path
- make advanced features discoverable without overwhelming new users
- clearly label support level and maturity
- connect every major workflow to a sample case or notebook
- support both first-time users and advanced integrators

## Design Principles

1. Fundamentals before reference.
   A user should understand the thermodynamic and workflow model before being
   dropped into `automodule` pages.

2. Curated API before exhaustive API.
   The reference should first present the supported user-facing entrypoints,
   then provide generated detail pages for complete coverage.

3. Organize by user question.
   Guides should answer questions like "How do I model a total site?" or "How
   do I evaluate a heat pump integration scenario?" rather than mirror the
   Python package tree.

4. Be explicit about maturity.
   Stable, advanced, experimental, and internal surfaces must be labeled.

5. Keep examples first-class.
   Packaged notebooks and sample cases are part of the product surface and
   should be documented as such.

## Current Baseline

The current docs tree is small and concentrated in:

- `docs/index.rst`
- `docs/getting-started.rst`
- `docs/user-guide/*.rst`
- `docs/reference/*.rst`

The current reference already covers many modules via autodoc, but it lacks:

- a full technical methods section
- a capability matrix
- a workflow map
- a support/stability matrix
- dedicated pages for the highest-value public surfaces
- page-level intent separation between new-user, advanced-user, and developer
  audiences

## Target Site Architecture

The RTD site should be reorganized into these top-level sections:

1. `Overview`
2. `Fundamentals`
3. `Guides`
4. `API Reference`
5. `Examples`
6. `Developer Notes`

### Proposed Top-Level Toctree

- `index`
- `overview/index`
- `fundamentals/index`
- `guides/index`
- `api/index`
- `examples/index`
- `developer/index`

## Proposed Page Inventory

### Overview

Purpose: establish scope, power, and navigation.

- `overview/what-is-openpinch.rst`
  - package scope
  - target audience
  - supported workflows
  - package positioning relative to spreadsheet and code workflows

- `overview/capability-matrix.rst`
  - direct integration
  - total-process / total-site targeting
  - graph generation
  - notebook workflows
  - heat pump / refrigeration targeting
  - cogeneration
  - area / cost targeting
  - schema-driven programmatic integration
  - CLI-driven workflows

- `overview/workflow-map.rst`
  - CLI
  - `PinchProblem`
  - service-layer API
  - schema-first workflows
  - sample-case and notebook-driven workflows

- `overview/support-and-stability.rst`
  - stable public API
  - advanced but supported API
  - experimental surfaces
  - internal implementation layers

### Fundamentals

Purpose: provide technical grounding in OpenPinch terms.

- `fundamentals/pinch-analysis.rst`
  - minimum utility targeting
  - heat recovery framing
  - pinch constraints
  - role of `dt_cont`

- `fundamentals/problem-table-and-temperature-shifting.rst`
  - shifted temperatures
  - interval cascades
  - real vs shifted curves
  - how OpenPinch stores these ideas

- `fundamentals/direct-vs-indirect-integration.rst`
  - direct integration
  - indirect utility-mediated integration
  - total-process / total-site logic
  - when results differ and why

- `fundamentals/zones-streams-utilities-and-targets.rst`
  - `Stream`
  - `StreamCollection`
  - `Zone`
  - target models
  - hierarchy semantics

- `fundamentals/graphs-and-interpretation.rst`
  - composite curves
  - shifted composite curves
  - balanced composite curves
  - GCC
  - total-site profiles
  - SUGCC
  - what decision each graph supports

- `fundamentals/heat-pump-and-refrigeration-methods.rst`
  - integration-first framing
  - direct vs indirect HPR targeting
  - screening vs detailed thermodynamic cycle solving

- `fundamentals/cogeneration-methods.rst`
  - above-pinch steam-turbine targeting
  - below-pinch condensing path
  - interpretation of work and efficiency targets

### Guides

Purpose: answer real user questions with end-to-end workflows.

- `guides/first-solve-cli.rst`
- `guides/first-solve-python.rst`
- `guides/input-formats-and-validation.rst`
- `guides/zonal-and-total-site-workflows.rst`
- `guides/heat-pump-workflows.rst`
- `guides/cogeneration-workflows.rst`
- `guides/graphing-and-interpretation.rst`
- `guides/exporting-results.rst`
- `guides/notebooks-and-sample-cases.rst`

Each guide should include:

- the decision question it answers
- the correct API surface to use
- one runnable example
- expected outputs
- links to deeper fundamentals and API pages

### API Reference

Purpose: show the supported entrypoints clearly, then provide exhaustive module
coverage.

- `api/package-root.rst`
  - root imports from `OpenPinch`

- `api/pinchproblem.rst`
  - lifecycle methods
  - `problem.plot.*`
  - `problem.target.*`
  - exports
  - comparison helpers
  - dashboard hooks

- `api/service-layer.rst`
  - `pinch_analysis_service`
  - `data_preprocessing_service`
  - direct / indirect / HPR / cogeneration entrypoints

- `api/schemas-and-config.rst`
  - `TargetInput`
  - `TargetOutput`
  - stream / utility / zone tree schemas
  - `Configuration`
  - important enums

- `api/domain-model.rst`
  - `Zone`
  - `Stream`
  - `StreamCollection`
  - `ProblemTable`
  - `Value`

- `api/cli-and-resources.rst`
  - CLI commands
  - sample-case helpers
  - notebook helpers

- `api/generated-index.rst`
  - generated exhaustive reference grouped by package

### Examples

Purpose: document the real learning assets and supported demonstrations.

- `examples/notebook-series.rst`
  - what each notebook demonstrates
  - expected learning path

- `examples/sample-cases.rst`
  - what each packaged sample case demonstrates
  - recommended use

- `examples/decision-workflows.rst`
  - "basic pinch"
  - "total site"
  - "heat pump integration"
  - "cogeneration"

### Developer Notes

Purpose: keep the docs maintainable and aligned with the codebase.

- `developer/docs-conventions.rst`
  - doc taxonomy
  - stable-vs-experimental labeling rules
  - code example style
  - cross-link rules

- `developer/build-and-coverage.rst`
  - local docs build
  - reference generation
  - expected CI checks

## Public API Inventory Workstream

Before or during the rewrite, create an explicit inventory of all surfaces that
must be represented in the docs.

### Stable Public Surfaces To Curate

- root package imports from `OpenPinch`
- `PinchProblem`
- `pinch_analysis_service`
- `Configuration`
- `TargetInput` / `TargetOutput`
- CLI commands from `openpinch`
- packaged sample-case helpers
- packaged notebook helpers

### Advanced Public Surfaces To Curate

- `prepare_problem`
- service-layer targeting helpers
- `problem.target.*`
- `problem.plot.*`
- graph payload accessors
- stream linearisation helper

### Expert / Partial Surfaces To Label Clearly

- exergy analysis modules
- energy transfer analysis modules
- community / region hierarchy language where implementation maturity is lower
- any remaining package surfaces that are callable but not part of the
  recommended stable user workflow

## Content Workstreams

### Workstream 1: Fundamentals Rewrite

Deliverables:

- fully replace the current architecture page
- add the full fundamentals section
- add at least two diagrams:
  - analysis dataflow
  - zone / target hierarchy

Acceptance:

- a technically literate reader can understand the package model without
  reading source code

### Workstream 2: Workflow Guides Rewrite

Deliverables:

- rewrite getting-started and quickstart into a CLI-first and Python-first pair
- add input-format guidance
- add total-site, heat pump, and cogeneration guides
- make every guide point to one sample case or notebook

Acceptance:

- a new user can choose the right workflow without guessing

### Workstream 3: API Curation

Deliverables:

- dedicated `PinchProblem` page
- dedicated service-layer page
- dedicated schema/config page
- dedicated CLI/resources page
- explicit public API map at the top of the API section

Acceptance:

- the site reveals the practical package power without forcing users through raw
  module trees

### Workstream 4: Examples and Assets

Deliverables:

- notebook-series page
- sample-case page
- one-line purpose statement for every packaged learning asset

Acceptance:

- examples are treated as documented product surface rather than side assets

### Workstream 5: Support and Stability Labeling

Deliverables:

- support matrix page
- consistent labels in API and architecture pages
- explicit notes for partial or expert-only surfaces

Acceptance:

- users can tell what is production-grade versus exploratory

### Workstream 6: Docs Tooling and Coverage

Deliverables:

- clean Sphinx toctree structure
- generated API appendix page
- docs build instructions aligned with repo reality
- CI docs build
- optional link check and docs coverage check for major public surfaces

Acceptance:

- docs regressions are caught alongside code regressions

## Page-by-Page Treatment of Existing Docs

### Keep and Rewrite Heavily

- `docs/index.rst`
- `docs/getting-started.rst`
- `docs/user-guide/quickstart.rst`
- `docs/user-guide/heat-pump-targeting.rst`
- `docs/user-guide/interpreting-results.rst`
- `docs/user-guide/notebooks.rst`
- `docs/reference/architecture.rst`
- `docs/reference/api*.rst`

### Split or Retire

- the current single `architecture.rst` should become multiple fundamentals
  pages
- the current `api.rst` should become a curated API landing page plus a
  generated appendix
- the current `user-guide` grouping should be refactored into task-driven guide
  pages

## Sequencing

### Wave 1: Site Skeleton and Core Narrative

Deliver first:

- new top-level nav
- new landing page
- capability matrix
- workflow map
- support/stability page
- rewritten architecture/fundamentals core

### Wave 2: First-Use and Primary API Coverage

Deliver next:

- CLI first-solve guide
- Python first-solve guide
- `PinchProblem` page
- service-layer page
- schemas/config page
- CLI/resources page

### Wave 3: Advanced Workflows

Deliver next:

- total-site guide
- heat pump guide
- cogeneration guide
- graph interpretation guide
- examples pages

### Wave 4: Exhaustive Reference and Tooling

Deliver last:

- generated API appendix cleanup
- docs conventions
- build-and-coverage page
- CI docs gates
- README / RTD synchronization pass

## Dependencies and Inputs Needed

To execute the overhaul well, the docs pass should draw from:

- public package entrypoints
- packaged notebooks
- packaged sample cases
- CLI commands
- target schemas and config fields
- actual target and graph outputs
- stability decisions for partial subsystems

## Risks

1. Exposing too much internal API as though it were stable.
   Mitigation: add support-level labels early.

2. Rewriting architecture prose without validating against actual code paths.
   Mitigation: use code-based API inventory and cite concrete modules.

3. Building a large reference that still does not help users choose a workflow.
   Mitigation: prioritize capability map and workflow guides before exhaustive
   autodoc polishing.

4. Letting notebooks drift away from written docs.
   Mitigation: give notebooks a dedicated docs section and cross-link them from
   guide pages.

## Definition of Done

The RTD overhaul is complete when:

- the site explains the technical grounding of the package clearly
- the package capability map is visible from the landing flow
- every stable public entrypoint is documented and easy to find
- every major workflow has both a guide and an example path
- advanced and partial surfaces are labeled clearly
- the current notebook and sample-case learning assets are documented directly
- README and RTD no longer describe different product shapes
- docs builds are part of normal maintenance discipline

## Suggested Initial Execution Order

1. Finalize support and stability labels for exposed subsystems.
2. Restructure the toctree and create empty destination pages.
3. Rewrite the landing page and overview section.
4. Rewrite the fundamentals/architecture section.
5. Add the curated `PinchProblem` and API entrypoint pages.
6. Rewrite first-solve guides.
7. Add advanced workflow guides.
8. Add examples pages.
9. Clean and regroup autogenerated API reference.
10. Add docs build and quality gates.
