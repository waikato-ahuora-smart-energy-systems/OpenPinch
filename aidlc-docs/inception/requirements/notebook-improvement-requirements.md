# Notebook Improvement Requirements

## Intent Analysis

- **User request**: Improve the OpenPinch tutorial notebooks.
- **Request type**: User-facing tutorial and presentation enhancement.
- **Scope estimate**: All 18 packaged tutorial notebooks, their canonical
  generator, and notebook-focused verification.
- **Complexity estimate**: Moderate. The content is generator-owned and spans
  four execution profiles, including slow and external-solver workflows.
- **Requirements depth**: Standard, with explicit acceptance criteria for the
  cross-notebook presentation contract.

## Objective

Turn every packaged tutorial into an unattended, process-engineering
demonstration that concludes with a useful inline plot or table and concise
engineering interpretation. Preserve the current source-only distribution
contract while making the results visible when a user executes the notebook.

## Current Context

- `scripts/generate_tutorial_notebooks.py` is the canonical source for all 18
  packaged notebooks.
- The notebooks cover ten `base`, four `slow-hpr`, three `solver`, and one
  `interactive` execution profile.
- Packaged notebooks currently store no execution counts or cell outputs.
- Existing tests enforce notebook inventory, public imports, source-only JSON,
  operation coverage, and profile-specific execution.
- The previously observed generated notebook edits and workspace JSON file are
  no longer present in the working tree.

## Functional Requirements

### FR-01: Complete notebook scope

The improvement shall cover every `.ipynb` file generated into
`OpenPinch/data/notebooks/` by `scripts/generate_tutorial_notebooks.py`. It shall
not add a nineteenth tutorial or modify `examples/debug_support_notebook.ipynb`
unless a later approved change expands scope.

### FR-02: Generator authority

All durable notebook structure and content changes shall be made in
`scripts/generate_tutorial_notebooks.py`. Regenerating notebooks from that script
shall reproduce the complete packaged notebook set without hand-edited source
divergence.

### FR-03: Source-only distribution

Generated notebooks committed to the repository shall retain null execution
counts and empty stored output arrays. The work shall not commit executed cell
outputs, embedded plot payloads, or environment-specific execution metadata.

### FR-04: Visible inline conclusions

Each tutorial shall conclude its demonstrated workflow with at least one useful
inline result appropriate to the engineering task. The result may be a plot,
summary table, comparison table, report view, metrics view, network grid, or
other reviewable result object exposed through the current public workflow API.
Merely assigning a result to a variable without presenting it shall not satisfy
this requirement.

### FR-05: Engineering interpretation

The inline conclusion in each tutorial shall be followed or accompanied by
concise process-engineering interpretation that explains what to inspect and
how the result informs a study decision. Interpretation shall be specific to the
tutorial subject rather than repeated generic prose.

### FR-06: Comprehensive narrative

Each tutorial shall provide sufficient narrative to demonstrate the complete
study flow even when that makes the notebook longer. The narrative shall retain
or improve prerequisites, learning outcome, study question, ordered execution
steps, assumptions, result interpretation, and adaptation guidance.

### FR-07: Unattended execution

Every tutorial shall execute sequentially without `input()`, manual file
selection, widget interaction, edits between cells, or other user-supplied
runtime decisions. Optional capabilities may require their declared dependency
profile but shall not require interactive choices after execution begins.

### FR-08: Public workflow contract

Examples shall use the public `PinchProblem` or `PinchWorkspace` workflow and
their public accessors. Presentation cells shall inspect cached analysis results
and shall not silently rerun expensive engineering methods merely to create a
view.

### FR-09: Profile-appropriate presentation

Visual and tabular conclusions shall match the tutorial subject. Heat
integration tutorials shall expose curves, profiles, targets, or comparisons;
heat-pump and cogeneration tutorials shall expose cycle or load results; HEN
tutorials shall expose ranked designs or network views; reporting tutorials
shall expose reportable tables and presentation surfaces. When no meaningful
plot exists, a well-labelled table or report view is preferred to a decorative
chart.

### FR-10: Existing curriculum and coverage preservation

The numbered 18-notebook curriculum, declared execution profiles, documented
public operation coverage, and packaged resource discovery shall remain
complete. Improvements may reorganize cells and prose but shall not remove an
operation required by the tutorial coverage manifest.

## Non-Functional Requirements

### NFR-01: Executability

All ten base notebooks and all declared optional profiles shall pass their
execution checks. Verification shall include the four slow-HPR notebooks, three
external-solver HEN notebooks, and the interactive notebook when their existing
declared extras and solver prerequisites are available.

### NFR-02: Determinism and portability

Notebook source generation shall be deterministic. Tutorials shall use packaged
sample data or generated temporary paths and shall not depend on a developer's
working directory, local absolute paths, prior notebook execution state, or
network access.

### NFR-03: Maintainability

Shared notebook conventions shall be expressed through generator helpers or
structured guidance data where practical. Notebook-specific interpretation and
presentation choices shall remain explicit enough for maintainers to review.

### NFR-04: Runtime transparency

Each notebook shall retain an accurate execution profile, expected runtime, and
optional-extra declaration. Presentation improvements shall not obscure or
materially inflate expensive analysis calls.

### NFR-05: Validation and regression protection

Automated tests shall verify inventory, valid notebook JSON, source-only output
state, public imports, required narrative, result-presentation structure,
operation coverage, generator reproducibility, and execution profiles. Visual
verification shall confirm that representative plot and table conclusions are
renderable rather than only syntactically present.

### NFR-06: Property-based testing extension

Partial Property-Based Testing is enabled. PBT-02, PBT-03, PBT-07, PBT-08, and
PBT-09 are blocking only where the implementation changes pure transformations
or serialization round trips. Hypothesis remains the selected Python framework,
with shrinking and deterministic seed reproduction retained. Pure prose and
fixed notebook examples do not require artificial property tests.

### NFR-07: Security and resiliency extensions

The Security Baseline and Resiliency Baseline are disabled for this workflow.
Ordinary repository validation and safe temporary-file practices still apply.

## Acceptance Criteria

| ID | Criterion |
|---|---|
| AC-01 | Running the canonical generator creates exactly the expected 18 numbered packaged notebooks. |
| AC-02 | A second generator run produces byte-identical notebook sources and no new Git diff. |
| AC-03 | Every generated code cell has a null execution count and an empty output list in the committed notebook. |
| AC-04 | Every notebook visibly presents at least one profile-appropriate inline plot, table, report, grid, metrics view, or result object during execution. |
| AC-05 | Every notebook includes subject-specific engineering interpretation of its concluding presented result. |
| AC-06 | No notebook requires runtime user input, manual cell edits, or interactive selection to complete its declared workflow. |
| AC-07 | All existing tutorial coverage manifest operations remain demonstrable from generated notebook source. |
| AC-08 | All base, slow-HPR, solver, and interactive notebook execution profiles pass in environments with their declared prerequisites. |
| AC-09 | Notebook packaging, resource discovery, documentation consistency, public-import, and source-compilation checks pass. |
| AC-10 | Representative inline plots and tables are verified as renderable, and failures identify the notebook and presentation cell. |
| AC-11 | No new runtime or notebook dependency is introduced without explicit approval and matching package metadata. |
| AC-12 | Partial PBT compliance is documented; applicable pure transformation or serialization invariants use domain-appropriate generators, shrinking, and reproducible seeds. |

## Out of Scope

- Adding new tutorial notebooks or changing the debug-support notebook.
- Committing executed cell outputs or execution counts.
- Requiring an exported HTML, image, or workbook as the conclusion of every
  tutorial; existing export-focused examples may still demonstrate exports.
- Changing the root public API or engineering algorithms solely to create
  tutorial visuals.
- Adding deployment, cloud infrastructure, or hosted notebook services.
- Replacing existing optional solver or plotting dependencies.

## Traceability

| Source decision | Requirements |
|---|---|
| All 18 packaged notebooks | FR-01, FR-10, AC-01, AC-07 |
| Visual presentation is the primary goal | FR-04, FR-05, FR-09, AC-04, AC-05, AC-10 |
| Previous generated edits are gone | Current Context, FR-02 |
| Generator is authoritative | FR-02, NFR-02, NFR-03, AC-02 |
| Execute every declared profile | NFR-01, AC-08 |
| Comprehensive narrative with no user input | FR-06, FR-07, AC-05, AC-06 |
| Keep notebooks source-only | FR-03, AC-03 |
| Prefer inline result plus interpretation | FR-04, FR-05, AC-04, AC-05 |
| Security disabled | NFR-07 |
| Resiliency disabled | NFR-07 |
| Partial Property-Based Testing | NFR-06, AC-12 |

## Requirements Summary

The approved change will improve all 18 generated tutorials as source-only,
unattended demonstrations. Each notebook will visibly conclude with an
engineering-relevant inline plot or table and subject-specific interpretation,
while preserving the public API curriculum and passing every declared execution
profile. The generator remains the only durable notebook source.

## Extension Compliance

| Extension rule | Status | Rationale |
|---|---|---|
| Security Baseline | N/A | Disabled by the user for this workflow. |
| Resiliency Baseline | N/A | Disabled by the user for this workflow. |
| PBT-02 Round trips | N/A at Requirements Analysis | No implementation or serialization pair has yet been changed; applicability is required during design and code planning. |
| PBT-03 Invariants | N/A at Requirements Analysis | Generator invariants are specified in NFR-02 and AC-01 through AC-03; test design occurs later. |
| PBT-07 Generator quality | N/A at Requirements Analysis | No property strategy is created in this stage. |
| PBT-08 Shrinking and reproducibility | Compliant | NFR-06 and AC-12 retain shrinking and deterministic seed reproduction when PBT applies. |
| PBT-09 Framework selection | Compliant | Hypothesis is the existing selected Python PBT framework and is retained. |
