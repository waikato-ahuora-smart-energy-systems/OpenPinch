# RTD and Comprehensive HPR Notebook Requirements

## Intent Analysis

- User request: "Update RTD and make sure there is a notebook covering a
  comprehensive example".
- Request type: user-facing documentation and tutorial enhancement.
- Scope: multiple documentation/testing components within the existing
  OpenPinch package; no thermodynamic model or public behavior change is
  intended.
- Complexity: moderate. The notebook is generated, packaged, inventoried,
  optionally executed, and described by Read the Docs, so all owners must stay
  synchronized.
- Project type: brownfield Python library.

## Functional Requirements

### FR-NB-01: Canonical Comprehensive Tutorial

Extend the existing
`09_vapour_compression_and_brayton.ipynb` tutorial rather than adding a
duplicate notebook. The canonical generator remains the source of truth, and
the packaged notebook must be regenerated from it.

### FR-NB-02: Targeting Coverage

The tutorial must demonstrate current vapour-compression heat-pump targeting,
including the omitted CoolProp default and explicit TESPy selection. It must
show the supported one-evaporator, one-condenser, single-stage boundary and use
bounded tutorial search settings.

### FR-NB-03: Refrigerant and Mixture Coverage

The tutorial must show at least one pure/registered working-fluid example and
one explicit molar-mixture specification using the published CoolProp syntax.
It must explain that installed CoolProp/TESPy property support and the requested
state determine actual mixture feasibility.

### FR-NB-04: Winning Record and Performance Map

For a successful scalar HPR target, the tutorial must inspect the detached
winning simulation record, derive source/sink map coordinates from that record,
construct `HprPerformanceMapRequest`, call
`problem.target.hpr_performance_map`, and inspect multiple part-load points.

### FR-NB-05: Plain Export Boundary

The tutorial must serialize the performance map with
`model_dump(mode="json")` and make clear that OpenUtility consumes this plain,
versioned structure without either package importing the other.

### FR-NB-06: Screening and Failure Behavior

The example must retain an explicit guarded screening pattern for infeasible
thermodynamic candidates. TESPy selection must not imply fallback to CoolProp,
and map generation must not imply a partial result on failure.

### FR-NB-07: RTD Integration

Update the heat-pump workflow guide, notebook series, tutorial coverage page,
and any relevant reference/capability text so notebook 09 is clearly identified
as the comprehensive target-to-map example. The documented code and notebook
source must use the same public contract.

### FR-NB-08: Tutorial Inventory

Restore `problem.target.hpr_performance_map` to notebook-demonstrated coverage
once notebook 09 contains an executable call. The public-operation manifest,
notebook source audit, and documented denominator must agree.

## Non-Functional Requirements

### NFR-NB-01: Reproducibility

The generator and packaged notebook must be byte-stable when regenerated with
unchanged source. Existing notebook metadata, profile conventions, and public
import rules must remain explicit.

### NFR-NB-02: Executability

All notebook code cells must compile. The comprehensive notebook must execute
in the declared optional HPR/TESPy profile or have deterministic tests for any
bounded real-engine section whose runtime is deliberately isolated.

### NFR-NB-03: Runtime Bound

The comprehensive real target-to-map path must use bounded tutorial grids and
search settings. It must remain within the existing 300-second guarded public
smoke budget on the supported environment.

### NFR-NB-04: Documentation Quality

Sphinx must build all RTD sources with warnings treated as errors. Examples
must distinguish design targeting from performance-map offdesign generation,
state compressor-only power, and avoid implying that OpenUtility functionality
is part of OpenPinch.

### NFR-NB-05: Compatibility

No package-root export, CLI command, schema version, solver dependency, or
thermodynamic numerical behavior may change solely to support the tutorial.
The specialist public contract import
`OpenPinch.contracts.hpr_performance_map` may be explicitly allowed by the
notebook source policy because it is already the documented request owner.

### NFR-NB-06: Regression Coverage

Run generator idempotence, notebook compilation/source inventory, tutorial
coverage, RTD consistency, warning-strict Sphinx, focused HPR public workflow,
Ruff, and patch-hygiene gates. Retain Hypothesis seed `20260715` where generated
tests run.

## Success Criteria

- Notebook 09 contains a coherent executable target-to-map study rather than a
  disconnected API mention.
- The generator and packaged notebook are synchronized.
- RTD identifies notebook 09 as the comprehensive example and accurately states
  backend, mixture, failure, and OpenUtility boundaries.
- The tutorial manifest once again classifies the map call as mapped and
  executable.
- Relevant notebook, documentation, HPR, packaging, and quality gates pass.

## Extension Configuration

- Property-Based Testing: enabled. Existing generator idempotence, manifest
  completeness, serialization, and HPR physical properties remain blocking
  where applicable.
- Security: disabled; N/A for documentation/notebook-only scope.
- Resiliency: disabled; N/A for documentation/notebook-only scope.

This document contains no Mermaid or ASCII diagram. Markdown headings, lists,
paths, identifiers, quotations, and inline code were validated before creation.
