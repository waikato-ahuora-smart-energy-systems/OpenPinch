# OpenPinch Improvement Plan

This file captures the highest-value cleanup and maintenance work identified
from a repo scan on 2026-05-10. The ranking favors user-visible correctness
issues first, then API safety, then architectural and documentation cleanup.

## Dedicated Plans

- Read the Docs overhaul backlog:
  - `RTD_OVERHAUL_PLAN.md`

## Future Development Ideas

These items came into focus while rebuilding the documentation and reviewing
the public surface area of the package.

### Public API stability metadata

Priority: High

- Why it matters: the docs now distinguish between `Stable`, `Advanced`, and
  `Experimental / partial` surfaces, but that status still lives only in prose.
- Recommended action: add code-adjacent metadata or a maintained registry that
  identifies the support level of public modules, CLI commands, notebooks, and
  sample cases.

### Generated configuration catalog

Priority: High

- Why it matters: `Configuration` has become a real public surface through
  `zone.config`, but its fields are still documented manually and are easy to
  let drift.
- Recommended action: add field-level metadata and generate a config catalog
  for docs and validation, especially for HPR, costing, and turbine settings.

### Packaged asset manifest

Priority: Medium

- Why it matters: notebooks and sample cases are now part of the supported
  product surface, but there is no machine-readable description of what each
  asset demonstrates.
- Recommended action: add a manifest for packaged notebooks and sample cases
  with names, workflow tags, descriptions, and support levels that can drive
  both docs and CLI discovery.

### Scenario and comparison API expansion

Priority: Medium

- Why it matters: the package has strong case-comparison and HPR comparison
  patterns, but they are still mostly wrapper-specific and notebook-driven.
- Recommended action: promote scenario comparison into a clearer public API for
  baseline-versus-modified studies, including summaries, deltas, and graph
  comparison helpers.

### Complete or retire the helper-backed heat-pump comparison path

Priority: High

- Why it matters: the docs and CLI reference a dedicated
  `evaluate_heat_pump_integration(...)` / `openpinch heat-pump` route, but that
  surface is not yet complete as a first-class public wrapper API.
- Recommended action: either implement the missing wrapper contract and tests
  end to end, or explicitly retire that route and keep HPR guidance centered on
  the supported `problem.target.*` surfaces.

### Optional dependency profiles

Priority: Medium

- Why it matters: the docs now expose distinct workflows for core targeting,
  graphing, Streamlit, and HPR/cogeneration, but installation is still heavier
  than those usage modes require.
- Recommended action: split optional features into extras so core pinch users,
  notebook users, dashboard users, and advanced cycle users can install tighter
  dependency sets.

### Docs and public-surface CI gates

Priority: Medium

- Why it matters: the new docs structure is broader and more explicit, so drift
  between code, assets, and docs will become visible quickly.
- Recommended action: add CI checks for Sphinx warnings, broken links, and
  coverage of package-root exports, packaged assets, and major `PinchProblem`
  workflow members.

## Ranked TODOs

### 4. Decide the fate of unfinished analysis subsystems

Priority: High

Some modules remain as large commented-out placeholders or partial
implementations.

- Files:
  - `OpenPinch/services/energy_transfer_analysis/energy_transfer_analysis.py`
  - `OpenPinch/services/exergy_analysis/exergy_targeting_entry.py`
- Why it matters: dead or half-supported package surface area creates user
  confusion and raises maintenance cost.
- Recommended action: choose one of three states for each subsystem:
  restore, mark experimental, or remove from the public surface.

### 6. Rework packaging and dependency boundaries

Priority: Medium

The packaging metadata is functional but heavier and stricter than it likely
needs to be.

- File: `pyproject.toml`
- Why it matters: install friction limits adoption and contributor setup.
- Observations:
  - Python `>=3.14` is unusually restrictive.
  - heavy packages such as `streamlit`, `plotly`, and `tespy` are unconditional
    runtime dependencies.
  - the dev dependency list contains duplicate `ruff` entries.
- Recommended action: review the minimum supported Python version, split
  optional features into extras where practical, and clean up duplicated dev
  dependencies.
