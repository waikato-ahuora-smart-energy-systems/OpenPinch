# OpenPinch Improvement Plan

This file captures the highest-value cleanup and maintenance work identified
from a repo scan on 2026-05-10. The ranking favors user-visible correctness
issues first, then API safety, then architectural and documentation cleanup.

## Dedicated Plans

- Read the Docs overhaul backlog:
  - `RTD_OVERHAUL_PLAN.md`

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
