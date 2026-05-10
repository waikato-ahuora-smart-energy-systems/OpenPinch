# OpenPinch Improvement Plan

This file captures the highest-value cleanup and maintenance work identified
from a repo scan on 2026-05-10. The ranking favors user-visible correctness
issues first, then API safety, then architectural and documentation cleanup.

## Ranked TODOs

### 1. Fix validation warning reporting

Priority: Critical

The validation warning path in `PinchProblem` reports the wrong issue list,
which produces misleading warnings even when the test suite passes.

- File: `OpenPinch/classes/pinch_problem.py`
- Why it matters: diagnostic output is part of the public contract; misleading
  warnings undermine trust in validation results.
- Recommended action: use the warning issue list when building warning
  messages, and add a regression test around mixed warning/fatal cases.

### 2. Remove mutable default arguments and mutable schema defaults

Priority: High

Several public and internal APIs still use shared mutable defaults.

- Files:
  - `OpenPinch/services/services_entry.py`
  - `OpenPinch/services/common/problem_table_analysis.py`
  - `OpenPinch/utils/blackbox_minimisers.py`
  - `OpenPinch/lib/schemas/io.py`
- Why it matters: this can create shared-state bugs that are hard to reproduce.
- Recommended action: replace mutable defaults with `None` plus local
  initialization, and use Pydantic `Field(default_factory=...)` for schema
  containers.

### 3. Tighten exception handling and remove stray library-side prints

Priority: High

There are still broad `except` blocks and direct `print()` calls in code paths
that behave like library APIs.

- Files:
  - `OpenPinch/classes/pinch_problem.py`
  - `OpenPinch/utils/stream_linearisation.py`
  - `OpenPinch/classes/value.py`
  - `OpenPinch/utils/export.py`
  - `OpenPinch/classes/vapour_compression_cycle.py`
  - `OpenPinch/classes/brayton_heat_pump.py`
- Why it matters: broad exception handling hides root causes and stdout side
  effects make the package harder to embed and test cleanly.
- Recommended action: narrow exception scopes, preserve traceback context, and
  replace direct prints with structured logging or explicit return values.

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

### 5. Refactor configuration into a cleaner typed model

Priority: Medium

The configuration layer carries visible legacy structure and is already marked
for refactor in comments.

- File: `OpenPinch/lib/config.py`
- Why it matters: configuration touches many parts of the pipeline, so unclear
  structure increases coupling and slows future changes.
- Recommended action: separate legacy compatibility constants from canonical
  typed config objects and document ownership of each config group.

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

### 7. Align docs with actual feature readiness

Priority: Medium

The docs do not fully reflect the current support level of some features.

- Files:
  - `README.md`
  - `docs/reference/architecture.rst`
  - `docs/user-guide/notebooks.rst`
- Why it matters: mismatches between docs and implementation lead users toward
  unsupported or unstable workflows.
- Recommended action: explicitly label mature, experimental, and planned
  features, especially for multi-scale targeting and the less-complete analysis
  modes.

### 8. Strengthen notebook and workflow regression coverage

Priority: Medium

The notebooks are presented as a primary learning path, so workflow regressions
need stronger protection.

- Files:
  - `OpenPinch/data/notebooks/`
  - `tests/test_notebooks.py`
- Why it matters: tutorial breakage directly affects first-time user success.
- Recommended action: keep the notebooks on supported public APIs, add focused
  checks for key cells and outputs, and treat notebook regressions as
  user-facing failures.

### 9. Normalize CLI and utility output conventions

Priority: Low

Several utility-style modules still write directly to stdout.

- Files:
  - `OpenPinch/utils/csv_to_json.py`
  - `OpenPinch/utils/wkbook_to_json.py`
  - parts of `OpenPinch/__main__.py`
- Why it matters: inconsistent output policy makes the toolkit harder to use as
  both a library and a CLI.
- Recommended action: define a clear policy for logging, quiet mode, and
  user-facing terminal output.

### 10. Clean up scattered TODOs and placeholders

Priority: Low

There are a number of lower-signal TODOs that should either be tracked
explicitly or completed.

- Files:
  - `OpenPinch/classes/cascade_vapour_compression_cycle.py`
  - `OpenPinch/services/common/gcc_manipulation.py`
  - `OpenPinch/services/heat_pump_integration/cycles/multi_temperature_carnot.py`
  - `tests/test_analysis/test_capital_cost_and_area_targeting.py`
- Why it matters: small TODOs are easy to ignore until they become stale or
  misleading.
- Recommended action: convert the meaningful ones into issues or backlog items
  with owners, and remove obsolete placeholders.

## Suggested Execution Order

### Phase 1: Correctness and API safety

1. Fix validation warning reporting.
2. Remove mutable defaults.
3. Tighten exception handling and remove library-side prints.

### Phase 2: Public surface cleanup

4. Decide the fate of unfinished analysis subsystems.
5. Refactor configuration.
6. Rework packaging and dependency boundaries.

### Phase 3: User guidance and polish

7. Align docs with feature readiness.
8. Strengthen notebook regression coverage.
9. Normalize CLI output conventions.
10. Clean up smaller TODOs and placeholders.

## Notes

- This plan is intentionally maintenance-focused rather than feature-focused.
- The recent `dt_cont` multiplier and dual-`dt_cont` stream work reduced the
  need for deeper zone/collection refactors in the near term.
- If this list turns into tracked issues, each item should be split into
  concrete tasks with a target release or owner.
