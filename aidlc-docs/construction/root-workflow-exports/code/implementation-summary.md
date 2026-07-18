# Root Workflow Exports Implementation Summary

## Outcome

`OpenPinch` now exports exactly `PinchProblem` and `PinchWorkspace`. Both names
resolve to their concrete application-owner classes, so the canonical user
import is:

```python
from OpenPinch import PinchProblem, PinchWorkspace
```

## Design Decisions

- The concrete application modules remain the implementation owners.
- The root `__all__` is explicit and limited to the two workflow classes.
- Schemas, enums, resources, lower-level services, and
  `pinch_analysis_service` are not added to the root surface.
- The strict service import remains
  `from OpenPinch.main import pinch_analysis_service`.
- No legacy alias, fallback, dynamic barrel, or migration behavior was added.

## Affected Surfaces

- Package-root API and architecture regressions.
- Fresh-process optional-dependency import checks.
- Curated API, guide, example, stability, reference, and release documentation.
- All ten packaged notebooks and their import/support contract checks.

## Validation

The focused export identity, optional-dependency cold-import, curated-doc, and
packaged-notebook contract checks pass. The complete affected gate passes 2,092
non-solver tests with four solver tests and the pre-existing notebook-output
cleanliness assertion deselected. Ruff lint/format, warning-as-error Sphinx,
notebook JSON parsing, stale-contract search, and patch hygiene pass.

The isolated cleanliness assertion continues to detect execution counts and
outputs already present in notebook 01. Those unrelated local results were
preserved; the notebook's import and support-notice source cells validate.

## Extension Compliance

- Security: N/A; disabled in the workflow state.
- Resiliency: N/A; disabled in the workflow state.
- Partial Property-Based Testing: N/A; no numerical algorithm or generated
  invariant changed.
