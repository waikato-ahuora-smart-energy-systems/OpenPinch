# Staged Change Quality Audit Plan

## Scope

- Audit the complete Git index: 101 staged files with approximately 6,400 added lines, including this completed audit record.
- Keep unrelated external state out of scope; review and verify the exact staged application, tests, documentation, workflow, and dependency changes.
- Enforce the repository CI coverage floor of 95% using the same non-solver command and Hypothesis seed as CI.

## Audit Steps

- [x] Step 1: Inventory staged versus unstaged state, staged file categories, diff size, whitespace validity, and the configured CI coverage target.
- [x] Step 2: Review domain models, schemas, input preparation, stream helpers, collection views, public exports, serialization, and persistence behavior.
- [x] Step 3: Review targeting, area/cost, HPR, MVR, and stream-linearisation changes for parent/segment identity and numerical correctness.
- [x] Step 4: Review HEN arrays, formulations, piecewise mappings, extraction, verification, area slices, and solver-path behavior.
- [x] Step 5: Review tests, property strategies, fixtures, docs, dependency/lock changes, and CI workflow consistency.
- [x] Step 6: Run full CI-equivalent coverage, Ruff, formatting, non-solver tests, segmented synthesis tests, docs, notebooks/resources, packaging, and patch checks.
- [x] Step 7: Correct validated findings, add regression coverage where needed, and rerun proportional checks.
- [x] Step 8: Update coverage/build records, AI-DLC state, audit log, and this checklist with the final reviewed result.

## Review Standards

- Behavior and public contracts are explicit and backward compatible where required.
- Parent identity controls physical topology; segments control piecewise thermal calculations.
- Transactions are atomic, caches are revision-aware, and multiperiod identities remain stable.
- Numerical code rejects invalid or unresolved states rather than silently averaging or repairing profiles.
- Helpers have narrow responsibilities, no duplicated calculations, and no unnecessary public exposure.
- Tests cover valid, invalid, boundary, persistence, multiperiod, and solver-path behavior.
