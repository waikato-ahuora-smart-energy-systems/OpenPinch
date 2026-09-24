# Unit 2 Code Generation Plan

Status: Approved under the user's authorization through completion; Part 2 active.

This plan is the single source of truth for Unit 2 Code Generation.

## Unit Context

- Requirements: FR-4 through FR-9; VR-1, VR-2, VR-5, VR-7 through VR-9.
- Dependency: completed Unit 1 contracts and finalizer.
- Runtime owners: existing public target accessor, HPR preprocessing/service,
  optimisation adapter, multiperiod coordinator, fluid resolver, and topology modules.
- Database/infrastructure: none.

## Part 1

- [x] Read Unit 2 design, dependencies, existing owners, and focused tests.
- [x] Define the eight-step TDD/PBT sequence below.
- [x] Approve the complete sequence under the user's completion authorization.

## Part 2

### Step 1 — RED budget, preflight, cache, and failure tests
- [x] Add focused examples and properties for U2-P1 through U2-P12.
- [x] Confirm failures reflect missing Unit 2 behavior.

### Step 2 — Budget propagation
- [x] Add strict budgets to HPR inputs, public wrappers, preprocessing, service, and multiperiod paths.
- [x] Map resolved limits to generic `OptimisationOptions`.

### Step 3 — CoolProp capability preflight
- [x] Add detached topology/stage capability facts using the existing fluid resolver.
- [x] Invoke preflight before optimiser entry for cascade, parallel, VC+MVR, and multiperiod paths.

### Step 4 — Warm-start-first exact cache
- [x] Evaluate normalized warm starts before backend invocation.
- [x] Memoize exact search coordinates and retain viable warm starts through backend exhaustion.

### Step 5 — Bounded typed diagnostics
- [x] Accumulate classified counts and at most 16 detached representatives.
- [x] Raise `HPRTargetingError` for preflight/no-viable outcomes while fatal defects propagate.

### Step 6 — Public and multiperiod integration
- [x] Align direct/utility, HP/refrigeration, VC+MVR, and multiperiod budget/preflight behavior.
- [x] Preserve TESPy and old-call compatibility.

### Step 7 — Focused verification and refactor
- [x] Run properties, real/fake engine tests, regressions, Ruff, formatting, compilation, and patch hygiene.
- [x] Verify no dependencies or duplicate brownfield source files.

### Step 8 — Summary and transition
- [x] Generate Unit 2 code summary with FR/NFR/PBT traceability.
- [x] Mark Unit 2 complete and continue to Unit 3.
