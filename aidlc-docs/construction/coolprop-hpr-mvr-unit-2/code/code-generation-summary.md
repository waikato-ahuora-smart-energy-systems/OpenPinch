# Unit 2 Code Generation Summary

Unit: CoolProp HPR and VC+MVR Search Reliability

Status: Complete and approved under the user's authorization through completion.

## Outcome

Optimized CoolProp HPR services now validate topology fluids before generic
search, honor explicit iteration/evaluation budgets, evaluate and retain warm
starts first, memoize exact repeated search points, distinguish returned
candidate infeasibility from raised internal defects, and publish bounded typed
failure diagnostics.

## Implemented Changes

- Added `HPRSearchBudget` propagation through every public HPR wrapper,
  single-period preprocessing/service, and multiperiod inputs.
- Applied named-over-options-over-default precedence and early strict validation.
- Mapped budgets to generic `OptimisationOptions.maxiter` and `maxfun` without
  adding HPR knowledge to the reusable optimiser.
- Added detached CoolProp stage-capability preflight for cascade, parallel, and
  VC+MVR topology warm-start states, including every MVR fluid.
- Added warm-start-first evaluation, exact-coordinate search memoization, and
  retained viable baselines when the backend returns no candidates.
- Added bounded `HPRTargetingError` summaries for preflight and no-viable
  outcomes; unexpected runtime/type/lifecycle defects now propagate.
- Preserved TESPy selection behavior and legacy calls with omitted controls.

## Traceability

| Obligation | Evidence |
|---|---|
| FR-4, FR-5 | candidate-local returned failures continue; raised fatal errors abort; bounded structured errors replace generic failures |
| FR-6 | topology/stage CoolProp preflight uses the existing fluid resolver and stops before optimizer entry |
| FR-7 | public strict budgets map exactly to generic optimizer limits |
| FR-9 | warm starts are evaluated first, cached, ranked, and retained through backend exhaustion |
| VR-2, VR-9 | invalid VC/MVR fluid tests identify fluid/stage and prove zero optimizer calls |
| VR-5 | sequence tests cover candidate-local continuation and fatal propagation |
| VR-7 | deterministic evaluation/caching and exact option mapping gates |
| U2-P1 through U2-P12 | fixed-seed properties/examples cover validation, precedence, ordering, caching, caps, and parity |

## Verification

- Focused HPR, contract, and application gate: `762 passed`.
- Ruff and Ruff formatting: clean.
- Python compilation: clean.
- Patch whitespace: clean.
- Dependencies and infrastructure: unchanged.

## Extension Compliance

- Property-Based Testing: compliant; generated limits and deterministic search
  properties run with fixed seeds and shrinking.
- Security and Resiliency: disabled and not applicable to this unit.

## Remaining Work

Unit 3 owns direct process-MVR compression validation, contextual property
failures, observable named fallback evidence, notebook hardening, and the final
integrated repository gates.
