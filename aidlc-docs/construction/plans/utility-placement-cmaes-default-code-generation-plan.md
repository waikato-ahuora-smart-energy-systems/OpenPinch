# Utility Placement CMA-ES Default Code Generation Plan

This corrective TDD amendment changes only the default utility-placement
black-box optimizer. Explicit backend selection remains unchanged.

## Scope

- Default `UtilityPlacementOptions.method` to `cmaes`.
- Retain `dual_annealing`, `cmaes`, `bo`, and `rbf_surrogate` as exact choices.
- Retain per-call selection through `options["method"]`.
- Align current requirements, functional design, and RTD guidance.
- Preserve the user's locally modified notebook 19; verify the canonical
  generated notebook in a temporary location instead of overwriting it.

## Execution steps

- [x] Step 1 - Change default-contract and coordinator tests to expect CMA-ES,
  then run them in RED against the dual-annealing default.
- [x] Step 2 - Change the public default to CMA-ES and align requirements,
  functional design, and RTD documentation.
- [x] Step 3 - Run focused regression gates and verify a temporary canonical
  notebook under the new default without changing the local notebook.
- [x] Step 4 - Run the broad feasible regression gate, review scope, update
  completion evidence, and commit only the amendment files to `develop`.

## Extension compliance

Property-Based Testing remains enabled. Existing deterministic backend,
selection, and utility-placement properties remain applicable. Security and
Resiliency remain disabled and N/A for this default-selection amendment.
