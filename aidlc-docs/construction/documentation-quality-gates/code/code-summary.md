# Documentation and Executable Quality Gates Code Summary

## Outcome

The package now presents one process-engineer workflow rooted at
`PinchProblem` and `PinchWorkspace`. Read the Docs publishes the complete
interaction model and an operation-level tutorial coverage map. Every one of
the 129 live problem, workspace, target, all-period, component, design-result,
and plot operations has a primary executable tutorial and an execution profile.

## Implementation

- Added a generated operation manifest at `docs/_data/tutorial-coverage.csv`
  and exact live-inventory drift tests.
- Rebuilt the package-root, problem, workspace, guide, capability, and example
  documentation around explicit execution and cached observation.
- Documented the exact `HeatExchangerNetwork.model_dump(mode="json")` bridge
  through `TargetInput.network`, including title-case `StreamID` endpoint roles.
- Removed the retired `pinch_analysis_service`, workflow-string workspace
  execution/comparison layer, and dead variant-view contracts without aliases.
- Reduced persisted workspace bundles to schema version 3 case inputs.
- Made Heat Pump and turbine-model selection private call-local state;
  `HPR_TYPE` and `POWER_TURB_MODEL` are rejected as user configuration and are
  absent from configuration catalogs and `problem.config`.
- Updated release, import, notebook, documentation, and package-content guards
  for the clean two-class root contract.

## Verification

- Complete non-solver suite: 2,079 passed, 4 solver tests deselected.
- Architecture, packaging, and root workflow checkpoint: 121 passed.
- Documentation, notebook, and operation-manifest checkpoint: 32 passed.
- Warning-as-error Sphinx build: passed with all intersphinx inventories.
- Repository Ruff lint: passed.
- Repository Ruff format: 458 files formatted.
- Wheel and source distribution: built successfully.
- Isolated installed-wheel smoke: passed outside the source checkout, including
  root imports, a direct target solve, workspace construction, CLI help, and all
  eighteen packaged notebooks.
- Patch whitespace validation: passed.

## Extension Compliance

- Security: disabled; N/A because no enabled security extension applies.
- Resiliency: disabled; N/A because no enabled resiliency extension applies.
- Partial PBT: compliant for argument precedence, ordered case batches, and
  multiperiod aggregation through fixed-seed generated properties. Tutorial and
  documentation drift is guarded by deterministic inventory and execution
  contracts.
