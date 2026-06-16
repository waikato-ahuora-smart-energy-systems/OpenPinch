# HENS-03 Adversarial Review

## Findings

No findings. HENS-03 is cleared.

## Residual Risks

- I did not rerun the full pytest command from the implementation notes because this review was constrained to read-only work except for this findings file. I did run `rtk git diff --check -- . ':!.DS_Store'`, which passed.
- Nine-stream is validated as the final-verification fixture only; HENS-03 intentionally does not include a Nine-stream adapter parity snapshot. Later solver/parity tasks still need to prove Nine-stream final verification against the baseline artifacts.
- Nine-stream preparation currently records an OpenPinch-added default hot utility in the structural fixture snapshot while the converted JSON itself contains the source utility rows. This follows the existing `prepare_problem(...)` utility-completion path, but later parity work should account for it when comparing OpenHENS arrays.
- Utility prices are stored through `UtilitySchema.price` and numerically passed into the adapter for OpenHENS annual cost equations. The converted fixture uses the current OpenPinch price unit convention, so later equation-move work should keep the annual-cost convention explicit.

## Scope Review

- The dirty HENS-03 worktree is limited to the task document, the private adapter, conversion script, converted fixtures, fixture/adapter snapshots, and validation tests. `.DS_Store` remains modified but outside the HENS-03 scoped status set.
- I found no solver model execution, moved GEKKO/Pyomo equations, public runtime CSV synthesis API, public raw-input runner, or public TargetInput-direct synthesis path in the HENS-03 changes.
- The adapter package barrel stays empty, so `problem_to_solver_arrays` is not root-exported or promoted through `OpenPinch.services`.

## Fixture Review

- Both required fixtures exist at `tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.json` and `tests/fixtures/openhens/Nine-stream-Linnhoff-and-Ahmad-1999-1.json`, with reordered variants present for both.
- The fixtures are standard `TargetInput` payloads using `streams`, `utilities`, `options`, and `zone_tree`. Process streams use `StreamSchema` fields, utilities use `UtilitySchema` fields, and source temperatures are explicit `K` values.
- Utility operating costs are stored in `UtilitySchema.price`. Shared exchanger costing is mapped to `FIXED_COST`, `VARIABLE_COST`, and `COST_EXP` in `TargetInput.options`.
- HEN controls are stored in `TargetInput.options` with `HENS_*` OpenPinch configuration keys backed by `CONFIG_FIELD_SPECS`; no `OPENHENS_*` compatibility aliases are added.
- Source OpenHENS CSVs are used only by `scripts/convert_openhens_fixtures.py` as development/source-material conversion input. The converter is outside the `OpenPinch` package namespace and is not exported as a runtime API.

## Adapter Review

- `problem_to_solver_arrays(problem, dTmin)` accepts only `PinchProblem`, requires a prepared `Zone`, rejects non-positive `dTmin`, and reads arrays from `problem.master_zone` stream and utility collections.
- The adapter rejects raw fixture rows, raw `TargetInput`, HEN schema records, cached array payloads, JSON snapshots, and unprepared `PinchProblem()` in the test suite.
- Missing prepared temperature contributions fall back to `dTmin / 2`, preserving the source OpenHENS behavior for cases without explicit `T cont`.
- Adapter output includes HEN option values, utility prices, costing coefficients, stage selection, solver controls, labelled axis maps, stream identities, utility identities, array shapes, and unit conventions.
- Four-stream has the required routine adapter snapshot at `openhens_baseline_results/adapter_snapshots/Four-stream-Yee-and-Grossmann-1990-1/dTmin-14.json`; it includes source path/hash, source fixture hash, extraction command, source OpenHENS arrays, active dTmin, shapes, axis maps, identities, and unit conventions.

## Evidence Review

- The OpenHENS source audit supports excluding process stream `h_cost` and `c_cost` from the runtime fixture schema: source `CaseStudy.to_legacy_arrays(...)` emits them and `GenericHENModel` initializes them, but active objective, post-processing, and verification paths use `hu_cost`, `cu_cost`, and exchanger cost arrays.
- Row and field context exists for converter errors through `ConversionError` messages such as `process streams row <n>: <field> must be numeric`; the test suite covers this path.
- The tests are meaningful for HENS-03: they validate both converted fixtures through `TargetInput` and `PinchProblem`, assert Kelvin stream units and utility prices, compare Four-stream dTmin=14 adapter arrays against the source payload snapshot, validate Nine-stream fixture loading, cover reordered fixtures, and include negative bypass tests.
- HENS-03 task checkboxes and implementation notes are consistent with the evidence I reviewed. Definition of Done checkboxes remain intentionally unchecked pending adversarial review clearance.
