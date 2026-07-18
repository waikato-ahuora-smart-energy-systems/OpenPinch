# Residual Compatibility Shim Cleanup Requirements

## Intent Analysis

- **User request**: Scan the repository for remaining compatibility shims, clarify which compatibility-shaped behaviors are intentional, and prepare a repository-wide clean-break cleanup.
- **Request type**: Internal refactoring, contract hardening, developer-tool cleanup, and documentation cleanup.
- **Scope estimate**: Multiple components spanning numerical helpers, HEN solver integration, unit policy, developer scripts, tests, and Read the Docs content.
- **Complexity estimate**: Moderate. The implementation is small, but compatibility must be distinguished from canonical engineering normalization, algorithmic resilience, and solver invariants.
- **Persona**: Process engineers using the canonical `PinchProblem` and `PinchWorkspace` workflows.
- **Compatibility policy**: The repository must not retain aliases, migration behavior, dependency-version retries, monkeypatches, or transition pages merely to preserve retired behavior.

## Confirmed Decisions

1. The cleanup applies repository-wide, including shipped code, developer scripts, tests, and documentation.
2. `g_ineq_penalty()` will accept a canonical `PenaltyForm` enum rather than case-insensitive or alternative string spellings.
3. The OpenHENS comparison script will not monkeypatch upstream code. It will require a supported upstream capability and fail with a clear prerequisite error when that capability is absent.
4. The remaining legacy RTD page and generated-index entry will be removed. Stale test terminology will be renamed, and regression guards will prevent reintroduction.
5. The Pyomo 6.10-or-newer `SolverFactory.available(exception_flag=False)` API will be called directly. The positional-call `TypeError` retry will be removed.
6. The warning-backed missing-Couenne workflow fallback remains because it is current algorithmic resilience rather than compatibility behavior.
7. Shared dimensional unit overrides remain. Internal `aliases` terminology will be renamed to `unit_groups` or `override_keys` to describe the current contract accurately.
8. Fluid phases continue accepting enum members, short codes, descriptive values, case differences, and `vapor` or `vapour`, while serialization emits one canonical value.
9. `Value` continues accepting numeric scalars, period arrays, `Value`, Pint quantities, canonical serialized mappings, Pydantic models, and foreign value-with-unit objects.
10. Compact wire keys, optional-dependency guards, and the segmented-stream parent-axis solver shape invariant remain unchanged.

## Functional Requirements

### FR-1: Canonical penalty selection

- Define one `PenaltyForm` enum with members for the square and square-root-smoothing algorithms.
- Require `g_ineq_penalty()` callers to supply `PenaltyForm` values.
- Reject strings, including previously accepted canonical-looking, mixed-case, and spaced spellings.
- Migrate every production and test caller to the enum without changing either numerical algorithm.
- Do not export `PenaltyForm` from the root package; the root remains limited to `PinchProblem` and `PinchWorkspace`.

### FR-2: OpenHENS comparison prerequisite

- Remove `_install_openhens_compatibility()` and all assignments that modify imported OpenHENS modules or runner functions.
- Validate the exact upstream functions required by the comparison workflow before executing a comparison.
- Fail before comparison work begins with an actionable error identifying the unsupported OpenHENS checkout or missing capability.
- Do not add an adapter that recreates or silently substitutes the missing upstream behavior.

### FR-3: Current Pyomo API only

- Call `solver_factory.available(exception_flag=False)` once.
- Do not retry the call positionally after `TypeError` or other signature failures.
- Preserve the existing focused error when a correctly constructed solver factory reports that its solver is unavailable.

### FR-4: Canonical unit-group terminology

- Preserve the behavior of flat `INPUT_UNIT_*` and `OUTPUT_UNIT_*` configuration keys and their shared dimensional application.
- Rename internal dataclass fields, parameters, documentation, and tests that call shared dimensional override keys `aliases`.
- Use terminology such as `unit_groups` or `override_keys`; do not create transitional constructor parameters or properties for `aliases`.
- Preserve all current unit conversions and output-unit selection results.

### FR-5: Documentation and stale terminology cleanup

- Delete `docs/reference/api-lib.rst` and remove it from generated API navigation.
- Rename `test_validate_utilities_data_alias_executes` to describe utility validation rather than an absent alias.
- Add closed-contract checks for the removed legacy page, penalty string spellings, upstream monkeypatch assignments, Pyomo signature retry, and misleading unit-policy alias terminology.
- Historical release-note statements that accurately document prior breaking releases may remain.

### FR-6: Preserved canonical behavior

- Preserve compact JSON keys and exact input and HEN serialization round trips.
- Preserve current fluid-phase normalization and canonical serialization.
- Preserve documented `Value` and Pint/value-like coercion.
- Preserve optional-dependency guards and focused installation errors.
- Preserve the missing-Couenne algorithmic fallback and warning.
- Preserve the segmented-stream parent-axis zero required by current equation shapes.

## Non-Functional Requirements

- **Numerical stability**: Penalty calculations, unit conversions, HEN results, and workflow ordering must remain numerically unchanged.
- **Usability**: Process engineers must not need to learn new string selectors or duplicate shared unit configuration.
- **Maintainability**: Static terminology must distinguish canonical normalization, resilience, and invariants from compatibility behavior.
- **Packaging**: Root exports, optional dependency boundaries, wheel contents, and source-distribution contents must remain unchanged except for intentionally removed documentation.
- **Documentation**: Warning-free Sphinx validation must pass, and no deleted page may remain in generated navigation.
- **Repository safety**: Existing unrelated working-tree changes must be preserved.

## Test Requirements

1. Add example-based tests for both `PenaltyForm` members and rejection of every string form.
2. Use Hypothesis to verify penalty invariants across domain-constrained finite scalar and array inputs: output finiteness for finite bounded inputs, non-negativity for non-negative residuals, and stable scalar-versus-array aggregation semantics.
3. Keep Hypothesis shrinking enabled and execute with seed `20260715` in CI-compatible pytest runs.
4. Test that the comparison script accepts a supported upstream capability surface and fails clearly without modifying imported modules when the surface is incomplete.
5. Test that Pyomo availability is called once with `exception_flag=False` and that `TypeError` propagates rather than triggering a positional retry.
6. Test unit-group application across every field sharing a dimension and confirm direct mapping behavior remains unchanged after terminology replacement.
7. Retain the existing phase, `Value`, compact-wire, optional-dependency, Couenne-fallback, and segmented-stream invariant tests.
8. Extend stale-symbol or AST checks so removed compatibility mechanisms cannot return.
9. Run focused numerical, unit, input, HEN backend, developer-script, architecture, documentation, and packaging suites; then Ruff and warning-free Sphinx validation.

## Acceptance Criteria

- No runtime string spelling or case alias remains in `g_ineq_penalty()`.
- No OpenHENS module is monkeypatched by repository tooling.
- No Pyomo availability signature retry remains.
- Shared unit override behavior is unchanged and no longer described internally as aliases.
- `docs/reference/api-lib.rst` and its navigation entry are absent.
- The stale utility-validation test name is removed.
- Canonical engineering normalization, wire serialization, optional-dependency behavior, Couenne resilience, and segmented-stream equation shapes remain unchanged.
- The root package exports exactly `PinchProblem` and `PinchWorkspace`.
- Focused and complete applicable verification gates pass without warnings.

## Out of Scope

- Renaming compact JSON fields or configuration keys.
- Removing supported phase, Pint, value-like, or multiperiod input forms.
- Removing the missing-Couenne algorithmic fallback.
- Changing HEN equations, solver tensors, numerical tolerances, or synthesis ranking.
- Adding compatibility aliases, deprecation warnings, migration instructions, or transition pages.

## User Stories Assessment

User stories are skipped. This is an internal clean-break refactor with no new process-engineer workflow; existing package-usability stories already define the affected persona and canonical root workflow.

## Extension Compliance

- **Security Baseline**: Disabled; N/A because no security boundary changes.
- **Resiliency Baseline**: Disabled; N/A. The existing Couenne fallback is preserved by explicit functional requirement.
- **Partial Property-Based Testing**:
  - **PBT-02**: N/A because no serialization transformation is changed; existing round-trip properties remain required and must stay green.
  - **PBT-03**: Applicable to penalty-function and shared unit-group invariants.
  - **PBT-07**: Applicable; strategies must generate finite, bounded engineering residuals and valid unit-group mappings.
  - **PBT-08**: Applicable; shrinking and seed `20260715` are required.
  - **PBT-09**: Compliant; Hypothesis is the existing selected Python framework.

## Requirements Summary

The implementation removes the final genuine compatibility mechanisms without confusing them with canonical engineering flexibility. It introduces enum-only penalty selection, removes upstream monkeypatching and a Pyomo version retry, deletes the last legacy RTD transition page, and renames misleading unit-policy terminology. Current serialization, phase handling, value coercion, optional dependencies, solver resilience, and equation invariants remain stable.
