# Repository Issue Remediation Requirements

## Intent Analysis

- **User request**: Convert the six confirmed repository findings into an
  implementation-ready remediation plan.
- **Request type**: Correctness fixes, contract hardening, internal refactoring,
  and current-documentation repair.
- **Scope**: Multiple existing components in the application, contracts,
  reporting, scripts, tests, and AI-DLC documentation.
- **Complexity**: Moderate. Each repair is bounded, but filesystem containment,
  mutable-state isolation, concurrent export naming, and Python import identity
  require explicit invariants.
- **Compatibility policy**: Clean break. No aliases, deprecation paths, or
  permissive handling of unsafe case identifiers will be introduced.

## Functional Requirements

### FR-1: Workspace case identifiers and batch export containment

1. Workspace case identifiers must be non-empty strings with no leading or
   trailing whitespace.
2. Identifiers must reject path separators, control characters, dot-only path
   components, Windows-reserved filename characters, and Windows-reserved
   device names.
3. Validation must apply consistently to constructor `baseline_name`, `load`,
   `scenario`, internal case creation, and workspace bundle loading.
4. Batch export must use standard-library path operations, resolve the case
   directory, and prove that it remains beneath the resolved destination without
   violating the application-layer dependency boundary.
5. `CaseBatchResult` keys and workspace display names remain the canonical case
   identifiers.
6. Valid existing identifiers such as `baseline`, `retrofit`, and
   `Retrofit 2026` remain valid.

### FR-2: Detached problem-input observation

1. `PinchProblem.problem_data` must return a detached deep snapshot of the
   loaded input rather than the mutable internal object.
2. `PinchWorkspace.problem_data` must inherit the same behavior through its
   active-case delegation.
3. Mutating a returned mapping or schema must not alter validation,
   `to_problem_json()`, prepared streams, cached results, or persisted workspace
   bundles.
4. Supported mutations continue through explicit rebuilding operations such as
   `load`, `update_options`, and `set_dt_cont_multiplier`.

### FR-3: Exact OpenHENS checkout identity

1. The comparison script must load OpenHENS from the exact checkout supplied by
   `openhens_root`.
2. Cached `openhens` modules and competing installations must not satisfy the
   prerequisite check.
3. Every required module with a source file must resolve beneath the requested
   checkout.
4. The verified `OpenHENS` callable must be passed into source execution; the
   execution path must not perform a second unverified import.
5. Temporary `sys.path` and `sys.modules` changes must be restored after the
   comparison or after an import failure.

### FR-4: Collision-free workbook allocation

1. Every workbook export must receive a unique path, including exports started
   within the same second and concurrent exports.
2. Path allocation must use exclusive creation rather than timestamp uniqueness
   alone.
3. The filename must retain a readable sanitized project prefix and `.xlsx`
   suffix.
4. Failed workbook creation must remove its reserved empty or partial file.

### FR-5: Consistent unloaded-problem error

1. `PinchProblem.set_dt_cont_multiplier()` must use the prepared-root-zone
   guard before accessing a zone.
2. An unloaded problem must raise the established actionable `RuntimeError`
   rather than an internal `AttributeError`.
3. A loaded problem whose root zone needs rebuilding must continue to rebuild
   lazily before applying the multiplier.

### FR-6: Current contract documentation

1. Active AI-DLC state and reverse-engineering documents must describe the
   canonical package-root imports for `PinchProblem` and `PinchWorkspace`.
2. Current-contract statements about `OpenPinch.main` and
   `pinch_analysis_service` must be removed.
3. Audit records and explicitly historical implementation artifacts must remain
   unchanged as historical evidence.
4. A scoped stale-symbol guard must prevent retired API statements from
   returning to current-state and reverse-engineering documents.

## Non-Functional Requirements

- **Safety**: Batch exports cannot write outside the selected destination.
- **Reliability**: Workbook allocation remains unique across threads and
  processes supported by the local filesystem.
- **State consistency**: Public observation cannot mutate the authoritative
  problem input behind prepared runtime state.
- **Determinism**: OpenHENS comparison results identify the requested source
  checkout rather than ambient interpreter state.
- **Portability**: Case identifier and export behavior must work on POSIX and
  Windows paths.
- **Maintainability**: Shared validation is implemented once and used by both
  runtime workspace entry points and bundle schemas.
- **Dependencies**: No new runtime dependency is introduced.
- **Performance**: Snapshot copying and path validation must not affect solver
  execution; their overhead is bounded by input size and export frequency.

## Scope Exclusions

- No solver formulation, targeting calculation, or HEN result changes.
- No compatibility aliases or legacy case-name normalization.
- No new public mutation API for individual stream fields.
- No renaming of existing valid workbook sheets or report columns.
- No rewrite of historical audit entries.

## Acceptance Criteria

1. All six confirmed reproductions have focused regression tests that fail on
   the current implementation and pass after remediation.
2. Unsafe case identifiers are rejected at runtime and during bundle loading;
   valid case identifiers export only below the requested destination.
3. Mutating `problem.problem_data` or `workspace.problem_data` snapshots cannot
   affect internal or serialized state.
4. Foreign cached OpenHENS modules are rejected, and verified modules originate
   from the requested checkout.
5. Repeated and concurrent workbook exports never return the same path and do
   not leave failed reservations behind.
6. Empty-problem multiplier updates raise the canonical `RuntimeError`.
7. Current API documentation contains no retired `OpenPinch.main` or
   `pinch_analysis_service` claims.
8. Focused tests, the complete non-solver suite, Ruff, warning-free Sphinx,
   package build, isolated-wheel smoke, and patch hygiene pass.

## Completion Evidence

- [x] AC-1: all six reproductions have regression-first coverage.
- [x] AC-2: runtime/bundle identifiers and resolved batch containment pass.
- [x] AC-3: problem/workspace input snapshots are detached.
- [x] AC-4: exact-checkout module identity and factory injection pass.
- [x] AC-5: repeated/concurrent workbook allocation and cleanup pass.
- [x] AC-6: unloaded multiplier mutation raises the canonical error.
- [x] AC-7: scoped current-document guard reports no retired claims.
- [x] AC-8: 2,181 non-solver tests, 458 focused HEN/OpenHENS tests, Ruff,
  53-source warning-free Sphinx, distribution build, installed-wheel smoke, and
  patch hygiene pass.
