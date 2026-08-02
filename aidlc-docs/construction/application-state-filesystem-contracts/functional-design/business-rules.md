# Business Rules

## Workspace Case Identifiers

### CASE-01: Required string

An identifier must be a `str`. Runtime calls with another type fail before case
state changes. Pydantic bundle input reports the corresponding validation error.

### CASE-02: Exact non-empty form

The value must be non-empty and equal to `value.strip()`. Leading/trailing
whitespace is rejected rather than removed.

### CASE-03: One portable path component

Reject:

- `.` and `..`;
- `/` and `\\`;
- ASCII control characters and DEL;
- `<`, `>`, `:`, `"`, `|`, `?`, and `*`;
- values ending with a period;
- Windows device names `CON`, `PRN`, `AUX`, `NUL`, `COM1` through `COM9`, and
  `LPT1` through `LPT9`, case-insensitively and including an extension.

Spaces, Unicode letters, hyphens, underscores, and internal periods remain
valid when no rejection rule applies.

### CASE-04: Boundary coverage

Apply the shared rule to:

- `PinchWorkspace` constructor `baseline_name`;
- explicit `load(case_name=...)`;
- `_set_case_input` and internal case creation;
- `scenario(name, ...)`;
- workspace bundle `baseline_name` and every `cases` key;
- batch export before path composition.

### CASE-05: No aliasing

Never sanitize, trim, case-fold, or substitute an invalid case identifier. Valid
identifiers retain exact spelling and case.

## Batch Export

### PATH-01: Path-native composition

Use `Path` joining, never slash-containing string interpolation.

### PATH-02: Resolved containment

The resolved case directory must satisfy
`case_directory.is_relative_to(export_root)`. A failure is recorded in the
per-case batch errors without aborting unrelated cases.

### PATH-03: Original result identity

Result/error mappings use the original validated case identifier, not a
filesystem slug.

## Problem Input Observation

### STATE-01: Detached return

Every non-`None` `problem_data` return is a deep copy. Nested dictionaries,
lists, Pydantic models, and serialized HEN payloads share no mutable identity
with authoritative input.

### STATE-02: No implicit rebuild

Reading `problem_data` does not validate, canonicalize, rebuild, invalidate, or
solve the problem.

### STATE-03: Explicit mutation ownership

Only existing explicit application methods may update authoritative input or
prepared state.

## Multiplier Mutation

### MULT-01: Prepared root first

Call `_require_prepared_root_zone()` before `get_subzone()`.

### MULT-02: Preserve numeric behavior

Finite non-negative values are used. Existing invalid-value warning/default
behavior remains unchanged.

### MULT-03: Preserve invalidation

After mutation, clear `_results`, `_last_target_run_spec`, and `_period_results`
exactly once and return the prepared root zone.

## Workbook Allocation

### FILE-01: Exclusive ownership

Allocation succeeds only through exclusive file creation. An existing path is
never selected or overwritten by allocation.

### FILE-02: Readable stable shape

Names retain the sanitized project prefix, timestamp, optional collision suffix,
and `.xlsx` extension. The exact generated filename is not a public contract.

### FILE-03: Concurrent uniqueness

Two allocation calls targeting the same directory and frozen timestamp must
return different paths, including when executed concurrently.

### FILE-04: Failure cleanup

If workbook writing does not complete successfully, remove that invocation's
reserved file. Do not remove an existing file or another invocation's path.

### FILE-05: Successful artifact preservation

Successful exports retain current workbook sheets, data, return type, and
destination semantics.

## Error Contract

- Invalid case identifiers: `ValueError` at runtime and Pydantic validation
  errors for bundle input.
- Export containment failure: per-case `ValueError` captured by the batch.
- Unloaded multiplier update: canonical `RuntimeError` containing
  `No input loaded`.
- Allocation filesystem failures: propagate the original `OSError`.
- Workbook writer failures: propagate the original exception after cleanup.

