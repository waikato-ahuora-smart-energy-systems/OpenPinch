# Business Logic Model

## Scope

Unit 1 owns five related correctness flows inside existing package boundaries:

1. validate workspace case identifiers before storage or bundle construction;
2. contain every batch export directory beneath the selected root;
3. return detached snapshots of authoritative problem input;
4. acquire prepared analysis state before multiplier mutation;
5. reserve unique workbook paths before writing and clean failed outputs.

## Case Identifier Flow

1. A runtime caller or workspace bundle supplies a case identifier.
2. The shared validator confirms the value is a string and preserves the exact
   original value only when all rules pass.
3. Invalid values raise immediately before `_case_inputs`, active-case state, or
   bundle state changes.
4. Runtime boundaries call the same validator used by the Pydantic bundle
   contract.
5. Stored valid identifiers remain the keys returned by `list_cases()` and
   `CaseBatchResult`.

The validator performs rejection, not normalization. Two inputs can never
collapse onto one canonical alias because whitespace and unsafe characters are
not rewritten.

## Batch Export Flow

1. Convert the requested destination to a `Path` and resolve it without
   requiring it to exist.
2. Validate the selected stored case identifier again at the filesystem
   boundary as defense in depth.
3. Compose `export_root / case_name` with `Path`, then resolve the result.
4. Require the resolved case directory to be relative to the resolved export
   root. Otherwise raise before calling the case exporter.
5. Call each case exporter independently and retain the established batch
   success/error isolation.
6. Store results and errors under the original case identifier.

Resolution accounts for existing symlinked path components. The operation does
not promise containment against a hostile concurrent process replacing path
components after validation; OpenPinch remains a local in-process library rather
than a privileged file service.

## Problem Input Observation Flow

1. `_problem_data` remains the authoritative loaded input object.
2. `PinchProblem.problem_data` returns `deepcopy(_problem_data)` or `None`.
3. `PinchWorkspace.problem_data` delegates to the active case and therefore
   receives the same detached value.
4. Callers may inspect or mutate the returned object, but those mutations have
   no effect on validation, prepared zones, cached results, serialization, or
   workspace persistence.
5. Intentional changes continue through explicit application mutation methods,
   which validate, rebuild, and invalidate dependent state.

The property does not call `to_problem_json()`: it preserves the existing raw
input type family while isolating object identity.

## Multiplier Mutation Flow

1. Coerce and validate the requested multiplier using existing rules.
2. Call `_require_prepared_root_zone()` before any zone lookup.
3. If no input is loaded, propagate the canonical actionable `RuntimeError`.
4. If input exists but prepared state is absent, allow the existing lazy rebuild.
5. Resolve the requested zone, apply the multiplier, clear cached results,
   target-run specification, and period results, then return the prepared root.

## Workbook Allocation Flow

1. Sanitize the project prefix using the existing filename policy.
2. Create the destination directory before path allocation.
3. Construct a readable base name using date, time, and microseconds.
4. Attempt exclusive file creation with standard-library `O_CREAT | O_EXCL` and
   normal umask-respecting permissions.
5. On collision, append a monotonically increasing suffix and retry.
6. Close the reservation descriptor and return the reserved `.xlsx` path.
7. Write the workbook to the reserved path.
8. Track successful completion; on every unsuccessful exit, unlink the reserved
   empty or partial file with `missing_ok=True`.

Exclusive creation is the uniqueness authority. Timestamp precision improves
readability and reduces retries but is not relied on for correctness.

## Transaction Boundaries

- Identifier validation completes before workspace state mutation.
- Path containment completes before exporter invocation.
- Snapshot copying completes before public return.
- Prepared-root acquisition completes before multiplier mutation.
- Exclusive path reservation completes before workbook writing.
- Failed workbook writes remove only the path reserved by that invocation.

