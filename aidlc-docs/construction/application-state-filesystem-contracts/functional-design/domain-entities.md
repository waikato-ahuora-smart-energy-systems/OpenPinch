# Domain Entities

## Existing Entities Retained

### `PinchProblem`

- **Authoritative state**: `_problem_data`, `_master_zone`, cached target and
  period results.
- **Observation**: `problem_data` returns a detached snapshot.
- **Mutation**: existing explicit lifecycle and option/multiplier methods.
- **Invariant**: public observation cannot mutate authoritative or prepared
  state.

### `PinchWorkspace`

- **Authoritative state**: validated named case inputs, live case cache, active
  case identifier, baseline identifier.
- **Invariant**: every stored identifier satisfies the shared portable case-name
  contract.
- **Relationship**: owns zero or more `PinchProblem` cases and delegates active-
  case observation.

### `PinchWorkspaceBundle`

- **Fields retained**: `schema_version`, `project_name`, `baseline_name`, and
  case mapping.
- **Invariant**: `baseline_name` and all mapping keys satisfy the same runtime
  identifier contract.
- **Serialization**: schema version remains `3`; no migration or compatibility
  path is added.

### `CaseBatchResult`

- **Fields retained**: immutable success and error mappings.
- **Invariant**: mapping keys are original validated case identifiers.
- **Behavior**: one case failure does not prevent unrelated case operations.

### Workbook Export Artifact

- **Identity**: one exclusively reserved filesystem path.
- **Lifecycle**: reserved, written, then retained on success or removed on
  unsuccessful completion.
- **Invariant**: one invocation owns one path; no two successful reservations
  share identity.

## Internal Value Objects

### Validated Case Identifier

A plain `str` that has passed the shared validator. No wrapper type or public
alias is introduced. Its validity is established at every storage boundary and
rechecked at the filesystem boundary.

### Resolved Export Root

An absolute normalized `Path` used solely as the containment authority for one
batch export call.

### Reserved Workbook Path

A `Path` whose directory exists and whose filename has been claimed atomically
by the current invocation. Reservation ownership ends when a successful workbook
is returned or failed-output cleanup completes.

## Relationships

- `PinchWorkspaceBundle` validates identifiers before `PinchWorkspace` stores
  cases.
- `PinchWorkspace` materializes `PinchProblem` cases and delegates detached
  `problem_data` observation.
- `_CaseBatch` maps validated workspace identifiers to contained case export
  directories.
- `PinchProblem.export_excel()` delegates workbook creation to reporting, which
  owns exclusive path allocation.

## Explicitly Absent Entities

- No compatibility identifier or sanitized-name alias.
- No mutable `ProblemDataView` facade.
- No export registry or persistent lock database.
- No frontend component.
- No new package-root export.

