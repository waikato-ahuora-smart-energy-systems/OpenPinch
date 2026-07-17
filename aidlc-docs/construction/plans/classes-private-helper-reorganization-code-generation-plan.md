# Private Helper Reorganization Code Generation Plan

## Part 1 - Approved Planning

- [x] Confirm the public API break and parent-owned record policy.
- [x] Inventory current helper modules, imports, tests, documentation, and
  dependency-cycle constraints.
- [x] Establish the owner-oriented target hierarchy and dependency order.
- [x] Obtain explicit implementation approval through the user's request to
  implement the supplied plan.

## Part 2 - Generation

### Domain Foundations

- [x] Move the three runtime record classes to private owner modules and remove
  public aliases and exports.
- [x] Add segment input normalization and preserve transactional parent
  ownership, rollback, ordering, copying, pickling, and serialization.
- [x] Reorganize stream, value, collection, and problem-table helpers into the
  approved owner packages and delegate extracted responsibilities.
- [x] Run focused domain and property tests with seed `20260715`.

### Problem Orchestration

- [x] Replace `_problem` with `_pinch_problem` and group accessors, input,
  output, periods, and targeting helpers by responsibility.
- [x] Extract private targeting and multiperiod execution from
  `pinch_problem.py` while retaining public delegates.
- [x] Separate report/schema validation from semantic validation.
- [x] Run focused problem orchestration tests.

### Workspace and Integration

- [x] Replace `_workspace` with `_pinch_workspace` and extract case inputs,
  comparison, execution, state/cache, and views responsibilities.
- [x] Update services, schemas, tests, notebooks, documentation, and release
  notes for private runtime record paths and removed public API names.
- [x] Add structural, cold-import, public API, parent-construction, and required
  generated invariant tests.

### Build and Test

- [x] Run the complete seeded non-solver suite with coverage.
- [x] Run available solver tests.
- [x] Run Ruff lint and format checks.
- [x] Build warning-free Sphinx documentation and parse notebooks.
- [x] Build wheel and sdist in isolation.
- [x] Run stale-path/public-API searches and `git diff --check`.
- [x] Record final evidence, extension compliance, and stage completion.
