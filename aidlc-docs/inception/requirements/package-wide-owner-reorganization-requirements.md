# Package-Wide Owner-Oriented Reorganization Requirements

## Status

Approved for implementation by the user's explicit implementation request.

## Intent

Complete the owner-oriented private-helper cleanup across `OpenPinch`, preserving
documented user-facing APIs while removing unsupported runtime and solver-state
records from public barrels and documentation.

## Functional Requirements

- Complete the existing class extractions so semantic validation, interval
  insertion, segment transactions, value coercion/units, and workspace view
  shaping are substantively owned by private responsibility modules.
- Split HEN synthesis schemas into concrete common, topology, method, task, and
  result modules while preserving all public imports, model names, validation,
  JSON schemas, and legacy barrel-qualified public-schema pickle loading.
- Convert `OpenPinch.classes`, `OpenPinch.lib`, and `OpenPinch.lib.schemas` to
  typed lazy barrels with unchanged `__all__` contracts. Keep the root package
  eager and unchanged.
- Reorganize Process MVR, direct MVR, multiperiod HPR, graph construction,
  energy-transfer processing, Streamlit graphing, and network-grid rendering
  around their owning public services.
- Decompose HEN base, stagewise, pinch-design, and result-extraction internals
  using explicit model-state composition helpers without introducing mixins.
- Preserve equation order, solver axes, warm-start behavior, tolerances,
  numerical ordering, model dumps, result structures, and documented advanced
  import paths.

## Curated Compatibility Policy

- Preserve root APIs, public schemas, `ProcessMVRComponent`, direct-MVR public
  models and solve functions, HEN result models, and documented HEN equation
  model classes.
- Remove public aliases and documentation for `ProcessMVRStreamRecord`,
  `PreparedHPRPeriodCase`, `StreamlitGraphSet`, graph specification/metadata
  records, `InternalHeatExchangerNetworkProblem`,
  `ModelSliceUnavailableError`, and `SolverRun`.
- Internal records may remain inspectable through their owners but cannot be
  imported or constructed through public modules.
- No compatibility package, migration, dependency change, data migration,
  version bump, or deployment work is included.

## Quality Requirements

- Add structural and cold-import tests for concrete ownership, lightweight
  barrels, retired paths, optional-dependency isolation, and forbidden names.
- Preserve synthesis schema/model-dump round trips and public-schema pickle
  compatibility.
- Add deterministic and generated ownership, copying, pickling, rollback,
  ordering, and round-trip invariants using seed `20260715` where applicable.
- Run focused checks after each dependency-ordered unit, then the complete
  non-solver suite with coverage, available solver tests, Ruff lint/format,
  warning-free Sphinx, notebook parsing, isolated wheel/sdist builds, stale-path
  and public-API checks, and `git diff --check`.

## Workflow Decisions

- Existing reverse-engineering artifacts and the completed classes refactor are
  reused; focused package analysis is sufficient.
- User stories, infrastructure design, and Operations are skipped because this
  is an internal package refactor with no new user workflow or deployment.
- The user's supplied plan is the explicit Requirements, Workflow Planning,
  Units Generation, and Code Generation Part 1 approval.
- Security and Resiliency extensions remain disabled. Partial Property-Based
  Testing rules PBT-02, PBT-03, PBT-07, PBT-08, and PBT-09 remain blocking
  where applicable.
