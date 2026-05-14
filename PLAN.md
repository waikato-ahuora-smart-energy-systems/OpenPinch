# OpenPinch Frontend-Enablement Roadmap

OpenPinch should be developed as the analysis backend for an interactive GUI,
not as a CLI-first product surface. The first backend slice is now in place:
`PinchWorkspace` exists as the public multi-case orchestration layer,
`ScenarioWorkspace` has been removed, frontend-oriented schemas exist for
validation and solved results, `PinchWorkspaceBundle` round-trips as JSON, and
baseline-versus-variant comparison is available. The packaged notebooks now
teach the script-native API directly, including
`PinchWorkspace(source="sample_case.json", ...)` for bundled sample cases. The
roadmap below focuses on hardening those contracts and extending them into a
stable GUI backend.

Deep documentation restructuring can continue in `RTD_OVERHAUL_PLAN.md`, but
the main roadmap here should stay centered on backend product shape.

## Now

### PinchWorkspace contract hardening

- Why: the workspace foundation now exists, but a GUI needs explicit and
  durable semantics for case lifecycle, workflow execution, and cached result
  invalidation.
- Frontend user flow unlocked: create, edit, rerun, and revisit named
  study cases with predictable behavior instead of relying on notebook-style
  conventions.
- Backend/public surface: keep `PinchWorkspace` as the multi-case layer above
  `PinchProblem` and tighten the contract around the current public entry
  points: `PinchWorkspace(source=...)`, `from_payload(...)`,
  `from_problem(...)`, `case(...)`, `use_case(...)`, `copy_case(...)`,
  `list_cases()`, `list_variants()`, `set_variant_payload(name, payload, *,
  base=None)`, `solve_variant(name, *, workflow="target",
  workflow_options=None)`, `compare_cases(...)`, `compare_variants(...)`,
  `save_bundle(path)`, and `load_bundle(path)`. Add the missing lifecycle
  operations a GUI will need next, such as rename, remove, and explicit cache
  reset, while keeping full-payload editing as the primary mutation model.

### Frontend-ready response stabilization

- Why: the package now emits serializable view models, but they still need a
  stable contract for identifiers, warnings, and workflow status before a GUI
  can depend on them long term.
- Frontend user flow unlocked: bind validation errors to fields, preserve graph
  and zone selections across reruns, render solved results without bespoke
  translation logic, and expose advanced workflows with clear support signals.
- Backend/public surface: stabilize validation reports, summary cards and
  tables, graph catalogs and graph payloads, and shifted and real
  problem-table views. Standardize durable identifiers for zones, targets,
  graphs, streams, and utilities; define a consistent warning and status
  taxonomy for solved, invalid, error, partial, advanced, and unsupported
  workflows; and round out configuration metadata with labels, enum choices,
  numeric bounds when known, grouping, and support level.

### Dashboard consumption of backend contracts

- Why: the current Streamlit and dashboard surfaces should prove the backend
  contracts rather than define their own data-shaping logic in parallel.
- Frontend user flow unlocked: a demo GUI can exercise the same workspace and
  serialized result surfaces that a future dedicated frontend will consume.
- Backend/public surface: refactor dashboard entrypoints to consume
  `PinchWorkspace` views and related frontend schemas directly, and remove
  duplicate translation logic that currently lives only in presentation-layer
  code.

## Next

### Versioned study-bundle evolution

- Why: bundle save/load now exists, but long-lived GUI usage needs explicit
  schema guarantees, migration policy, and cache compatibility rules.
- Frontend user flow unlocked: save a study today, reopen it after backend
  upgrades, share it with another user, and decide when cached result snapshots
  should be reused versus recomputed.
- Backend/public surface: evolve the JSON-first `PinchWorkspaceBundle`
  contract with explicit versioning rules, migration hooks, cache invalidation
  semantics, and documented compatibility expectations.

### Richer interactive comparison surfaces

- Why: the current comparison layer covers summary deltas and structural table
  diffs, but GUI-driven studies need deeper and more deterministic comparison
  payloads.
- Frontend user flow unlocked: compare variants side by side by target, review
  graph-set changes, inspect table-level differences in more detail, and handle
  mixed-workflow studies without frontend aggregation code.
- Backend/public surface: expand `compare_variants(...)` to include richer
  target-by-target comparison tables, graph selection metadata, more detailed
  problem-table diffs, and explicit behavior when compared variants were solved
  with different workflows.

### Editable form metadata and guided asset discovery

- Why: a GUI needs more than raw payloads to build safe editors and guided
  onboarding flows.
- Frontend user flow unlocked: render grouped editors with the right widgets
  and support labels, and offer packaged notebooks and sample cases as guided
  starting points rather than opaque files.
- Backend/public surface: extend configuration metadata into a broader editor
  schema for payload sections and add machine-readable metadata for packaged
  sample cases and notebooks, including descriptions, workflow tags, support
  level, and recommended direct loader snippets such as
  `PinchWorkspace(source="sample_case.json", ...)`.

## Maintenance / Architecture

### Backend-first product framing

- Why: the package needs a stable architectural boundary so demo surfaces do
  not accidentally define the contracts that a future GUI must inherit.
- Frontend user flow unlocked: multiple frontend implementations can sit on the
  same backend contracts without depending on Streamlit-specific behavior.
- Backend/public surface: keep `PinchProblem` as the single-case compute
  wrapper, place multi-case orchestration in `PinchWorkspace`, and treat
  the current Streamlit/dashboard layer as a contract consumer and proving
  ground rather than the product center.

### CLI and documentation drift cleanup

- Why: CLI and documentation accuracy still matter, but they should follow the
  backend contracts rather than drive the main product roadmap.
- Frontend user flow unlocked: frontend teams can trust that examples, docs,
  and demo commands reflect the same supported backend surface they are
  integrating against.
- Backend/public surface: keep CLI and documentation drift work as maintenance
  only, including alignment of documented workflows, examples, and packaged
  learning assets with the supported backend APIs. The packaged notebooks
  should continue to prefer the direct sample-case loader path over ad hoc
  `json.loads(read_sample_case(...))` setup.

### Optional dependency profiles

- Why: backend deployments, notebook usage, and frontend integrations do not
  all need the same runtime footprint.
- Frontend user flow unlocked: GUI-oriented deployments can install the minimal
  backend runtime while notebook, graphing, or demo surfaces remain opt-in.
- Backend/public surface: split optional features into extras where practical,
  especially for notebook, dashboard, and advanced cycle workflows.

### Docs and public-surface CI gates

- Why: once frontend-facing contracts become explicit, drift between code,
  docs, sample assets, and exports becomes more expensive.
- Frontend user flow unlocked: frontend integrators can rely on documented
  backend contracts staying synchronized with actual package behavior.
- Backend/public surface: add CI checks for package-root exports, workspace
  schemas, packaged assets, major `PinchProblem` workflow members, and
  documentation pages that describe supported backend capabilities.

### Public API stability metadata

- Why: frontend integrations need machine-readable support signals, not support
  status buried only in prose.
- Frontend user flow unlocked: a GUI can distinguish stable, advanced, partial,
  and unsupported backend surfaces before exposing them to end users.
- Backend/public surface: add maintained metadata or a registry that identifies
  support level for public modules, workflows, packaged assets, and
  frontend-facing contracts.

## Deferred Research / Partial Subsystems

### Exergy targeting restoration

- Why: exergy analysis is valuable, but it is still a partial subsystem and
  should not compete with the core frontend-enablement roadmap.
- Frontend user flow unlocked: none in the immediate product cycle; this should
  remain deferred until the backend workspace and view-model contracts are
  stable.
- Backend/public surface: decide later whether to restore, explicitly mark
  experimental, or remove the public surface around
  `OpenPinch/services/exergy_analysis/exergy_targeting_entry.py`.

### Energy-transfer analysis restoration

- Why: large commented-out or placeholder analysis modules add maintenance cost
  and user confusion when left in an ambiguous state.
- Frontend user flow unlocked: none in the immediate product cycle; deferred
  research should not shape the first GUI backend contracts.
- Backend/public surface: decide later whether to restore, explicitly mark
  experimental, or remove the public surface around
  `OpenPinch/services/energy_transfer_analysis/energy_transfer_analysis.py`.

### Other partial or commented-out subsystems

- Why: the package still contains expert-only or unfinished surfaces that are
  not ready to anchor frontend product design.
- Frontend user flow unlocked: clearer separation between production-ready
  backend contracts and research or partial capability.
- Backend/public surface: review remaining partial, placeholder, or heavily
  commented-out subsystems and assign each one a clear status: restore, mark
  experimental, or retire.

## Acceptance Scenarios

- A GUI can create a baseline study, add named variants with fully edited
  payloads, solve them, compare them, and persist them without custom notebook
  loops.
- Validation responses include stable field paths and readable messages that
  map directly onto frontend form errors.
- Summary, graph catalog, graph payload, and problem-table outputs are stable
  serialized contracts rather than frontend-inferred structures.
- Study bundles round-trip cleanly with baseline, variants, workflow settings,
  schema versioning, and optional cached results intact.
- Multi-variant comparison outputs remain stable even when zones, target types,
  or workflows repeat.
- The Streamlit/dashboard layer can consume the workspace and view-model
  contracts without bespoke translation logic.
- Packaged sample cases and notebooks demonstrate the same supported loader and
  orchestration APIs that frontend clients and script users are expected to
  call directly.

## Assumptions

- The package should be backend-first for a separate GUI rather than centered
  on CLI expansion.
- Full case editing is in scope from the start, so roadmap features should
  accept whole editable payloads rather than only parameter tweaks.
- Study persistence is first-class, and the saved unit is a versioned
  baseline-plus-variants bundle.
- `PinchProblem` remains the single-case compute wrapper, while GUI-grade
  orchestration belongs in `PinchWorkspace`.
