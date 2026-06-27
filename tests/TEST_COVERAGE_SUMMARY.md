# OpenPinch Test Coverage Summary

- Full verification command:
  - `/opt/homebrew/bin/uv run coverage run --source=OpenPinch -m pytest -q`
  - `/opt/homebrew/bin/uv run coverage report -m`
- Latest full-suite result recorded during this work:
  - `1799 passed, 1 skipped`
  - Line coverage: `20,287 / 20,482` statements covered
  - Missed statements: `195`
  - Effective line coverage: approximately `99.05%`

## Coverage Expansion Summary

- Heat exchanger network controllability:
  - Added validation tests for endpoint, actuator, pairing, and result models.
  - Covered invalid identities, invalid counts, invalid ranks, invalid singular
    values, non-finite condition numbers, and score bounds.
  - Added service option validation for invalid approach temperatures, rank
    tolerances, condition thresholds, redundancy values, and interaction values.
  - Covered zero-duty outputs and zero-duty exchanger filtering.
  - Covered duplicate actuator identifier suffixing.
  - Covered matrix diagnostic edge cases for zero matrices and over-large rank
    tolerances.
  - Covered redundancy and thermal-margin helper edges.
  - Covered diagnostics for unusable singular directions.

- Plot accessor:
  - Added deterministic in-memory graph payload tests without adding more
    monkeypatching.
  - Covered all named graph shortcut methods, including shifted, balanced, real
    GCC, GCC with heat pump, net-load, total-site, and SUGCC helpers.
  - Covered selection errors for unmatched graph type, out-of-range graph index,
    and unknown zone name.
  - Covered descriptor class access.
  - Covered qualified-zone matching, suffix matching, ignored empty candidates,
    slug normalization, fallback `graph` slugs, and empty gallery HTML.

- Heat pump and refrigeration entrypoint:
  - Covered direct heat pump orchestration with a deterministic optimizer result.
  - Covered direct and indirect zero-load early exits.
  - Covered heat-pump, refrigeration, duty, fraction, period-value, missing-period,
    missing-config, unsupported-mode, and all-NaN load-selection paths.
  - Covered unknown HPR backend rejection.
  - Covered HPR target-summary field mapping.
  - Covered direct heat-pump graphs, indirect SUGCC graphs, and refrigeration graph
    suppression.
  - Covered `_calc_hpr_cascade` fallback behavior for legacy helper signatures
    that do not accept `period_idx`.
  - Covered refrigeration cascade updates for net, hot, and cold refrigeration
    columns.

- Network grid diagram renderer:
  - Added Plotly adapter tests for text annotations, tick storage, marker sizing,
    and hover-template updates.
  - Covered utility-aware stream unit x-position helpers for hot and cold
    utilities.
  - Covered recovery-match x-pair fallback for unscaled diagrams.
  - Covered split-branch early return.
  - Covered duplicate label stacking without explicit offsets.
  - Covered cold-stream rendering when a stream has no recovery branches.
  - Covered optional stage-boundary rendering.
  - Covered temperature-scaling fallbacks for missing temperatures and degenerate
    temperature ranges.
  - Covered empty-model temperature-scale defaults, missing midpoint
    temperatures, invalid stream-line width, and split-adjusted temperature
    positions inside an existing split group.

- HEN synthesis execution settings and task builders:
  - Covered solver lookup fallbacks and solver-option copies for PDM, TDM, EVM,
    and unknown methods.
  - Covered standard and expanded quality-tier fractions.
  - Covered standard-tier detection and empty generated PDM approach temperatures.
  - Covered default and explicit TDM parent-limit behavior.
  - Covered workflow-settings failure when no master zone is loaded.
  - Covered stage-count extraction from seeded recovery exchangers.
  - Covered stage-count contract errors for seed networks without staged recovery
    exchangers.
  - Covered seeded approach-temperature fallback order:
    summary metrics, solver metadata, exchanger approach temperatures, then
    workflow settings.
  - Covered `_required_stage_count` contract errors for successful outcomes with
    no stage metadata.

- Input canonicalization:
  - Covered invalid zone config option types.
  - Covered blank zone-type defaults by depth.
  - Covered invalid `dt_cont_multiplier` helper inputs.
  - Covered missing-child multiplier inheritance fallback.
  - Covered zone rewrite behavior for `None`, blank, and root-level stream zones.
  - Covered root-name collisions when creating process zones from site-level
    streams.
  - Covered generated zone trees with a non-string top-zone name.
  - Removed an unreachable operation-name collision loop from generated zone-tree
    construction.

- GCC manipulation and graph data:
  - Covered breakpoint insertion, interval manipulation, monotonic transforms,
    empty and repeated point handling, and target graph construction edge cases.
  - Removed unreachable branch logic where later code made the condition
    impossible.

- Heat exchanger network synthesis solver helpers:
  - Covered solver extraction, array preparation, decomposition helpers, backend
    error handling, and pinch-design model helper edges.
  - Covered invalid solver statuses, missing dependency handling, missing or
    malformed solver arrays, duplicate topology cases, and seeded-task behavior.

- Heat pump and refrigeration unit models:
  - Covered simple, cascade, and parallel vapour-compression normalization helpers.
  - Covered scalar, array, `None`, `NaN`, mismatched-shape, and non-numeric duty
    inputs.
  - Covered refrigeration duty allocation and heat-pump duty allocation.
  - Covered single-phase pressure-target solving guard rails and failure paths.
  - Covered invalid temperature inputs and secondary-duty conversion errors.

- Multi-stage steam turbine:
  - Covered stage-efficiency and enthalpy guard helpers.
  - Covered stage-work prediction branches.
  - Covered scalar, 2D, empty, non-finite, above-pinch, and below-pinch input
    normalization.
  - Covered solved-state accessors and zero-flow or zero-condensation segment
    cases.

- Public API, schema, and fixture-backed tests:
  - Added static JSON-backed fixtures for config, config metadata, graph data,
    HPR schemas, synthesis schemas, target/reporting schemas, unit systems,
    value/stream edge cases, value resolution, indirect integration, process MVR,
    problem validation, and stagewise helper cases.
  - Added tests that load those fixtures directly instead of constructing many
    ad-hoc monkeypatched inputs.
  - Added public entrypoint tests for services and package exports.
  - Added tests for resources, CLI helpers, report units, CoolProp fluid lookup,
    and config enum normalization.

## Edge-Case Themes Covered

- Empty input collections and empty networks.
- Zero, near-zero, and negative duty values.
- Duplicate identifiers and deterministic suffixing.
- Missing optional metadata with fallback defaults.
- Non-finite numeric inputs, including `NaN` and infinities.
- Invalid enum/configuration values.
- Period-specific data selection and missing period keys.
- Serialization and round-trip model validation.
- Solver dependency, solver status, and solver-output contract failures.
- Zone-tree ambiguity, blank labels, root-level stream assignment, and multiplier
  inheritance.
- Graph selector aliasing, unknown selector errors, and export/gallery edges.

## Monkeypatch Reduction Notes

- New tests prefer static JSON fixtures and direct helper calls where practical.
- Monkeypatching remains in orchestration tests where the goal is to verify that
  service wrappers pass period IDs, selected loads, or solver options to expensive
  downstream optimizers.
- Existing monkeypatch-heavy tests were not removed wholesale because many isolate
  expensive thermodynamic or solver boundaries; replacing all of them would require
  a broader fixture redesign.

## Future File-Structure Improvement Notes

- Split large mixed-purpose test modules into smaller files by behavior boundary:
  model validation, orchestration, solver contracts, graph rendering, and reporting.
- Move reusable test builders into package-local fixture modules instead of
  repeating long `SimpleNamespace` payloads across tests.
- Keep static JSON fixtures grouped by domain under `tests/fixtures/`, with one
  fixture file per stable schema or service boundary.
- Consider a `tests/fixtures/hens/` subfolder for HEN synthesis-specific topology,
  stage-packing, and solver-array fixtures.
- Consider a `tests/fixtures/hpr/` subfolder for HPR backend, layout, encoding, and
  stream-profile fixtures.
- Separate slow integration tests from pure helper tests with pytest markers so
  local TDD loops can run smaller suites without losing full coverage in CI.
- Prefer small pure helper functions in production modules for branch-heavy logic;
  they are easier to test directly than orchestration paths that require solver
  state or thermodynamic backends.
- Keep public API tests close to their package entrypoints and keep private helper
  tests close to the implementation domain they document.
