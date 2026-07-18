# Package Usability Refactor User Stories

## US-1: First solve

As a new process engineer, I want to load a packaged case and run an explicit
complete heat-integration workflow with a few discoverable calls so that I can
obtain direct and Total Site results across the zone hierarchy without
learning internal modules or manually iterating through zones.

### Acceptance Criteria

- The workflow imports `PinchProblem` from `OpenPinch`.
- `problem.target.all_heat_integration()` is the named bulk operation and
  `problem.target()` is removed.
- The bulk operation performs one deterministic, dependency-aware zone-tree
  traversal rather than invoking focused workflows repeatedly.
- It runs only direct and indirect/Total Site heat integration; advanced
  analyses remain explicit calls.
- Zone structure determines applicability; targeting-enable flags do not exist.
- Summary and graph calls do not trigger hidden additional targeting.

## US-2: Total Site workflow

As a process engineer, I want an explicit Total Site operation so that I do not
have to infer whether a generic target call generated indirect results.

### Acceptance Criteria

- The operation generates Direct, Total Process, and Total Site rows.
- The tutorial uses the same names as the API and output tables.
- `problem.target.indirect_heat_integration()` and
  `problem.target.total_site_heat_integration()` are deliberate peers over one
  focused implementation.
- Ambiguous `direct()`, `indirect()`, `all()`, `area_cost()`, and
  `configured_analyses()` shorthands are absent from the public accessor.

## US-3: Scenario comparison

As a process engineer, I want to create named scenarios and apply the same
discoverable workflow to a selected case set so that sensitivity studies do
not require copying, mutating, synchronizing, or supplying a workflow string.

### Acceptance Criteria

- `workspace.scenario(...)` covers common option and `dt_cont` changes.
- `scenario(...)` returns a `PinchProblem`; execution remains an explicit
  `target.*` or `design.*` call.
- `workspace.cases(names).target.*` applies one discoverable workflow to
  multiple named cases and preserves deterministic ordering.
- Multiple named cases can be compared deterministically.

## US-4: Advanced thermal workflows

As a process engineer, I want friendly heat-pump, refrigeration, MVR, and
multiperiod arguments, including support for multi-segment streams, so that I
do not need internal enums, flat option keys, private scalar resolvers, or
single-segment approximations.

### Acceptance Criteria

- Algorithm families use specialized callables rather than cycle, placement,
  or workflow strings.
- Carnot, vapour-compression, Brayton, and MVR heat-pump workflows expose only
  arguments meaningful to that model; unsupported boolean combinations cannot
  be constructed.
- Carnot screening is named explicitly through `carnot_heat_pump()` and
  `carnot_refrigeration()` rather than generic HPR method names.
- Utility placement and cascade topology use booleans because they are genuine
  independent binary decisions on the methods that expose them.
- Exactly one of load fraction, load duty, or period-load mapping is effective.
- Analysis selection occurs through `PinchProblem.target.*`, not
  `TARGETING_*_ENABLED` configuration options.
- Selected-period results expose ordinary scalar fields.
- Weighted summaries work on real shared-design HPR output.
- Dedicated tutorials demonstrate multiperiod heat integration, heat pumps,
  and cogeneration with ordered results for at least two periods.
- A dedicated tutorial validates and analyses multi-segment streams while
  preserving their piecewise representation.

## US-5: HEN design selection and visualization

As a process engineer, I want to inspect top-ranked networks and render a
selected grid through the problem design surface so that I do not import result
selection or presentation internals.

### Acceptance Criteria

- Top-`n`, network-by-rank, and grid-by-rank operations are public.
- Rank selection does not mutate a transport schema unexpectedly.
- Advanced tutorials cover every supported HEN design method and distinguish
  their engineering purpose, inputs, outputs, and optional solver needs.
- Multiperiod synthesis uses an explicit period selection, records period
  weights, and demonstrates a shared network design across ordered periods.

## US-8: Predictable `PinchProblem` interaction

As a process engineer, I want every `problem.*` operation to follow the same
execution, configuration, and state rules so that reading a result cannot
unexpectedly run an analysis and one-off method arguments cannot silently
rewrite my case.

### Acceptance Criteria

- Explicit named kwargs override advanced runtime options, stored configuration,
  and defaults in a documented order.
- Omitted kwargs use stored configuration; explicit falsey and `None` values are
  preserved where valid.
- Invocation overrides are ephemeral and resolved values are recorded for
  reproducibility.
- Only target, design, and component methods perform their documented work;
  reporting, plotting, comparison, inspection, and export methods never choose
  an analysis.
- Loading, persistent configuration changes, and component changes invalidate
  dependent cached state predictably.
- Period targeting mirrors named methods under `target.all_periods` rather than
  accepting a method string or configured default target.
- No normal workflow argument requires an OpenPinch-owned closed string value;
  strings remain only for user identities and external resource names.
- A checked tutorial coverage manifest maps every supported `PinchProblem`,
  `PinchWorkspace`, target, component, design, selected-network, and plot
  operation to at least one process-engineer notebook.
- Read the Docs publishes that same verified map with links to each tutorial and
  the relevant workflow API pages.
