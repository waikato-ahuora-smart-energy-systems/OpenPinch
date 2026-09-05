# Services

- Input preparation validates schemas, constructs segments in input order, then adds one parent to its zone.
- Direct and indirect targeting expand segments only inside thermodynamic kernels.
- Capital and area targeting sum segment contributions and deduplicate parent counts.
- HPR and MVR unit models call one shared profile-to-parent builder.
- HEN preparation emits parent axes plus segment tensors; model equations use cumulative parent heat coordinates.
- HEN extraction emits one parent exchanger with nested area contributions.
- Network diagrams and controllability consume only parent topology.

## Package Usability Refactor Service Orchestration

- Public accessors translate descriptive method calls into existing numerical
  service functions; numerical services remain internal and do not become the
  tutorial boundary.
- The effective-argument resolver maps named engineering values onto internal
  option keys, validates method-specific combinations, and attaches provenance
  without changing stored configuration.
- `all_heat_integration()` performs one post-order, dependency-aware zone-tree
  traversal rather than chaining focused public methods.
- `target.all_periods.*` prepares independent period execution contexts,
  dispatches only the selected mirrored method, and commits ordered results
  after successful validation.
- HPR method names map to fixed backend families. Utility placement and
  cascade/parallel topology choose valid branches within a family; invalid
  cross-family arguments fail before service invocation.
- Cogeneration, exergy, and energy-transfer services consume a compatible
  returned base target or establish one documented default prerequisite.
- HEN design methods own fixed prerequisites, delegate to synthesis services,
  and wrap serializable results in application-owned selection views.
- Workspace batches materialize named `PinchProblem` cases, invoke the mirrored
  accessor in insertion order, and collect results or structured case errors.
- Reporting and presentation services receive cached state only and are never
  allowed to invoke targeting or design.
- Tutorial verification generates a live public inventory, compares it with the
  CSV manifest, executes notebooks by dependency profile, and supplies the same
  manifest to the RTD coverage page.

## Repository Issue Remediation Orchestration

- Workspace construction, `load`, `scenario`, internal case creation, and bundle
  validation all invoke the same case-identifier contract before changing state.
- Batch export resolves the destination root once, validates each case
  directory independently, and preserves the established per-case error
  isolation contract.
- Problem input observation copies authoritative input at the property boundary;
  targeting and serialization continue to use internal validated state.
- Multiplier updates acquire the prepared root through the existing guard before
  changing zone state and invalidating cached results.
- Reporting reserves a unique workbook path before opening pandas/openpyxl and
  removes the reservation if the write fails.
- The OpenHENS comparison wraps prerequisite checking and source execution in
  one exact-checkout import context and injects the verified factory into every
  case run.
- Documentation verification scans only active current-state and reverse-
  engineering sources so historical audit evidence remains valid.

## Utility Placement Optimisation Service Orchestration

### Problem workflow

1. The target accessor normalizes explicit public keywords into a validated,
   immutable `UtilityPlacementRequest` before any targeting or optimisation.
2. The application layer resolves the requested zone, compatible direct or
   Total Site scope, canonical ordered periods, and period weights.
3. A deep-copied execution zone is passed to the context builder. Existing
   target services populate only that copy; the builder extracts immutable
   shifted profiles, residual demands, ambient conditions, units, and target
   identity.
4. The analysis service normalizes or generates fixed template identities,
   derives the intersection of feasible period bounds, creates deterministic
   initial candidates, and rejects an empty feasible region before backend
   execution.
5. The optimisation coordinator creates a solver-neutral
   `OptimisationProblem`. Each objective call decodes one point and delegates
   to the candidate engine.
6. The candidate engine replays that placement against every period, allocates
   all residual hot and cold demand through existing targeting calculations,
   and returns structured feasibility plus objective decomposition.
7. Candidate evaluation calls the pure balanced-composite entropy kernel and
   excludes positive-duty generated default utilities.
8. The coordinator preserves only feasible evaluations, orders them
   deterministically, enforces the requested alternative limit, and raises a
   typed exhaustion error if none remain.
9. The service assembles detached evidence containing request metadata, best
   candidate, alternatives, period and aggregate decompositions, coverage,
   units, termination data, and non-fatal diagnostics.
10. The application adapter converts the best candidate to canonical utilities
    on a new unsolved normal case and stores the detached evidence on that
    returned case. Existing source heat-target results and input state remain
    unchanged.

### All-period behavior

The primary method defaults to all canonical periods when a period axis exists.
`problem.target.all_periods.utility_placement(...)` explicitly requests the
same one-vector/all-period solve. It bypasses the current generic all-period
loop because independent per-period minimisations would violate the shared-
placement requirement. A caller may request an ordered period subset through
the primary method; weights are projected without reordering.

### Workspace batch behavior

Workspace case batches invoke the problem workflow once per requested case in
the established order. `CaseBatchResult` retains successful detached results
and typed exceptions independently. The batch accessor never changes the
active case and does not suppress successes after another case fails.

### Targeting boundary

The context builder resolves one owned hierarchy zone and composes existing
direct, Total Site, or aggregate indirect services rather than reimplementing
profile construction. Candidate allocation uses
detached target inputs or a thin analysis-owned allocation adapter. No
placement component imports the application layer, reaches into an optimiser
backend, or mutates canonical `Zone.targets`.

### Failure boundaries

- Public input and template errors fail before targeting.
- Scope, period, and profile errors fail during detached context preparation.
- Empty intersections of physical and caller bounds fail before optimisation.
- Ordinary candidate bound, ordering, separation, or coverage failures become
  deterministic infeasible evaluations and cannot outrank feasible candidates.
- Non-finite objective values and backend exhaustion retain
  method, seed, objective, scope, counts, period, and coverage diagnostics.
- If every candidate is infeasible, the service raises; it never returns a
  least-infeasible success.

### Normal-case return and workspace registration

The target call returns a normal detached `PinchProblem` containing the best
utilities. Complete evidence remains at `utility_placement_result`, while
ordinary targeting, summary, report, and plotting remain explicit normal case
operations. `workspace.add(case, name=..., activate=False)` registers the case,
preserves placement evidence, and does not copy unrelated analysis caches.

### Executable notebook delivery

After the public application workflow is stable, the existing notebook
generator emits `19_utility_placement_optimisation.ipynb`. It calls the concise
thermodynamic API once, registers the returned case through `workspace.add`,
and uses ordinary summary, target, GCC, and Total Site Profile operations. The
canonical tutorial manifest selects its dependency profile, CI executes the
generated artifact, and distribution tests verify its inclusion. The notebook
does not manually construct utilities or invoke a utility-placement CLI.

## TESPy HPR Performance-Map Service Orchestration

### Default CoolProp targeting

1. The current vapour-compression accessor normalizes
   `simulation_backend="coolprop"` before calling `_hpr`.
2. `_hpr` carries the value as transient runtime intent and records it for
   period replay; it does not overload `HPR_TYPE` or the black-box minimizer.
3. The HPR service selects the existing CoolProp simulator and otherwise follows
   the current placement, targeting, aggregation, and result-normalization path.
4. The returned target records `hpr_simulation_backend="coolprop"`. Existing
   calls, return types, and numerical baselines remain unchanged.

### Explicit TESPy targeting

1. The same current method accepts `simulation_backend="tespy"`.
2. The HPR service verifies that the selected vapour-compression configuration is
   supported before expensive targeting begins.
3. The TESPy adapter constructs its documented network lazily, solves the design
   condition, evaluates requested target conditions, and returns the same
   normalized internal result shape as the CoolProp adapter.
4. Import or convergence failures identify TESPy, the selected cycle, operating
   condition, and installation extra. No automatic CoolProp fallback occurs.
5. The ordinary target remains a normal `HeatPumpTargetBase` subtype with the
   selected backend recorded as plain provenance.

### Follow-up map generation

1. The caller passes a successful HPR target and `HprPerformanceMapRequest` to
   `problem.target.hpr_performance_map(...)`.
2. The application layer checks that the target belongs to a supported
   heat-pump/refrigeration result family, then delegates without equations.
3. The context builder validates the target backend, mode, single-port topology,
   reference-capacity basis, grid coordinates, and target configuration.
4. Grid traversal is deterministic: source temperature, then sink temperature,
   then ascending load fraction. Curve and point identifiers derive from these
   normalized coordinates rather than input object identity.
5. The selected simulator prepares once. TESPy uses the declared design point
   derived from the target's nominal condition, then solves offdesign points;
   CoolProp evaluates the equivalent existing cycle equations.
6. Each result is normalized to nonnegative `q_source`, `q_sink`, and total
   external electric power. Heat-pump COP is `q_sink / electric_power`;
   refrigeration COP is `q_source / electric_power`.
7. Any missing, failed, non-finite, or physically inconsistent point aborts the
   complete map. Diagnostics identify the coordinate and backend but are not
   embedded as successful map points.
8. The service validates the completed map and returns a detached Pydantic
   contract. Serialization is caller-controlled and performs no simulation.

### Fixed-capacity and load semantics

The map is absolute for one declared `reference_capacity`. At every active point,
useful duty equals `load_fraction * reference_capacity` within tolerance. The
first release does not scale capacity. OpenUtility must either use a candidate
whose capacity equals the map reference capacity or apply a documented constant
scale to every duty and power value; variable sizing remains downstream work.

### Temperature semantics

Request coordinates are external thermal-service temperatures in `degC`. The
simulator applies target approach-temperature assumptions to obtain internal
refrigerant conditions. Internal evaporation, condensation, superheat, subcool,
and characteristic values are structured provenance and never replace the
external coordinates used by OpenUtility thermal nodes.

### Multi-period behavior

The map grid contains the temperature combinations needed by downstream periods,
but it is not itself a period-dispatch result. A scalar successful target can
supply the default reference capacity. A multi-period target with an array or
aggregate useful duty requires the request to state one explicit fixed reference
capacity. Period weights, tariffs, node demands, candidate selection, and dispatch
remain exclusively in OpenUtility.

### Failure and dependency boundaries

- Invalid public selector or request values fail before simulation.
- Unsupported target families or multi-port configurations fail with a typed
  compatibility error rather than being flattened into one node pair.
- TESPy import occurs only when selected and uses the existing optional-dependency
  error style.
- A simulator failure closes its session and rejects the map; partial maps are not
  returned in schema `1.0`.
- No service imports OpenUtility, Pyomo, HiGHS, application objects, or presentation
  modules.
- Contract-only imports remain successful when TESPy is unavailable or blocked.

### Cross-package contract promotion

OpenPinch publishes canonical JSON Schema and golden heat-pump/refrigeration
fixtures. OpenUtility validates its plain mapping decoder against the same field
semantics and rejects unknown versions. Schema `1.0` is promoted only after the
consumer enforces capacity consistency, adjacent-segment interpolation, strict
COP/version/unit validation, mode-aware useful duty, and a separate
temperature-matching tolerance.
