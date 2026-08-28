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
