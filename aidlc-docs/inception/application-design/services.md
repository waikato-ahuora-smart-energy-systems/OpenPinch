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
