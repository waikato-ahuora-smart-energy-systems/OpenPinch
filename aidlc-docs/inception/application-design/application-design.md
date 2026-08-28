# Application Design Summary

The package-wide architecture modernization supersedes the earlier
stream-focused package-placement detail where ownership differs. Its current
design record is
`package-architecture-modernization-design.md` in this directory.

The design introduces an ordered child profile within the existing `Stream` aggregate. Collection and zone boundaries remain parent based. A shared segment projection prevents each numerical service from inventing its own flattening rules. HEN synthesis receives parent axes and segment tensors, preserving topology while replacing constant-CP equations with cumulative heat-coordinate relations. Reporting remains parent first and nests segment detail explicitly.

The approved package-usability refactor is documented separately in
`package-usability-refactor-design.md`. It preserves the two-class root facade,
adds descriptive target/design/workspace accessors, separates execution from
observation, and makes the tutorial/RTD manifest an enforced public-contract
consumer without overwriting the segmented-stream design above.

## Repository Issue Remediation Design

The remediation retains all existing public component boundaries. Workspace
case identifiers gain one shared strict validator used by runtime and bundle
entry points, with a second containment check at batch export. `problem_data`
becomes a detached observation boundary; explicit application methods remain the
only supported mutation paths. Reporting uses exclusive workbook reservation,
and the comparison tool uses one scoped import context that verifies every
OpenHENS module against the requested checkout before injecting its factory into
execution. Current documentation is corrected through a scoped drift guard that
does not rewrite historical records. No new runtime dependency or root export is
introduced.

## Utility Placement Optimisation Design

### Design outcome

Utility placement is an additive specialist analysis under
`OpenPinch.analysis.utility_placement`, reached through
`problem.target.utility_placement(...)`. The application layer owns unique
hierarchy-zone resolution, zone-type-derived scope, existing-utility template
inference, and period resolution; analysis owns templates, bounds, vector
conversion, candidate replay, coverage, objectives, optimisation coordination,
and result assembly. Existing direct, Total Site, aggregate indirect targeting,
and solver-neutral
optimisation are composed at explicit adapter boundaries. Contracts live in
`OpenPinch.contracts.utility_placement` and the
two-symbol package-root facade remains unchanged.

### Public contract

The primary method accepts optional concise `isothermal` and `sensible` counts,
an optional zone, period selection, and typed options. Existing utilities supply
typed templates when counts are omitted; supplied counts generate paired hot
and cold templates. Zone type selects the target profile. The method immediately
constructs the immutable internal request. A problem with canonical
periods uses one shared placement across those periods, and the all-period
accessor makes that behavior explicit rather than launching independent solves.

The public return is a detached normal `PinchProblem` whose utility input is
the best feasible set. Detailed JSON-serializable placement evidence remains at
`optimized_case.utility_placement_result`. `workspace.add(case, name=...,
activate=False)` explicitly registers the case while preserving that evidence.

### Internal component model

1. The application context builder runs existing targeting only against an
   isolated execution-zone copy and extracts immutable period profiles.
2. The pure template model validates counts and identities, derives feasible
   bound intersections, generates starts, and encodes or decodes placements.
3. The optimisation coordinator delegates bounded search to the existing
   solver-neutral service.
4. The candidate engine replays one decoded placement against every period,
   verifies full heating and cooling coverage, and calls one objective evaluator.
5. The thermodynamic evaluator integrates entropy stably in absolute
   temperature and reports ambient-temperature exergy destruction.
6. The application adapter converts only the best candidate's shared
   temperatures into canonical utilities on a new unsolved normal case and
   attaches the complete detached evidence.
7. The existing tutorial generator produces one thermodynamic-only notebook
   using the concise return/add/target/plot workflow; manifest, execution, and
   package-data gates own its delivery evidence.

### Error and state model

Placement-specific exceptions distinguish invalid requests, incompatible
scope, context preparation, empty feasible bounds, targeting/allocation,
non-finite thermodynamics, and optimiser exhaustion.
Ordinary candidate infeasibility is structured data until solve completion.
Feasible candidates always outrank penalties, and an all-infeasible run raises
instead of returning a least-infeasible placement. The source problem,
utilities, configuration, heat targets, workspace selection, and cached study
inputs are unchanged on success and failure.

### Alternatives considered

| Alternative | Decision | Reason |
|---|---|---|
| Put placement equations in `_TargetAccessor` | Rejected | Violates the existing orchestration-only application boundary and makes pure testing harder. |
| Mutate source utility streams | Rejected | The optimized utilities belong to a detached normal case; the source remains unchanged. |
| Solve every period independently through the generic all-period loop | Rejected | Violates the one-placement/all-period feasibility requirement. |
| Add placement logic directly to the general optimiser package | Rejected | Utility thermodynamics are domain-specific; the optimiser should remain solver-neutral. |
| Implement a new optimisation backend | Rejected | Existing bounded, seeded backends already satisfy the architectural need and avoid a new dependency. |
| Reimplement direct or Total Site physics | Rejected | Existing target services remain canonical and are composed through detached context. |
| Create a new root export | Rejected | Specialist imports plus the existing target accessor preserve the stable two-symbol root facade. |

### Requirements and story traceability

| Design area | Requirements | Stories |
|---|---|---|
| Public accessors, scopes, periods, batches | FR-001, FR-007, FR-008, FR-016 | UPO-02, UPO-06, UPO-10 |
| Request and template contracts | FR-002 through FR-006 | UPO-01, UPO-08, UPO-09 |
| Candidate coverage and constraint model | FR-005, FR-006, FR-008, FR-012 | UPO-03, UPO-06, UPO-08, UPO-11 |
| Thermodynamic evaluator | FR-009 | UPO-04, UPO-11 |
| Monetary capability exclusion | FR-010 | UPO-05 |
| Optimisation coordination and alternatives | FR-011, FR-012, FR-014 | UPO-07, UPO-08, UPO-11 |
| Detached optimized case, evidence, and workspace registration | FR-013 through FR-015 | UPO-07, UPO-09 |
| Executable notebook delivery | FR-017 | UPO-02, UPO-12 |
| Numerical, compatibility, maintainability, diagnostics | NFR-001 through NFR-006 | UPO-08, UPO-09, UPO-11, UPO-12 |
| No new infrastructure boundary | NFR-007 | UPO-12 |

All FR-001 through FR-017, NFR-001 through NFR-007, and UPO-01 through UPO-12
have an owning component, interface, and orchestration path. Detailed equations,
tolerance values, penalties, and property definitions remain intentionally
deferred to per-unit Functional Design.

Validation set: FR-001, FR-002, FR-003, FR-004, FR-005, FR-006, FR-007,
FR-008, FR-009, FR-010, FR-011, FR-012, FR-013, FR-014, FR-015, FR-016,
FR-017;
NFR-001, NFR-002, NFR-003, NFR-004, NFR-005, NFR-006, NFR-007; UPO-01,
UPO-02, UPO-03, UPO-04, UPO-05, UPO-06, UPO-07, UPO-08, UPO-09, UPO-10,
UPO-11, and UPO-12.

### Extension compliance at Application Design

- **PBT-01 through PBT-10**: N/A for blocking enforcement at Application
  Design. The enabled extension's applicability matrix begins formal property
  identification at Functional Design. This design nevertheless isolates pure
  round-trip, invariant, oracle, reproducibility, and non-mutation boundaries
  so those obligations can be assigned without architectural rework.
- **Security Baseline**: skipped because it is disabled for this feature.
- **Resiliency Baseline**: skipped because it is disabled for this feature.
