# HPR correction dependencies and data flow

## Dependency matrix

| Consumer | Dependency | Purpose |
|---|---|---|
| Domain HPR target models | Domain HPR value records | Typed duty/residual snapshots without outward imports |
| Input and graph contracts | Domain values and existing schema primitives | Portable residual basis and complete graph transport |
| HPR analysis | Domain values, targeting utilities and existing optimization | Calculate and verify duties/residuals |
| Application residual service | Domain values, input contracts and HPR target resolver | Validate local selection and create a detached case |
| Application target dispatch | Residual analysis service | Allocate utilities without process-stream replay |
| Placement application adapter | Residual snapshots and analysis placement context | Supply one immutable basis to every candidate |
| Placement analysis | Existing contracts, domain values and numerical helpers | Allocate/evaluate/optimize on the provided basis |
| Graph presentation | Graph contracts plus application-provided selection resolution | Select cached records and render them |
| Reporting and tutorials | Public problem/target interfaces | Explain and demonstrate verified outcomes |

Domain never imports application, analysis, presentation or contract modules.
Transport may import domain values. Analysis never imports application or
presentation. Application owns PinchProblem construction and provenance
selection; no method on a domain target constructs a problem object.

## Communication pattern

All calls are synchronous within existing library owners. Numerical snapshots
cross boundaries as finite immutable records. Live cycle/backend objects stay
inside their current execution/artifact paths. No internal serialized string
selector replaces the public named workflow methods.

## Data-flow diagram (text)

1. Original PinchProblem and explicit HPR call enter application targeting.
2. HPR analysis produces verified duty accounting, residual profiles, physical
   boundary data and graph data.
3. Application stamps and publishes the target and graph snapshots.
4. Plot observation selects and renders the matching cached graph; it stops
   there and does not lead back to analysis.
5. Explicit target.residual_utility conversion validates the selected target and creates
   a new PinchProblem input with residual_basis and utility definitions.
6. Residual utility targeting reads that basis and publishes utility results.
7. Residual utility placement supplies the same basis to every candidate,
   returns an optimized residual case, and preserves the original source.

This textual diagram is the canonical accessible representation; no graphics
renderer is required to understand the flow.

## Shared contract checkpoints

Unit 1 establishes domain load/residual values and physical correctness. Unit 2
uses target identity and verified data to repair plotting. Unit 3 adds the
canonical residual input and both dispatch/placement branches against those
values. Unit 4 integrates stable public operations into the notebook and docs.

Creation of a residual case and graph selection share a local-target resolver,
but do not share mutable caches or invoke one another. Storage/transport of
snapshots must retain tuples, finite values, units, exact profile grids and
source identity. New input semantics require validation and installed-package
checks in addition to owner-level numerical tests.
