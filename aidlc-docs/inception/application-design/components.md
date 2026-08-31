# Components

## Segmented Stream Domain

Owns `StreamSegment`, parent aggregation, ordered continuity, atomic mutation, period propagation, and revision tracking.

## Structured Input Normalizer

Converts nested segment or profile schemas into one prepared parent stream without grouping flat records.

## Segment Numeric Projection

Projects parent collections into dense per-segment arrays while retaining parent identity for counting and cache invalidation.

## Thermal Service Adapters

Use segment projections for targeting and convert calculated HPR/MVR profiles into parent streams.

## HEN Segment Profile Model

Keeps topology parent based while mapping cumulative heat to temperature and calculating ordered segment-slice areas.

## Segment-Aware Reporting

Keeps parent summaries stable and exposes explicit expanded stream and exchanger-area contribution records.

## Package Usability Refactor Components

### Workflow Root Facade

Keeps `OpenPinch` limited to `PinchProblem` and `PinchWorkspace`. Construction,
loading, validation, and serialization prepare cases without analysis.

### Target Accessor Family

Owns explicit heat-integration, HPR, cogeneration, exergy, energy-transfer,
area/cost, and mirrored all-period methods. Method names select algorithms;
booleans express independent engineering choices; named values override
configuration without mutating it.

### Effective Argument Resolver

Resolves omitted values through named argument, advanced `options`, stored
configuration, and library-default precedence. It validates mutually exclusive
loads and records value provenance for reproducible result metadata.

### Component and HEN Design Accessors

Owns process-MVR mutation and model-specific HEN synthesis methods. Component
changes invalidate derived state without solving. Design methods establish only
fixed documented prerequisites and return application-owned ranked/network/grid
views around serializable schemas.

### Observation and Presentation Accessors

Owns summaries, metrics, reports, comparisons, plot catalog/data/figures,
exports, and dashboard launch. Every operation consumes existing state and
raises an actionable missing-state error instead of selecting an analysis.

### Workspace Case and Batch Views

Owns named scenarios, case selection, persistence, active-case forwarding, and
ordered batch access. Batch `target` and `design` accessors mirror
`PinchProblem` methods rather than accepting workflow strings.

### Tutorial and RTD Coverage Contract

Owns the eighteen-notebook process-engineer learning path and the canonical CSV
manifest consumed by CI and Sphinx. Notebooks are executable study templates,
while the manifest proves every supported operation and semantic mode has an
owner.

## Repository Issue Remediation Components

### Workspace Identity Contract

Owns one portable, clean-break case-identifier validator shared by workspace
runtime entry points and persisted bundle validation. It rejects unsafe names;
it never aliases or silently sanitizes them.

### Workspace Export Boundary

Owns `Path`-based per-case destination construction, resolved-root containment,
and preservation of original case identifiers in `CaseBatchResult` mappings.

### Problem Input Observation Boundary

Keeps `_problem_data` authoritative and returns a detached deep snapshot through
the public `problem_data` property. Existing mutation methods remain responsible
for validation, runtime rebuilding, and cache invalidation.

### Workbook Path Allocator

Owns exclusive `.xlsx` path reservation, readable project/timestamp naming, and
cleanup of incomplete reservations after writer failure.

### Exact OpenHENS Checkout Loader

Owns temporary interpreter import isolation, required-capability validation,
module-origin containment, verified callable delivery, and restoration of the
previous interpreter state.

### Current Contract Drift Guard

Owns scoped assertions that current-state and reverse-engineering documentation
describe only the canonical `PinchProblem` and `PinchWorkspace` root API while
leaving explicitly historical records untouched.

## Utility Placement Optimisation Components

### Utility Placement Contract Family

Owns the specialist Pydantic request, template, option, candidate, period,
aggregate-result, diagnostic, and error-context schemas in
`OpenPinch.contracts.utility_placement`. The contracts normalize public values,
preserve units and stable identities, forbid backend-private objects, and
support detached JSON round-trips. Objective, level-kind, utility-side, and
base-target enums live with this specialist contract unless an existing domain
enum is already the canonical owner.

### Target Placement Accessors

The existing problem target accessor exposes hierarchy-aware
`utility_placement(...)`; its
all-period counterpart selects every canonical period for one shared placement.
The workspace case-batch accessors mirror both surfaces and reuse
`CaseBatchResult` ordering and failure isolation. The problem accessor delegates
once, converts the best result to a detached normal case, and retains placement
evidence on that returned case. `workspace.add(...)` registers a returned case
without activating it by default or copying unrelated analysis caches.

### Placement Context Builder

Owns unique hierarchy resolution plus read-only extraction of shifted direct,
Total Site, or aggregate indirect profiles, residual
heating and cooling demands, canonical periods and weights, configuration,
ambient temperature, approach temperatures, and units. It may invoke existing
targeting services against an isolated execution-zone copy, but the resulting
`PlacementContext` contains only immutable numerical data and stable metadata.

### Utility Template Model

Owns template normalization, generated defaults, existing-utility inference,
`Both` expansion, deterministic side padding, count-to-template agreement,
unique identities, hot/cold direction, fixed near-isothermal span, sensible
span bounds, feasible bound intersection, and
deterministic starting candidates. It also owns reversible decision-vector
encoding and decoding. These are pure operations with no optimiser or
application dependency.

### Candidate Evaluation Engine

Replays one decoded placement independently against every selected period.
It obtains duty allocation from detached copies of existing direct or Total
Site targeting inputs, verifies complete hot and cold coverage, evaluates the
selected objective, and returns a typed feasible or infeasible evaluation with
diagnostics. It never ranks or caches candidates.

### Thermodynamic Objective Evaluator

Owns stable sensible and near-isothermal entropy calculations in absolute
temperature, utility-side and process-side decomposition, finite-value checks,
noise tolerance, and exergy-destruction conversion. It is a pure numerical
kernel and does not know about accessors or optimiser backends.

### Placement Optimisation Coordinator

Builds one existing solver-neutral `OptimisationProblem`, evaluates bounded
candidates, normalizes feasible alternatives, applies deterministic objective
and coordinate ordering, and converts optimiser exhaustion into placement-
specific typed diagnostics. It does not implement backend algorithms.

### Utility Placement Analysis Service

Orchestrates request normalization, context preparation, template-model
construction, optimisation, candidate re-evaluation, result assembly, and
typed failure translation. This is the specialist service entry point under
`OpenPinch.analysis.utility_placement`; all numerical behavior remains below
the application layer.

### Optimized-Case Application Adapter

Builds a normal detached `PinchProblem` from source input plus the best solved
utility temperatures. It stores detailed evidence at
`utility_placement_result`; all ordinary target, summary, report, and plot
operations remain owned by the normal case APIs.

### Utility Placement Notebook Example

The existing tutorial generator owns exactly one new artifact,
`19_utility_placement_optimisation.ipynb`. The notebook consumes only public
problem-target and workspace APIs, demonstrates thermodynamic placement, adds
the returned normal case, and contains lightweight executable assertions over
standard summaries and plots. Existing
tutorial-manifest, execution-profile, and package-data owners register and
verify it. This component adds no CLI surface.
