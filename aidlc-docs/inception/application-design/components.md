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
