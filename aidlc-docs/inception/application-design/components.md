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
