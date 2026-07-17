# Domain and Input Code Generation Plan

This is the single source of truth for the Domain and Input unit. The user approved this sequence in the implementation request.

## Context

- Requirements: FR-01 through FR-10, FR-16, FR-17; NFR-01 through NFR-04 and NFR-07.
- Dependencies: existing `Value`, `Stream`, `StreamCollection`, schemas, validation, and preparation.
- Persistence: file/workspace serialization only; no database entities.

## Steps

- [x] Step 1: Add segment and profile schemas with conditional parent validation.
- [x] Step 2: Add `StreamSegment` and segmented parent construction, aggregation, continuity, atomic mutation, and period propagation.
- [x] Step 3: Add parent and expanded collection projections with recursive cache revisions and expanded reporting.
- [x] Step 4: Normalize nested schemas and calculated profiles during input preparation.
- [x] Step 5: Add focused domain, schema, persistence, and property-based tests.
- [x] Step 6: Add public exports and unit implementation summary.
