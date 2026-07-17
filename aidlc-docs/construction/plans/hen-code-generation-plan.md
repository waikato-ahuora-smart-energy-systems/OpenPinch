# HEN Code Generation Plan

This is the single source of truth for the Heat Exchanger Network unit. The user approved this sequence in the implementation request.

## Context

- Requirements: FR-13 through FR-15; NFR-05 through NFR-07.
- Dependencies: completed Domain/Input and Targeting/Integration units.
- Persistence: private solver snapshots and public network result schemas; no database entities.

## Steps

- [x] Step 1: Add segment-profile tensors and metadata to prepared solver arrays while retaining parent axes.
- [x] Step 2: Add parent cumulative heat-coordinate and ordered piecewise-temperature helpers.
- [x] Step 3: Integrate helpers into stagewise, PDM, TDM, EVM, utility, and pinch-decomposition equations.
- [x] Step 4: Retain the Chen area surrogate in topology optimization and add
  ordered duty-aligned segment area slices to exact post-processing.
- [x] Step 5: Add public area-contribution records, extraction, verification, diagrams, and controllability compatibility.
- [x] Step 6: Add solver, multiperiod, extraction, verification, and performance regression tests plus implementation summary.

## Post-Implementation Correction: Segmented PDM dTmin Propagation

The user's correction request explicitly authorizes this focused continuation of
the approved HEN generation plan.

- [x] Step 7: Trace the PDM copied-zone `dt_cont` convention through
  `segment_numeric_view` and confirm the stale-child failure mode.
- [x] Step 8: Apply the HEN `dTmin / 2` minimum to every explicit segment before
  direct-integration targeting, retaining the existing parent-only path for flat
  streams and preserving per-period maxima.
- [x] Step 9: Add an example regression proving copied parents and every child
  receive the contribution used by the expanded numeric view.
- [x] Step 10: Add a domain-generated invariant test covering arbitrary ordered
  segmented streams and finite non-negative HEN approaches.
- [x] Step 11: Run focused PDM, segmented-stream, lint, formatting, and regression
  checks; record the correction summary and extension compliance.
