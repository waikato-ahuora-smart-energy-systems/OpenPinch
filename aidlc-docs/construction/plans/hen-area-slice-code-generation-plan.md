# HEN Area-Slice Code Generation Plan

## Unit Context

- **Unit**: Heat Exchanger Network domain-model cleanup.
- **Approved source plan**: `aidlc-docs/inception/plans/heat-exchanger-area-slice-refinement-plan.md`.
- **Dependencies**: Existing `HeatExchanger` result model, piecewise HEN area kernel, extraction and verification services, package API barrels, and segmented-HEN tests.
- **Public contract**: `HeatExchanger.segment_area_contributions` and its serialized nested field structure remain stable; the nested element class becomes internal.
- **PBT scope**: PBT-02, PBT-03, PBT-07, PBT-08, and PBT-09 are blocking under Partial enforcement.

## Approved Generation Steps

- [x] Step 1: Review the approved refinement plan, current domain model, service consumers, documentation, tests, and package exports.
- [x] Step 2: Confirm the change stays within existing component boundaries and requires no new infrastructure or public service.
- [x] Step 3: Rename the frozen value model to `HeatExchangerAreaSlice`, remove barrel exports, and add parent aggregation and area-consistency behavior.
- [x] Step 4: Update the piecewise kernel, extraction, and verification services to use the internal slice type and parent aggregation behavior.
- [x] Step 5: Update example-based and property-based tests, including domain generators, serialization round trips, multiperiod aggregation, mismatch rejection, and unchanged topology counts.
- [x] Step 6: Update API and domain documentation without exposing the internal slice class as a package-root type.
- [x] Step 7: Run focused tests and static checks; correct any failures.
- [x] Step 8: Run the full proportional acceptance suite and package/documentation checks.
- [x] Step 9: Update implementation summaries, build/test results, AI-DLC state, audit, and both plan-level checklists.

## PBT Compliance Plan

- **PBT-02**: Generate valid exchangers with area slices and verify JSON serialization/deserialization round trips.
- **PBT-03**: Verify period duty/area totals, design-area maxima, ordering, and exchanger-count invariants over generated slices.
- **PBT-07**: Add a reusable domain strategy for valid `HeatExchanger` objects with internally consistent slices.
- **PBT-08**: Retain Hypothesis shrinking and CI seed reporting; do not suppress or retry failures.
- **PBT-09**: Continue using the existing Hypothesis dependency and pytest integration.

## Approval

- **Status**: Approved.
- **User response**: `Approved plus update docs.`
- **Approval timestamp**: 2026-07-15T17:44:14Z.
