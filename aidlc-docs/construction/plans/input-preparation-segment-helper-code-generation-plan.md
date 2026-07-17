# Private Helper Extractions Code Generation Plan

## Unit Context

- **Units**: Domain and Input preparation cleanup plus Heat Exchanger domain-model cleanup.
- **Requested changes**:
  - Move stream-segment-specific preparation functions out of `data_preparation.py` into a private helper module.
  - Move the internal `HeatExchangerAreaSlice` implementation and its pure aggregation/design-area helpers out of `heat_exchanger.py` into a private helper module.
- **Dependencies**: `Stream`, `StreamSegment`, `StreamSchema`, `Zone`, unit normalization, profile linearisation, `HeatExchanger`, Pydantic validation, and the existing input-preparation and area-slice test suites.
- **Public contract**: No public API, schema, validation, serialization, ordering, or numerical behavior changes.
- **Story traceability**: Internal maintainability follow-up to the approved segmented variable-CP stream refactor.

## Generation Steps

- [x] Step 1: Reconfirm the segment-related function boundary and all internal callers.
- [x] Step 2: Create `OpenPinch/services/input_data_processing/_stream_segment_preparation.py` containing segmented parent construction, profile conversion, and parent aggregate validation.
- [x] Step 3: Update `data_preparation.py` to delegate segmented streams to the private helper and remove imports used only by the extracted implementation.
- [x] Step 4: Create `OpenPinch/classes/_heat_exchanger_area.py` containing the internal `HeatExchangerAreaSlice` model and pure period aggregation/design-area functions.
- [x] Step 5: Update `heat_exchanger.py` to import the internal slice type, retain its existing field and property contract, and delegate slice calculations to the private helper.
- [x] Step 6: Verify that no public exports or external call sites change and that no duplicate implementation remains.
- [x] Step 7: Run focused input-preparation, segmented-stream, area-slice, API-surface, formatting, and lint checks.
- [x] Step 8: Update the implementation summaries, AI-DLC state, audit log, and this checklist with the verified result.

## Validation Scope

- Existing flat-stream preparation remains unchanged.
- Explicit segment and temperature-heat profile preparation remain behaviorally identical.
- Ordered segment identity, multiperiod alignment, authoritative parent aggregates, errors, and units remain unchanged.
- No new public symbol is introduced.
- Existing imports of the internal slice type from `OpenPinch.classes.heat_exchanger` continue to resolve.
- Existing `HeatExchanger` property names and nested JSON structure remain unchanged.

## Approval

- **Status**: Approved.
- **User response**: `Approved.`
- **Approval timestamp**: 2026-07-15T17:56:00Z.
