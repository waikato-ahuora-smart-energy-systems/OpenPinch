# Code Generation Summary: Heat-Pump Zero-Duty CI Repair

## Outcome

Vapour-compression process-stream construction now omits condenser or
evaporator streams when the corresponding external duty magnitude is within the
shared project tolerance. The duty check occurs before CoolProp-derived profile
construction, preventing platform-specific one-ULP state residues from reaching
strict segmented-stream validation.

## Modified Files

- `OpenPinch/services/heat_pump_integration/unit_models/vapour_compression_cycle.py`
  imports the shared `tol` constant and gates condenser and evaporator profile
  construction using the authoritative process-duty magnitude.
- `tests/test_classes/test_simple_heat_pump_cycle.py` contains deterministic
  condenser/evaporator one-ULP regressions and a generated zero-duty invariant.

## Created Files

- `tests/strategies/heat_pump_cycles.py` defines constrained, finite
  Hypothesis cases for zero-duty process sides.
- This summary records Code Generation scope and traceability.

## Requirements Traceability

- **FR-CI-01 / FR-CI-02**: Condenser `Q_heat` and evaporator `Q_cool`
  control whether their process profiles are constructed.
- **FR-CI-03**: Absolute magnitude preserves intentional negative duties above
  tolerance.
- **FR-CI-04**: Cascade aggregation remains unchanged and inherits corrected
  stage collections.
- **FR-CI-05**: Generic monotonicity and positive-duty validators are unchanged.
- **NFR-CI-01 / NFR-CI-04**: Tests inject `numpy.nextafter` residues rather
  than relying on platform-specific CoolProp output.
- **NFR-CI-03 / NFR-CI-05**: Existing examples remain, complemented by a
  constrained Hypothesis invariant.

## Numerical Rationale

External process duty is the authoritative signal for process-stream emission.
A zero process duty can coexist with positive refrigerant mass flow when the
remaining condenser or evaporator duty belongs to a cascade stage. Testing only
the recomputed raw enthalpy span is therefore insufficient and platform
sensitive. Shared profile validation remains strict so every emitted segmented
stream retains positive, monotonic duty.

## Generated Test Coverage

- Deterministic condenser and evaporator cases at the reported enthalpy scale.
- An assertion that a skipped process side never invokes its property-profile
  builder.
- Hypothesis cases spanning both process sides, finite positive mass flows, and
  all finite duties from `-tol` through `tol`.
- Existing positive-duty, negative-duty, cascade, and profile-invariant tests
  remain available for Build and Test.

## Verification Boundary

Test execution is intentionally deferred to the subsequent Build and Test stage
required by the approved workflow. Code Generation closes with scoped Python
syntax, whitespace, diff, and workspace-integrity checks only.

## Extension Compliance

- **Property-Based Testing (Partial)**: PBT-03, PBT-07, PBT-08, and PBT-09 are
  implemented as planned; PBT-02 is N/A because no round-trip exists.
- **Security Baseline**: Disabled; skipped.
- **Resiliency Baseline**: Disabled; skipped.
