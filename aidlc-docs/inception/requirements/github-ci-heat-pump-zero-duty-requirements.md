# GitHub CI Heat-Pump Zero-Duty Regression Requirements

## Intent Analysis

- **User request**: Diagnose and repair the three failing GitHub CI tests in the
  heat-pump cycle suite.
- **Request type**: Bug fix.
- **Scope estimate**: Single component with focused regression coverage.
- **Complexity estimate**: Simple implementation with moderate numerical risk
  because the failure is platform dependent.
- **Requirements depth**: Minimal. The supplied traceback, live workflow run,
  existing tests, and current stream-domain rules define the expected behavior
  without further clarification.

## Diagnosed Cause

The failing cycles have zero external condenser duty (`Q_heat == 0`) and
non-zero cascade duty, so they retain positive refrigerant mass flow. On Linux,
the CoolProp pressure-enthalpy round trip leaves a microscopic difference
between the two nominally identical zero-duty condenser states. The current
exact-zero profile guard accepts that residue, while the segmented-stream
validator correctly rejects the resulting near-zero-duty segment.

The authoritative physical condition is the requested external process duty,
not the recomputed enthalpy-profile span. A zero external duty must therefore
produce no corresponding process stream.

## Functional Requirements

- **FR-CI-01**: A solved vapour-compression cycle with
  `abs(Q_heat) <= OpenPinch.lib.config.tol` must emit no condenser process
  stream, even if CoolProp produces a non-zero floating-point profile residue.
- **FR-CI-02**: A solved vapour-compression cycle with
  `abs(Q_cool) <= OpenPinch.lib.config.tol` must emit no evaporator process
  stream under the same conditions.
- **FR-CI-03**: Non-zero condenser and evaporator process duties must retain
  their current segmented-stream behavior. The magnitude check must preserve
  intentional negative evaporator duties, which are represented as hot streams.
- **FR-CI-04**: Cascade stream collections must remain the union of the streams
  emitted by their stages; zero-duty stage sides contribute no stream.
- **FR-CI-05**: Strict monotonic heat-coordinate validation and the positive-duty
  invariant for segmented streams must remain unchanged.

## Non-Functional Requirements

- **NFR-CI-01**: Behavior must be deterministic across supported platforms with
  Python 3.14 and CoolProp 7.2.
- **NFR-CI-02**: The repair must be narrowly scoped to vapour-compression cycle
  process-stream emission; generic profile validation and CI dependency policy
  are out of scope.
- **NFR-CI-03**: Existing positive-duty, negative-duty, cascade, and segmented
  stream behavior must remain backward compatible.
- **NFR-CI-04**: Regression coverage must not depend on a particular CoolProp
  platform result. It must inject a deterministic one-ULP enthalpy residue with
  `numpy.nextafter`.
- **NFR-CI-05**: The zero-duty omission invariant must receive focused
  property-based coverage using the project's existing Hypothesis framework,
  in addition to the concrete reported regression case.

## Acceptance Criteria

- The three test nodes reported by GitHub Actions pass.
- A deterministic one-ULP zero-condenser-duty regression test returns an empty
  stream collection without raising an exception.
- Symmetric zero-evaporator-duty coverage returns an empty collection.
- Existing non-zero and negative-duty heat-pump tests still pass.
- Existing cascade union and aggregate-duty assertions still pass.
- Generic segmented-profile tests continue to reject duplicate, near-zero, and
  reversing heat coordinates.
- The focused heat-pump and stream-profile suites pass locally, followed by the
  full supported test suite.

## Extension Compliance

- **Security Baseline**: Disabled in `aidlc-state.md`; skipped.
- **Resiliency Baseline**: Disabled in `aidlc-state.md`; skipped.
- **Property-Based Testing (Partial)**:
  - **PBT-02**: N/A; no inverse or round-trip operation is changed.
  - **PBT-03**: Applicable; NFR-CI-05 requires coverage of the zero-duty omission
    invariant.
  - **PBT-07**: Applicable; generated residual profiles must use constrained,
    finite thermodynamic coordinates.
  - **PBT-08**: Applicable; existing Hypothesis shrinking and seed reporting must
    remain enabled.
  - **PBT-09**: Compliant; Hypothesis is already selected and locked for Python.

## Non-Goals

- Relaxing segmented-stream monotonicity or positive-duty validation.
- Broadly changing profile deduplication thresholds in shared utilities.
- Changing CoolProp or NumPy versions to mask the platform difference.
- Modifying GitHub Actions workflow structure.
