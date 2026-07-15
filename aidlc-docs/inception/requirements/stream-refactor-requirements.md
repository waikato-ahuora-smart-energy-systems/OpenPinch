# Stream Model Refactor Requirements

## Intent

Refactor the 1,388-line `OpenPinch/classes/stream.py` module so that value/period handling, derived thermodynamic calculations, and segmented-profile calculations have explicit private boundaries. This is an internal maintainability change and must not alter thermal results, mutation semantics, imports, serialization, or persistence.

## Approved Scope

- Extract three stateless private helper modules.
- Keep `Stream` and `StreamSegment` physically defined in `stream.py`.
- Keep state mutation, ownership propagation, transaction rollback, and public properties/methods on the classes.
- Preserve existing private method entry points as thin wrappers where tests or internal callers use them.
- Extract and verify one responsibility at a time.

## Functional Requirements

- SR-FR-01: `_stream_value_state.py` owns pure value coercion, period normalization, vector expansion, value construction/copying, and period-count validation.
- SR-FR-02: `_stream_thermodynamics.py` owns pure calculations for completed core state and derived temperature, CP, resistance, cost, and entropic-mean values.
- SR-FR-03: `_stream_profile.py` owns pure profile-point validation, segment continuity/direction validation, detached segment cloning, and aggregate duty/effective-HTC calculations.
- SR-FR-04: `Stream` retains orchestration, property setters, revision updates, period-context mutation, segment ownership, atomic replacement/update, inversion, and rollback.
- SR-FR-05: `StreamSegment` retains attached-child mutation routing and inherited metadata protection.
- SR-FR-06: `Stream.from_temperature_heat_profile`, `replace_segments`, `update_segment`, `_calculate_missing_properties`, `update_derived_properties`, and existing private utility methods remain callable with the same signatures.

## Compatibility Requirements

- SR-NFR-01: `Stream` and `StreamSegment` remain importable from `OpenPinch`, `OpenPinch.classes`, and `OpenPinch.classes.stream`.
- SR-NFR-02: Both classes retain `OpenPinch.classes.stream` as their defining module for pickle compatibility.
- SR-NFR-03: Existing values, units, properties, exception types/messages, tolerance behavior, revisions, and mutation rollback remain unchanged.
- SR-NFR-04: Existing deepcopy, pickle, and structured serialization round trips remain unchanged.
- SR-NFR-05: No new helper is exported from a package barrel or documented as public API.
- SR-NFR-06: Constant-CP and segmented-stream numerical results remain identical to the pre-refactor baseline.
- SR-NFR-07: Security and resiliency extensions remain disabled. Existing Partial property-based enforcement remains applicable.

## Acceptance Criteria

- Each extracted calculation has one implementation and no duplicate remains in `stream.py`.
- Class methods delegate to the private helpers without moving transactional state ownership out of the classes.
- Existing focused Stream, StreamSegment, linearisation, input-preparation, collection, and targeting tests pass after each phase.
- Serialization, deepcopy, pickle, class module identity, ordered continuity, atomic rollback, multiperiod aggregation, and flat-versus-segmented parity remain covered.
- Full proportional non-solver and segmented solver-marked suites, Ruff, formatting, documentation, packaging, and patch validation pass before completion.
