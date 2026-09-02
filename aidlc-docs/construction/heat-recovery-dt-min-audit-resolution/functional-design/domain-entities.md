# Heat-Recovery DT_MIN Audit Resolution Domain Entities

No new public entity is introduced.

## Existing entities with strengthened invariants

- `HeatRecoveryQuantity`: frozen finite numeric magnitude plus explicit unit.
  Its enclosing result determines the required physical dimension.
- `HeatRecoveryDtMinResult`: frozen specialist result with canonical scope and
  period metadata, dimensionally valid quantities, status-specific semantics,
  and tolerance-aware cross-field consistency.
- `HeatRecoveryDtMinSolution`: canonical-unit internal numerical result. It is
  emitted only after bracket post-verification.
- `Zone`: accepted as a scope selector by address. Streams and configuration
  always come from the current problem's locally resolved entity.
- `Value`: accepted only when scalar for this service; its general-purpose
  permissive constructors do not define the service input boundary.

## Testable properties

- **Round-trip, PBT-02**: every generated valid result survives JSON
  serialization and strict deserialization unchanged.
- **Invariant, PBT-03**: recovery is non-increasing with precise global
  `dt_min`; returned temperatures stay within the no-overlap bound; compatible
  units do not change the canonical solution; inputs and caches remain
  unchanged.
- **Idempotence, PBT-04**: repeated calls with equivalent inputs return equal
  results and leave observable problem state unchanged.
- **Oracle, PBT-05**: precise inverse results agree with analytical two-stream
  cases and detached uniform-`dt_min` forward targeting.
- **Easy verification**: the returned bracket sides satisfy opposite recovery
  predicates, and the result fields satisfy dimensional and arithmetic
  relationships.
- **Ordering and commutativity**: stream order does not change results, period
  results retain canonical order, and sequential and parallel all-period calls
  agree.
- **PBT-06**: N/A because the service owns no persistent mutable state.

Reusable generators must include multiple active streams, segmented and
sensible profiles, inactive streams, thresholds, no-overlap cases,
tolerance-scale duties, arbitrary valid period identifiers, and compatible
unit representations. Fixed seeds and shrinking remain enabled. Each audit
counterexample also receives an explicit example regression under PBT-10.
