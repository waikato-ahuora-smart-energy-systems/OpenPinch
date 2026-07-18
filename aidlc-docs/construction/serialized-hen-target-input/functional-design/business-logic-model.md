# Business Logic Model

1. A runtime network serializes with `model_dump(mode="json")`.
2. The caller places that mapping under `TargetInput.network`.
3. Pydantic constructs the independent nested transport schemas and applies
   the same observable invariants as the runtime records.
4. TargetInput JSON-mode serialization emits the same canonical mapping.
5. Existing canonical problem preparation retains the network mapping without
   interpreting it as a synthesis instruction.

## Testable Properties

- **Round-trip**: TargetInput JSON serialization and deserialization preserves
  the canonical network mapping.
- **Invariant**: exchanger order, period order, and area-slice order are preserved.
- **Invariant**: runtime and transport JSON-visible key sets remain aligned.
