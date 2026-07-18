# Business Logic Model

## Canonical Boundaries

- Transport schemas retain compact wire keys and translate them once into descriptive
  runtime constructor arguments.
- `Stream` exposes descriptive state and derived-value properties backed by private
  compact storage.
- Analysis services consume only descriptive runtime properties.
- Enum consumers import full enum classes directly from their concrete owner.
- Design workflows return one closed view containing an explicit result model.

## Data Flow

1. Compact JSON validates strictly through `TargetInput`.
2. Input construction maps wire fields to descriptive `Stream` arguments.
3. Target, HPR, HEN, graph, and report services use the canonical runtime vocabulary.
4. Canonical serialization maps domain state back to the unchanged compact wire shape.
5. Workspace persistence requires an explicit version and strict case payload shape.

## Testable Properties

- **PBT-02 Round trip**: valid compact input, including HEN data, survives validation,
  runtime construction, canonical serialization, and JSON restoration.
- **PBT-03 Invariant**: descriptive stream mutation preserves period ordering, units,
  derived-state consistency, and segmented-stream atomicity.
- **PBT-03 Invariant**: closed public inventories contain canonical names only.
- **PBT-07**: reuse structured stream, period-output, HEN, and public-argument strategies.
- **PBT-08/PBT-09**: Hypothesis shrinking remains enabled under seed `20260715`.

