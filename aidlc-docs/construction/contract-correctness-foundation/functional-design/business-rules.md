# Unit 1 Business Rules

1. Period outputs must be non-empty and contain identical unique target keys.
2. Weights must be finite, non-negative, aligned with output count, and have a
   positive total.
3. Weighted scalar values use compatible units and normalized weights.
4. Array-valued report fields are not aggregated unless assigned a dedicated
   policy.
5. Partially missing optional diagnostics resolve to `None`.
6. Partially missing required duty, cost, or design fields raise an error naming
   the public field.
7. Peak-design fields use the maximum across periods after unit conversion.
8. Utility order is deterministic and follows first appearance.
9. Aggregation never mutates source outputs or their target rows.
10. Root exports and the supported public inventory are deliberate exact sets.
11. Regression tests may initially specify behavior implemented by later units,
    but final Build and Test must leave no expected-failure compatibility layer.
12. Closed workflow strings and selector config keys are excluded from the
    target contract rather than translated.

## Error Handling

Structural alignment and required-field failures raise `ValueError` with the
field or target-key mismatch. Missing cached state raises `RuntimeError` with an
explicit target/design method suggestion. Optional diagnostics do not raise.
