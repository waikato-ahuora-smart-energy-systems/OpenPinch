# Component Dependencies

| Component | Depends on | Consumed by |
|---|---|---|
| Segmented stream domain | `Stream`, `Value`, units | Collections, zones, all thermal services |
| Input normalizer | Schemas, segmented domain | Problem/workspace loading |
| Segment numeric projection | Collections, segmented domain | Problem tables, area targeting |
| Thermal adapters | Segment projection, profile builder | Direct/indirect/HPR/MVR |
| HEN profile model | Prepared arrays, solver abstraction | PDM, TDM, EVM |
| Reporting | Domain and solved HEN data | Public outputs, diagrams, verification |

Data flows from structured input or calculated thermodynamic profiles into one parent stream, expands into segment rows only for thermal calculations, and collapses back to parent-level outputs with nested detail.
