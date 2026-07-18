# Unit 1 Domain Entities

| Entity | Role |
|---|---|
| `TargetOutput` | Serializable selected-period or aggregated output. |
| `TargetResults` | One aligned target row with thermal, cost, HPR, and diagnostic fields. |
| `AggregationPolicy` | Internal classification of a report field as weighted, peak, derived, consensus, optional, or collection. |
| `PublicOperation` | Test-owned record of owner, name, behavior class, support level, and tutorial owner. |
| `ProblemState` | Conceptual prepared, targeted, designed, or invalidated state used by contracts. |
| `TutorialContract` | Planned notebook identity, profile, public operations, and semantic modes. |

No new persistence entity or public schema is introduced by Unit 1. Test-owned
contract records remain internal evidence and do not expand the package root.
