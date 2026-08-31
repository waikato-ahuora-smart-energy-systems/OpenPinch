# Unit 1 Domain Entities

## Modeling Conventions

- Public/specialist models are frozen Pydantic contracts with `extra="forbid"`.
- Ordered collections use tuples; lookup maps are derived internally and never
  serialized as mutable shared state.
- Enums serialize to their stable lowercase values.
- Every quantity carries or inherits a declared unit and contains a finite
  signed-zero-normalized float.
- `None` means not applicable or not calculated. It is not interchangeable with
  a zero physical/economic value.
- Source runtime objects and optimiser-private objects are not entity fields.

## Primitive Value Objects

### `QuantityValue`

| Field | Type | Rule |
|---|---|---|
| `value` | `float` | Finite canonical magnitude |
| `unit` | `str` | Non-empty compatible canonical unit |

Purpose: scalar contract value used where the quantity is always present.
Existing public `ValueWithUnit` inputs normalize into this stricter frozen
specialist value rather than changing the shared legacy schema.

### `QuantityInterval`

| Field | Type | Rule |
|---|---|---|
| `lower` | `float` | Finite; no greater than upper |
| `upper` | `float` | Finite; no less than lower |
| `unit` | `str` | Absolute or difference dimension required by owner field |

Purpose: caller, period-physical, and effective coordinate bounds. Equality is
valid and represents a fixed coordinate.

### `PlacementTolerances`

| Field | Type | Rule |
|---|---|---|
| `absolute` | `float` | Finite and non-negative |
| `relative` | `float` | Finite and non-negative |
| `bounds` | `float` | Finite and non-negative |
| `coverage` | `float` | Finite and non-negative; consumed in Unit 2 |
| `ordering` | `float` | Finite and non-negative |

Purpose: one named policy used by model comparisons, serialization checks, and
later numerical service checks.

### `PlacementUnitSystem`

Carries canonical labels for absolute temperature, temperature difference,
heat flow, work, price, entropy, monetary rate, and exergy. It is metadata, not
a conversion engine.

## Identity and Request Entities

### `TemplateKey`

| Field | Type | Rule |
|---|---|---|
| `side` | `UtilitySide` | Hot or cold |
| `name` | `str` | Trimmed, non-empty, globally unique in request |

`TemplateKey` is the stable lookup and diagnostic identity. It does not contain
temperature because placement changes temperatures without changing identity.

### `UtilityLevelTemplate`

| Field | Type | Rule |
|---|---|---|
| `key` | `TemplateKey` | Stable identity |
| `kind` | `UtilityLevelKind` | Isothermal or sensible |
| `placement_rank` | `int` | Unique zero-based rank within side |
| `supply_bounds` | `QuantityInterval or None` | Optional narrowing override |
| `fixed_span` | `QuantityValue or None` | Required only for isothermal |
| `span_bounds` | `QuantityInterval or None` | Optional sensible narrowing override; absent means derive from envelope |
| `price` | `QuantityValue or None` | Required by monetary request |
| `cogeneration_eligible` | `bool` | Hot only; default false |
| `fluid_name` | `str or None` | Optional unless eligible rules require it |
| `fluid_phase` | stable enum/string or `None` | Optional structural turbine metadata |
| `approach_metadata` | immutable optional value object | Must not duplicate process state |

Kind-specific fields are mutually exclusive. The target temperature is derived
from candidate supply/span and is not stored in the template.

### `TemplateBlueprintSet`

Contains the complete generated-or-supplied keys, kinds, placement ranks, and
non-physical metadata required for Unit 2 to build a keyed feasibility envelope.
It contains no effective process-derived bounds. Reusing the blueprint set in
final model construction guarantees identity stability across the Unit 1/2
boundary.

### `UtilityPlacementOptions`

Carries method, seed, iteration/evaluation limits, alternative limit, named
tolerances, minimum separation, minimum sensible span, default isothermal span,
and validated backend overrides. Unit 1 validates generic shape; Unit 2
validates method-specific names through the existing optimiser.

### `UtilityPlacementRequest`

| Field | Type | Rule |
|---|---|---|
| `isothermal_level_count` | `int` | At least two; not bool |
| `sensible_level_count` | `int` | At least zero; not bool |
| `hot_templates` | tuple of `UtilityLevelTemplate` or `None` | Exact supplied inventory or complete-generation sentinel |
| `cold_templates` | tuple of `UtilityLevelTemplate` or `None` | Exact supplied inventory or complete-generation sentinel |
| `objective` | `UtilityPlacementObjective` | Default thermodynamic |
| `base_target` | `UtilityPlacementBaseTarget or None` | Direct, Total Site, or resolvable automatic selection |
| `period_ids` | tuple of `str` or `None` | Ordered unique explicit selection |
| `electricity_price` | `QuantityValue or None` | Required for monetary mode |
| `options` | `UtilityPlacementOptions` | Frozen validated options |
| `units` | `PlacementUnitSystem` | Canonical labels |

Relationships: owns templates/options/unit policy; is independent of any loaded
`PinchProblem` or `Zone`.

## Feasibility Entities

### `CoordinateKey`

Pairs a `TemplateKey` with `DecisionField.SUPPLY_TEMPERATURE` or
`DecisionField.TEMPERATURE_SPAN`.

### `PhysicalCoordinateBound`

| Field | Type | Rule |
|---|---|---|
| `coordinate` | `CoordinateKey` | Must be declared by request schema |
| `bounds` | `QuantityInterval` | Physical interval for one period |
| `reason` | stable string/code | Source profile/approach explanation |

### `PlacementPeriodEnvelope`

| Field | Type | Rule |
|---|---|---|
| `period_id` | `str` | Ordered unique identity |
| `weight` | `float` | Finite and non-negative |
| `coordinate_bounds` | tuple of `PhysicalCoordinateBound` | Every request coordinate exactly once |

### `PlacementFeasibilityEnvelope`

| Field | Type | Rule |
|---|---|---|
| `periods` | tuple of `PlacementPeriodEnvelope` | Non-empty; at least one positive weight |
| `minimum_separation` | `QuantityValue` | Positive temperature difference |
| `approach_limits` | immutable metadata tuple | Diagnostic provenance |
| `scope` | stable string/enum | Direct or Total Site |
| `base_target_id` | `str` | Detached target identity |
| `units` | `PlacementUnitSystem` | Must agree with request |

This is the only process-derived input to Unit 1. Unit 2 constructs it; Unit 1
validates and consumes it without importing Unit 2.

## Pure Placement Model Entities

### `EffectiveUtilityTemplate`

Extends normalized template identity/kind/rank with effective canonical supply
bounds and either fixed span or effective sensible span bounds. It records the
period physical bounds and caller override used to derive each effective
interval for diagnostics.

### `DecisionCoordinate`

| Field | Type | Rule |
|---|---|---|
| `index` | `int` | Contiguous zero-based schema position |
| `coordinate` | `CoordinateKey` | Unique |
| `bounds` | `QuantityInterval` | Canonical, possibly fixed |

### `UtilityTemplateSet`

Owns hot/cold `EffectiveUtilityTemplate` tuples in declaration order plus
derived family views. Family views preserve relative declaration order and do
not own independent mutable copies.

### `UtilityPlacementModel`

| Field | Type | Rule |
|---|---|---|
| `request` | `UtilityPlacementRequest` | Normalized frozen request |
| `envelope` | `PlacementFeasibilityEnvelope` | Validated detached envelope |
| `templates` | `UtilityTemplateSet` | Effective bound model |
| `coordinates` | tuple of `DecisionCoordinate` | Exact stable vector schema |
| `initial_points` | tuple of float tuples | Non-empty; each independently valid |

Generated-pair dimension must equal `N_iso + 2*N_sens`; independent explicit
or inferred dimension must equal `2*N_iso + 4*N_sens`.

### `DecodedUtilityLevel`

Contains template key, kind, placement rank, supply temperature, target
temperature, and span as canonical `QuantityValue` objects. It contains no duty
because duty allocation belongs to Unit 2.

### `DecodedPlacement`

Contains hot and cold tuples of `DecodedUtilityLevel` in original declaration
order plus the exact coordinate tuple. It is the Unit 1 output consumed by Unit
2 candidate evaluation.

### `CandidateDiagnostic`

| Field | Type | Rule |
|---|---|---|
| `code` | stable enum/string | Non-empty machine-readable class |
| `constraint` | stable string | Failed rule name |
| `message` | `str` | Actionable human explanation |
| `side` | `UtilitySide or None` | When applicable |
| `template_key` | `TemplateKey or None` | When applicable |
| `period_id` | `str or None` | Populated by Unit 2 when applicable |
| `measured` | `QuantityValue or None` | Failing observed value |
| `limit` | `QuantityValue or None` | Required bound/tolerance |
| `details` | immutable serializable tuple | No exception/runtime object |

### `CandidateVerification`

Contains `feasible: bool` and ordered diagnostic tuples. A feasible verification
has no blocking diagnostic.

## Result Contract Entities

Unit 1 defines these shapes; Unit 2 supplies numerical contents and Unit 3
exposes/presents them.

### `UtilityLevelPeriodResult`

Stable level identity/kind/rank, solved supply/target/span, allocated duty,
price/eligibility metadata, and applicable diagnostics for one period.

### `ThermodynamicCostBreakdown`

Utility-side entropy, process-side entropy, total entropy generation, ambient
temperature, and exergy destruction with units.

### `MonetaryCostBreakdown`

Thermal purchase cost, cogenerated work, electricity price, electricity credit,
and net monetary objective with units.

### `PlacementPeriodResult`

Period identity/weight, ordered hot/cold level results, residual and allocated
hot/cold duties, coverage residuals/tolerance, feasibility, both optional cost
breakdowns, selected scalar objective, and diagnostics.

### `UtilityPlacementCandidate`

Coordinate tuple, decoded hot/cold levels, ordered period results, aggregate
objective and decompositions, feasibility (always true for a returned
successful candidate), and non-fatal diagnostics.

### `PlacementTermination`

Method, seed, status/code, message, iterations/evaluations where known,
candidate counts, and configured limits. It contains no backend result object.

### `UtilityPlacementResult`

Request metadata, resolved scope/base-target identity, objective, counts,
period identities/weights, units, best candidate, deterministically ordered
alternatives, termination, and result-level diagnostics. Best and alternatives
contain feasible candidates only.

## Error Entities

```text
UtilityPlacementError(RuntimeError)
+- PlacementRequestValidationError(ValueError)
+- UtilityTemplateValidationError(ValueError)
+- UtilityPlacementUnitError(ValueError)
+- PlacementModelValidationError(ValueError)
+- EmptyPlacementFeasibleRegionError(ValueError)
```

Later units may add targeting, objective, turbine, and optimiser-exhaustion
subclasses below the same root. Each instance exposes serializable context but
exceptions themselves are never nested in public result JSON.

## Entity Relationships

```text
UtilityPlacementRequest
+- UtilityLevelTemplate (optional hot/cold tuples)
+- TemplateBlueprintSet (derived complete identity view)
+- UtilityPlacementOptions
+- PlacementUnitSystem

PlacementFeasibilityEnvelope
+- PlacementPeriodEnvelope
   +- PhysicalCoordinateBound

UtilityPlacementRequest + PlacementFeasibilityEnvelope
+- UtilityPlacementModel
   +- UtilityTemplateSet -> EffectiveUtilityTemplate
   +- DecisionCoordinate tuple
   +- initial point tuples

UtilityPlacementModel + coordinate tuple
+- DecodedPlacement -> DecodedUtilityLevel

UtilityPlacementResult
+- UtilityPlacementRequest metadata
+- UtilityPlacementCandidate (best and alternatives)
|  +- PlacementPeriodResult -> UtilityLevelPeriodResult
|     +- ThermodynamicCostBreakdown
|     +- MonetaryCostBreakdown
+- PlacementTermination
```

The diagrams are plain text trees with one parent-to-child relationship per
line; they contain no crossing connectors or unlabeled branches.

## PBT-01 Entity Ownership

| Entity/component | Identified categories | Required property focus |
|---|---|---|
| Quantity values/intervals/unit system | Invariant, idempotence, oracle | Finiteness, interval order, signed-zero normalization, supported conversions |
| Request/template contracts | Round-trip, invariant, idempotence | JSON, counts, identity/order, kind fields, detached normalization |
| Feasibility envelope | Invariant, commutativity, oracle | Exact key/period coverage and order-independent interval intersection |
| Effective template/model builder | Invariant, easy verification | Bound narrowing, chain feasibility, exact dimension, verified starts |
| Vector codec/decoded placement | Round-trip, invariant | Both codec directions, identity, direction, bounds, fixed spans |
| Result contracts | Round-trip, invariant | Nested JSON, candidate order, serializable finite detached values |
| Candidate diagnostic/error context | Invariant | Stable codes and applicable field/template/period evidence |
| Enum declarations | No separate PBT property | Exhaustive example/contract tests are clearer for two/few fixed values |
| Exception inheritance | No separate PBT property | Static/example inheritance tests are sufficient; behavior is not input-wide transformation |

Induction is N/A because no recursive entity exists. Mutable-state sequence
properties are N/A because every Unit 1 entity is frozen; Unit 3 owns the only
planned observation cache and workspace state interactions.
