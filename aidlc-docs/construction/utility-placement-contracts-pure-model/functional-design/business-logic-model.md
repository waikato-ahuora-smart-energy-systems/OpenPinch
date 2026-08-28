# Unit 1 Business Logic Model

## Purpose and Boundary

Unit 1 converts public utility-placement values and a detached physical-
feasibility envelope into a validated, immutable placement model. It also owns
the stable result vocabulary that later units populate. Every operation is
deterministic and side-effect free: no process targeting, objective evaluation,
optimiser execution, caching, presentation, or source-study mutation occurs.

## Canonical Vocabulary

- `UtilityPlacementObjective`: `thermodynamic` or `monetary`.
- `UtilityPlacementBaseTarget`: `direct`, `total_site`, or an explicit automatic
  selection value used by the application layer.
- `UtilitySide`: `hot` or `cold`.
- `UtilityLevelKind`: `isothermal` or `sensible`.
- `TemplateKey`: stable `(side, name)` identity plus declared side order.
- `DecisionCoordinate`: template key, field (`supply_temperature` or
  `temperature_span`), lower bound, upper bound, and canonical unit.

Counts apply symmetrically: `N_iso` means exactly `N_iso` hot and `N_iso` cold
isothermal templates; `N_sens` means exactly `N_sens` hot and `N_sens` cold
sensible templates. The minimum valid request therefore contains four levels:
two hot and two cold isothermal levels.

## Canonical Units

The public boundary accepts bare scalars under the problem's configured input
units or established OpenPinch value-with-unit inputs. Normalized values use the
current canonical analysis units:

| Quantity | Canonical value | Result unit metadata |
|---|---|---|
| Absolute temperature | degrees Celsius at contract/model boundaries | `degC`; Unit 2 converts to Kelvin for thermodynamics |
| Temperature difference/span | Celsius-degree difference | `delta_degC` |
| Heat duty/work | kilowatts | `kW` |
| Utility/electricity price | configured canonical price unit | normally `$/MWh` |
| Entropy generation | kilowatts per kelvin | `kW/K` |
| Monetary rate | configured utility-cost output unit | normally `$/h` |
| Dimensionless values | plain finite float | `dimensionless` where exposed |

Unit labels remain explicit in serialized results. Currency symbols are treated
as unit vocabulary, not converted across currencies. Compatible unit
conversion uses existing OpenPinch conversion ownership; unknown or
dimensionally incompatible labels are rejected.

## End-to-End Pure-Model Flow

### 1. Normalize public arguments

1. Reject booleans before integral coercion.
2. Require `isothermal_level_count >= 2` and
   `sensible_level_count >= 0`.
3. Normalize objective, base-target request, period selection, tolerances, and
   optimiser options to specialist enums/value objects.
4. Convert caller lists/mappings to immutable tuples and frozen mappings.
5. Reject unknown fields and non-finite numerical values.
6. For monetary mode, require all economics identified by the request contract;
   generated templates do not invent prices.

The output is a frozen `UtilityPlacementRequest` specification. Its hot/cold
template collections may remain `None` to mean complete deterministic
generation. Normalizing an already normalized request returns an equal detached
value.

### 2. Establish template identities

For each side independently:

- if the template collection is omitted, generate all templates;
- if supplied, require a complete collection—partial generation is not allowed;
- require exactly `N_iso` isothermal and `N_sens` sensible templates;
- retain the full caller declaration index as `placement_rank` and preserve
  declared order in the normalized template tuple;
- require names to be non-empty and unique across the complete request;
- generate omitted names as `hot_iso_1`, `hot_sensible_1`, `cold_iso_1`, and so
  on, using one-based stable ordinals;
- generated order is isothermal then sensible within each side, while explicit
  caller order may interleave kinds;
- generated templates default `cogeneration_eligible` to false and contain no
  utility price.

Template identity never changes after normalization. Candidate validation uses
`placement_rank` for physical ordering and never sorts, merges, or renames a
template to repair a candidate.

This identity pass produces a `TemplateBlueprintSet` before Unit 2 constructs
the feasibility envelope. The later model-building pass combines those same
blueprints with envelope intervals to produce complete effective templates,
avoiding a Unit 1 -> Unit 2 dependency.

### 3. Normalize kind-specific temperature behavior

Every level has a supply-temperature coordinate.

- Isothermal templates have a fixed positive span. The default is
  `0.01 delta_degC`; a caller may supply another valid configured fixed span.
  The span is metadata and consumes no decision coordinate.
- Sensible templates have a positive span interval and consume one additional
  span coordinate. Supplied templates may narrow it; generated templates take
  their valid physical span interval from the feasibility envelope.
- For hot levels,
  `target_temperature = supply_temperature - temperature_span`.
- For cold levels,
  `target_temperature = supply_temperature + temperature_span`.
- Supply and derived target temperatures must both be strictly above absolute
  zero after conversion.

Thus the first-release decision-vector dimension is:

`2 * N_iso + 4 * N_sens`.

The first term is one supply coordinate for every hot/cold isothermal level;
the second is supply plus span for every hot/cold sensible level.

### 4. Validate the detached feasibility envelope

`PlacementFeasibilityEnvelope` is defined in Unit 1 and populated by Unit 2. It
contains:

- ordered, unique period identities and non-negative weights with at least one
  positive weight;
- one physical supply interval per period and template key;
- one physical span interval per period and sensible template key;
- the minimum positive adjacent-level separation;
- approach limits and canonical unit metadata needed to explain bounds;
- source scope/base-target identity as passive metadata.

Envelope keys must match normalized template keys exactly. Missing, duplicate,
unknown, reversed, or non-finite intervals fail before model construction.
Period order is retained for downstream replay, but interval intersection is
mathematically independent of period order.

### 5. Intersect physical and caller bounds

For each coordinate with period intervals `[L_p, U_p]`, compute:

- `physical_lower = max(L_p)`;
- `physical_upper = min(U_p)`.

A caller override is valid only when it narrows or equals the physical
interval. An override that extends below `physical_lower` or above
`physical_upper` is rejected rather than silently clipped. The effective bound
is the valid override when present, otherwise the physical intersection.

An interval is empty only when its lower bound exceeds its upper bound beyond
the named tolerance. A zero-width interval is an explicit fixed coordinate and
remains in the stable vector schema; the existing optimiser contract supports
equal bounds. An isothermal fixed span is metadata rather than a coordinate.
Empty intersections raise
`EmptyPlacementFeasibleRegionError` with template, coordinate, period bounds,
and unit context.

### 6. Tighten physical ordering constraints

Use the normalized `placement_rank` on each side. For adjacent supply
temperatures and minimum separation `d > 0`:

- hot side: `S_i >= S_(i+1) + d`;
- cold side: `S_(i+1) >= S_i + d`.

Constraint propagation tightens the interval chain in both directions until no
bound changes beyond the named bound tolerance. Any `lower > upper` outcome is
an empty feasible region. The algorithm never swaps template identities.

Hot-chain propagation raises upstream lower bounds from downstream requirements
and lowers downstream upper bounds from upstream limits. Cold-chain propagation
raises downstream lower bounds and lowers upstream upper bounds. Because these
are monotone finite chains, a forward/backward pass repeated to stability
terminates without a numerical optimiser.

### 7. Build the vector schema

Coordinates use this fixed family sequence:

1. hot isothermal supply temperatures in their relative declaration order;
2. each hot sensible template's supply then span;
3. cold isothermal supply temperatures in their relative declaration order;
4. each cold sensible template's supply then span.

`DecisionCoordinate` retains `TemplateKey` and `placement_rank`, so vector order
and physical order are related explicitly rather than inferred by sorting.
Encoding requires every declared coordinate exactly once. Decoding requires the
exact vector length, finite values, and in-bound coordinates, then reconstructs
templates in original side declaration order with derived target temperatures.

### 8. Construct deterministic starting candidates

At least one feasible start is produced whenever the tightened interval chain
and span bounds are nonempty.

- For hot supply coordinates, choose the hottest valid first value, then each
  following value as the greatest value not exceeding the previous value minus
  separation and its own upper bound; verify every lower bound.
- For cold supply coordinates, choose the coldest valid first value, then each
  following value as the least value not below the previous value plus
  separation and its own lower bound; verify every upper bound.
- For sensible spans, choose the finite interval midpoint, normalized to
  signed-zero-neutral form.
- Fixed isothermal spans are copied from template metadata.

Optional additional starts may use deterministic midpoint projections, but the
required primary start is always first and no random start is generated in
Unit 1.

### 9. Verify a decoded placement

Verification is deliberately cheaper than finding a placement. It checks:

- exact template-key and coordinate coverage;
- coordinate dimension, finiteness, and effective bounds;
- fixed isothermal spans and sensible span bounds;
- hot/cold target direction and positive absolute temperatures;
- placement-rank ordering and adjacent separation;
- immutable source/template identity preservation.

It returns a structured verification result for an ordinary candidate failure.
Invalid model construction raises a typed exception; later optimiser callbacks
convert candidate verification failures to deterministic penalties.

### 10. Serialize contracts and results

All specialist public models are frozen and reject extra fields. JSON output
contains only primitives, enum values, arrays/objects, unit labels, and finite
numbers. It excludes callables, NumPy arrays, mutable streams/zones, optimiser
objects, and exception instances.

Round-trip equality is exact for enum values, identities, counts, collection
order, booleans, text, and unit labels. Converted or calculated floats compare
with named absolute and relative tolerances. `-0.0` is normalized to `0.0` at
contract construction so signed zero does not create unequal public results.

## Failure Flow

1. Call-shape/count/objective failures -> `PlacementRequestValidationError`.
2. Template identity/kind/direction/economics failures ->
   `UtilityTemplateValidationError`.
3. Unknown or incompatible units -> `UtilityPlacementUnitError`.
4. Missing/mismatched envelope data or empty intersections/order chains ->
   `EmptyPlacementFeasibleRegionError` where physical feasibility is empty,
   otherwise `PlacementModelValidationError`.
5. Decoded candidate failures -> structured `CandidateDiagnostic`, not an
   exception during ordinary search.

Validation exceptions carry stable fields (`code`, `message`, `field_path`,
`template_key`, `period_id`, and `details`) and inherit from the specialist
root. Invalid caller-data subclasses also inherit from `ValueError`; the root
inherits from `RuntimeError` consistently with existing optimisation errors.

## Testable Properties - PBT-01

| Component | Category | Property requirement |
|---|---|---|
| Request normalizer | Idempotence | Normalizing a valid normalized request again yields an equal request. |
| Request contracts | Round-trip | Valid request -> JSON -> request preserves structural fields exactly and floats within named tolerance. |
| Template generator | Invariant | Generated inventories contain exactly `2*N_iso + 2*N_sens` unique identities with exact per-side/kind counts. |
| Template normalizer | Idempotence | Re-normalization preserves identity, declaration order, units, and values. |
| Template normalizer | Invariant | Normalization preserves complete caller template identity/order and produces finite canonical values. |
| Unit normalizer | Oracle | Supported conversions agree with the existing OpenPinch unit-conversion owner within tolerance. |
| Envelope intersection | Commutativity | Permuting periods does not change effective coordinate intervals, although stored period order remains unchanged. |
| Envelope intersection | Oracle | Effective lower/upper bounds equal explicit max-lower/min-upper reference calculations. |
| Order-bound propagation | Invariant | Every accepted bound chain admits the stated hot/cold separation inequalities and every rejected chain has a checkable contradiction. |
| Vector codec | Round-trip | `decode(encode(valid_placement))` equals the normalized placement; `encode(decode(valid_point))` equals the signed-zero-normalized point. |
| Vector schema | Invariant | Dimension always equals `2*N_iso + 4*N_sens`, with every expected coordinate exactly once. |
| Candidate verifier | Easy verification | Every accepted decoded placement independently satisfies keys, bounds, spans, direction, Kelvin positivity, order, and separation. |
| Initial-candidate builder | Easy verification | Every returned start passes the independent candidate verifier. |
| Result contracts | Round-trip | Valid nested result -> JSON -> result preserves all public structure/order and floats within tolerance. |
| Error-context builder | Invariant | Every typed validation error contains a stable non-empty code and field/template/period context when applicable. |

**Induction** is N/A: Unit 1 owns bounded flat template/period collections and
non-recursive transformations. **Additional commutativity** is N/A where caller
or period order is semantically observable; only mathematical interval
intersection is order-independent. Components do not manage mutable state, so
stateful PBT belongs to Unit 3 rather than Unit 1.

These properties are mandatory inputs to Unit 1 Code Generation planning. Each
business-critical path also requires explicit examples; PBT does not replace
the minimum-count, generated-template, invalid-unit, empty-region, and mixed-
level examples.
