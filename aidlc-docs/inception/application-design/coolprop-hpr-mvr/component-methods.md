# Component Methods: CoolProp HPR and MVR Reliability

## Conventions

Signatures are high-level design contracts, not final code. Exact module names,
field defaults, and exception subclasses are finalized during Functional Design.
All diagnostic text is sanitized and length-bounded.

## Contract Types

### HPRSearchBudget

**Owner**: `OpenPinch/contracts/hpr.py`

```python
@dataclass(frozen=True)
class HPRSearchBudget:
    maximum_iterations: int
    maximum_evaluations: int
```

Both values are positive. Defaults preserve a bounded version of current
behavior. The same instance is carried by `HeatPumpTargetInputs` and
`MultiPeriodHPRTargetInputs`.

### HPRFailureDiagnostic

```python
@dataclass(frozen=True)
class HPRFailureDiagnostic:
    category: HPRFailureCategory
    reason_code: str
    summary: str
    fluid: str | None = None
    stage_index: int | None = None
    period_id: str | None = None
```

The category is closed and engine-neutral. `summary` never contains an
unbounded traceback or serialized engine state.

### HPRFailureSummary

```python
@dataclass(frozen=True)
class HPRFailureSummary:
    simulation_backend: str
    cycle: str
    evaluated_count: int
    category_counts: Mapping[HPRFailureCategory, int]
    representative_failures: tuple[HPRFailureDiagnostic, ...]
    budget: HPRSearchBudget
    warm_start_evaluated: bool
    warm_start_viable: bool
```

The representative tuple has a fixed small maximum length. Counts may grow;
retained messages may not.

### HPRTargetingError

```python
class HPRTargetingError(ValueError):
    diagnostics: HPRFailureSummary
```

Used only when the public service cannot return a viable result or preflight
rejects the request. Its message is concise; callers may inspect
`diagnostics`.

### DirectGasMVRFallbackDiagnostic

```python
@dataclass(frozen=True)
class DirectGasMVRFallbackDiagnostic:
    policy: Literal["dry_stage", "reduced_profile"]
    reason_code: str
    source_stream: str
    period_index: int
    stage_index: int
    fluid: str
```

`DirectGasMVRStageResult` gains an immutable collection of these records.

## Public Accessor Methods

Every existing public HPR method adds:

```python
def hpr_method(
    ...,
    maximum_restarts: int | None = None,
    maximum_iterations: int | None = None,
    maximum_evaluations: int | None = None,
) -> BaseTargetModel:
    ...
```

This applies to Carnot, vapour-compression, and MVR heat-pump/refrigeration
surfaces. `_hpr(...)` resolves and forwards the controls. Explicit named
arguments take precedence over keys in `options`; `options` takes precedence
over configuration defaults. Invalid booleans, non-integral values, zero, and
negative limits fail before targeting.

## Request Preparation

```python
def resolve_hpr_search_budget(
    *,
    maximum_iterations: object | None,
    maximum_evaluations: object | None,
    defaults: HPRSearchBudget,
) -> HPRSearchBudget:
    """Validate and return one bounded search budget."""
```

```python
def construct_HPRTargetInputs(
    ...,
    search_budget: HPRSearchBudget,
) -> HeatPumpTargetInputs:
    """Prepare one single-period request with a validated budget."""
```

Multiperiod preparation copies the same budget into
`MultiPeriodHPRTargetInputs`; it does not silently restore optimiser defaults.

## CoolProp Preflight

```python
def preflight_coolprop_hpr_targeting(
    args: HeatPumpTargetInputs | MultiPeriodHPRTargetInputs,
) -> PreparedCoolPropHPRSpecification:
    """Prove configured fluids and required state envelopes before search."""
```

The prepared specification contains normalized fluid identities and detached
capability facts. It is passed to topology handlers so candidate evaluation does
not repeat capability discovery.

```python
def validate_direct_mvr_compression_request(
    *,
    fluid: str,
    inlet_temperature: float,
    inlet_pressure: float,
    compression_target: tuple[str, float],
    source_stream: str,
    period_index: int,
) -> PreparedDirectMVRStageCapability:
    """Reject unsupported process-MVR requests before stage solving."""
```

## Candidate Evaluation

```python
def evaluate_hpr_candidate(
    *,
    objective: HPRObjective,
    point: Sequence[float],
    args: HPRTargetInputs | MultiPeriodHPRTargetInputs,
    artifact_mode: Literal["search", "final"],
    prepared_coolprop: PreparedCoolPropHPRSpecification | None = None,
) -> HPRBackendResult:
    """Evaluate one point without hiding fatal failures."""
```

Search mode returns scalar and diagnostic facts only. Final mode constructs
streams, economics, and the detached simulation record for one accepted point.

```python
def normalize_hpr_penalty_terms(value: object) -> tuple[float, ...]:
    """Flatten documented scalar/array penalty forms to finite scalar terms."""
```

Unsupported shapes and non-numeric values are contract errors, not candidate
penalties.

## Search Coordination

```python
def run_hpr_candidate_search(
    *,
    objective: HPRObjective,
    initial_points: Sequence[Sequence[float]],
    bounds: Sequence[Sequence[float]],
    args: HPRTargetInputs | MultiPeriodHPRTargetInputs,
    budget: HPRSearchBudget,
    diagnostics: HPRDiagnosticAccumulator,
    optimiser: OptimisationRunner = run_multistart_minimisation,
) -> HPRCandidateSearchResult:
    """Evaluate warm starts first, then run bounded global search."""
```

`HPRCandidateSearchResult` contains ranked unique candidates, cached
search-time outcomes, budget state, and the best viable warm start. It contains
no engine object.

```python
def solve_hpr_placement(
    ...,
    prepared_coolprop: PreparedCoolPropHPRSpecification | None = None,
) -> HPRBackendResult:
    """Select a viable point, run one final evaluation, or raise typed failure."""
```

The multiperiod solver uses the same coordinator and exception contract. Exact
point keys are normalized once so backend results and warm starts share cached
evaluations.

## Diagnostics

```python
class HPRDiagnosticAccumulator:
    def record(self, diagnostic: HPRFailureDiagnostic) -> None: ...
    def freeze(self, *, budget_state: HPRBudgetState) -> HPRFailureSummary: ...
```

`record` increments category counts but retains at most the configured
representative limit. It never accepts arbitrary exception objects.

```python
def classify_candidate_failure(exc: Exception) -> HPRFailureDiagnostic | None:
    """Return a recoverable diagnostic only for an approved exception class."""
```

A `None` classification means re-raise the original exception with its
traceback.

## Result Finalization

```python
def finalize_hpr_output(result: HPRBackendResult) -> HeatPumpTargetOutputs:
    """Build and validate a detached public output."""
```

The method:

1. requires successful streams and a simulation record;
2. recursively finalizes period outputs;
3. omits the internal model from `to_output_fields()`;
4. explicitly sets public `model=None`;
5. validates the public schema and copy safety.

`HPRBackendResult.model` may remain an internal convenience during final
evaluation. It is never returned by `to_output_fields()`.

## Direct Process-MVR

```python
def record_direct_mvr_fallback(
    *,
    policy: Literal["dry_stage", "reduced_profile"],
    context: DirectMVRStageContext,
    cause: Exception,
) -> DirectGasMVRFallbackDiagnostic:
    """Map one approved property failure to bounded stage evidence."""
```

Only explicitly classified saturation lookup failures may select
`dry_stage`; only explicitly classified unavailable saturation breakpoints may
select `reduced_profile`. All other property errors are translated to a
stage-context `ValueError`-compatible domain exception and chained from the
original cause.

## Compatibility Rules

- Existing callers that omit budget parameters continue to work.
- `HeatPumpTargetOutputs.model` remains readable and evaluates to `None`.
- Existing target fields retain their names and meaning.
- TESPy preflight and explicit backend selection remain unchanged.
- No raw CoolProp message is part of a stable public contract.
