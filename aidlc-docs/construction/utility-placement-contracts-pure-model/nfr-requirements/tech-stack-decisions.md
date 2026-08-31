# Unit 1 Technology Stack Decisions

## Decision Summary

Unit 1 uses the existing OpenPinch core and development stack. It adds source
modules and tests only; it adds no runtime, optional, build, or test dependency.

| Concern | Decision | Local evidence/rationale |
|---|---|---|
| Runtime | Python `>=3.14.2` | Existing `pyproject.toml` project requirement and CI runtime |
| Public/specialist schemas | Pydantic v2 API under existing `pydantic<3` dependency | Existing contract layer already uses Pydantic v2 validators/configuration |
| Units | Existing `OpenPinch.domain.value.Value` and Pint under `pint<1` | Reuses canonical registry and dimensional conversion; no second unit system |
| Immutable collections/math | Standard-library tuples, frozen models, enums, `math` | Sufficient for the pure model and prevents backend object leakage |
| Test runner | pytest `>=9.0.3` development dependency | Existing suite and CI runner |
| Property testing | Hypothesis `>=6.135.0` development dependency | Custom strategies, shrinking, seeds, pytest integration |
| Coverage | coverage.py `>=7.14.3`, branch mode | Existing 95% repository gate |
| Lint/static gate | Ruff `>=0.15.8`, target `py314` | Existing repository configuration and CI |
| Build/distribution | Hatchling `>=1.26` | Existing wheel/source distribution |

The planning question originally used the broader phrase Python 3.11+. Local
inspection established that OpenPinch requires Python `>=3.14.2`; this stricter
existing contract governs Unit 1 and is recorded in the answered plan.

## Production Module Decisions

### Contract owner

New frozen specialist schemas belong in
`OpenPinch/contracts/utility_placement.py` or a cohesive package of the same
owner if file size requires decomposition. They use:

- `BaseModel` and `ConfigDict(frozen=True, extra="forbid")`;
- Pydantic field/model validators for local structural invariants;
- tuples for ordered nested values;
- stable `StrEnum` values for objective, side, kind, scope, decision field, and
  diagnostic code;
- explicit unit-bearing specialist values rather than modifying shared legacy
  `ValueWithUnit` behavior.

This concrete module is a specialist import. `OpenPinch/__init__.py` remains the
two-symbol facade.

### Pure analysis owner

Pure operations belong under `OpenPinch/analysis/utility_placement/`, separated
by responsibility when implemented:

- `errors.py`: specialist root and Unit 1 typed validation failures;
- `normalization.py`: public request, units, templates, and blueprint creation;
- `bounds.py`: envelope validation, intersection, order propagation, and starts;
- `codec.py`: decision schema, encode, decode, and verification;
- `models.py` only for internal immutable values not appropriate in public
  contracts.

The package must not import application accessors, presentation, targeting
services, power/turbine services, or optimisation backends. Unit 2 may consume
these Unit 1 owners, never the reverse.

### Numerical implementation

Use built-in `math.isfinite`, `math.isclose`, `max`, `min`, tuple iteration, and
explicit chain passes. Unit 1 does not require NumPy arrays: keeping public and
pure-model values as Python floats/tuples simplifies JSON, equality, shrinking,
and detached-state guarantees. Existing NumPy remains a project dependency but
is not a Unit 1 design dependency.

### Unit conversion

Use the canonical OpenPinch `Value`/Pint registry for dimensional validation and
conversion. Do not instantiate another `UnitRegistry`, create a parallel unit
alias table, parse units manually, or perform currency conversion. Convert at
normalization boundaries, then store canonical magnitudes and labels.

## Test Stack and Organization

### Example-based tests

Focused examples should be grouped by behavior, for example:

- `tests/contracts/test_utility_placement_contracts.py`;
- `tests/analysis/utility_placement/test_normalization.py`;
- `tests/analysis/utility_placement/test_bounds.py`;
- `tests/analysis/utility_placement/test_codec.py`;
- `tests/analysis/utility_placement/test_errors.py`.

They pin minimum counts, count-only generation, mixed kinds, units, monetary
missing inputs, empty intersections, fixed bounds, ordering, vector dimension,
known encoding, JSON, and error context.

### Property-based tests

Reusable strategies belong in
`tests/strategies/utility_placement.py`. Property modules use an explicit
`*_properties.py` suffix or clearly named property classes/functions so example
and generated evidence remain distinguishable. Strategies generate complete
domain types, including:

- valid counts and complete/omitted template inventories;
- unique identities and interleaved kinds;
- compatible canonical/non-canonical quantities;
- ordered periods, non-negative weights, and nonempty physical intersections;
- feasible hot/cold bound chains with fixed and variable coordinates;
- nested finite serializable result contracts;
- targeted invalid mutations for typed-error invariants.

Raw primitive strategies alone are not sufficient for domain parameters.

### Reproducibility and shrinking

Do not disable Hypothesis shrinking, example database behavior, deadlines, or
health checks globally to hide failures. CI commands already provide
`--hypothesis-seed=20260715` for the `not solver` suite. Failure output shall
retain the seed and shrunk example. A shrunk defect becomes a permanent focused
example before closure.

### Performance evidence

Use a deterministic valid model fixture matching U1-NFR-003. The performance
test records warm-up, ten measurements, the 95th percentile, Python/runtime
identity, `P`, `L`, `D`, and interval count. It shall avoid file/network I/O and
test the combined pure pipeline only. Scaling evidence compares fixtures that
vary one dimension at a time.

## PBT-09 Compliance

| Verification criterion | Status | Evidence |
|---|---|---|
| Framework selected and documented | Compliant | Hypothesis selected here and in approved requirements |
| Included in project dependencies | Compliant | `hypothesis>=6.135.0` in the `dev` dependency group |
| Custom domain strategies supported | Compliant | Hypothesis composite/flat-map strategy capability; strategy owner specified above |
| Automatic shrinking supported and enabled | Compliant | Hypothesis default shrinking; no disablement approved |
| Seed-based reproduction supported | Compliant | Existing CI uses `--hypothesis-seed=20260715` |
| Existing runner integration | Compliant | Hypothesis integrates with the repository's pytest commands |
| Primary language covered | Compliant | Unit 1 is Python-only |

PBT-09 has no blocking finding. PBT-01 property requirements from Functional
Design are carried into the proposed property modules. PBT-02 through PBT-08
and PBT-10 become blocking during Code Generation planning/generation according
to the extension applicability matrix.

## Alternatives Rejected

| Alternative | Reason rejected |
|---|---|
| Dataclasses only for public contracts | Would duplicate established schema validation/JSON conventions and reduce installed API consistency |
| A new unit library or manual conversion table | Would create competing registries and dimensional behavior |
| NumPy arrays in public models | Poor JSON boundary and increased mutable/backend-like leakage risk |
| A custom random-test harness | Lacks Hypothesis shrinking, strategy ecosystem, and existing pytest/CI integration |
| A new root export | Violates the approved package usability contract |
| A new microservice/framework | No network/deployment requirement exists and Infrastructure Design is skipped |

## Compatibility Gates

Implementation is acceptable only when it preserves current root imports,
passes core/specialist installed-package smoke, requires no lockfile dependency
addition for Unit 1, and leaves existing Pydantic contracts and unit behavior
unchanged.
