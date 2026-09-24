# Unit 1 Technology-Stack Decisions

## Decision Summary

| Area | Decision | Rationale |
|---|---|---|
| Language/runtime | Existing supported Python runtime from `pyproject.toml` | Preserve package compatibility and build profiles |
| Contract models | Existing Pydantic v2 models with frozen configuration where appropriate | Validation, typed nested records, and mapping/JSON-compatible round trips |
| Numeric handling | Existing NumPy dependency | Deterministic rectangular conversion, finite checks, flattening, and array compatibility |
| Exceptions | Native Python exceptions plus typed `HPRTargetingError(ValueError)` | Backward-compatible caller behavior and causal chaining |
| Property engine | Existing CoolProp/TESPy leaves only; no contract-layer imports | Preserve dependency direction and prevent engine leakage |
| Unit tests | Existing pytest stack | Repository-standard fixtures, marks, coverage, and profiles |
| Property tests | Existing Hypothesis integration | Domain strategies, shrinking, reproducibility, stateful support for later Unit 2 |
| Coverage | Existing coverage.py/pytest configuration and 95-percent changed-surface gate | Consistent repository quality threshold |
| Static/format checks | Existing Ruff and repository formatting configuration | No duplicate tooling |
| Documentation | Existing Sphinx/Markdown pipeline | Warning-strict contract and migration documentation |
| Packaging | Existing Hatchling/uv source and wheel workflow | No distribution or dependency change |

## Rejected Alternatives

- A new schema/serialization library is rejected because Pydantic already owns
  the public contracts.
- A generic recursive serialization dependency is rejected; explicit detached
  contracts and a small object-graph validator are sufficient.
- A distributed cache, worker, timeout service, or retry library is rejected
  because Unit 1 is call-local synchronous logic.
- A second property-testing framework is rejected because Hypothesis satisfies
  PBT-07 through PBT-09.
- Plain free-form dictionaries for records/diagnostics are rejected because they
  weaken closed validation and round-trip properties.

## Dependency Rules

1. `OpenPinch/contracts` may import Pydantic and detached domain value types,
   but not analysis, application, optimisation, CoolProp, or TESPy.
2. NumPy conversion occurs in analysis/helper code; public records use declared
   existing array support or detached primitive tuples.
3. Engine-type detection is isolated behind analysis boundary helpers and does
   not make contracts import optional engine packages.
4. Tests may import real CoolProp under existing solver/slow profiles; base
   contract tests remain deterministic and fast.
5. No dependency manifest or optional extra changes are expected.

## PBT-09 Verification

Hypothesis is already available in the repository test stack and integrates with
pytest. Unit 1 reuses centralized constrained strategies, keeps shrinking
enabled, and uses fixed or logged seeds according to the repository profile.
This satisfies PBT-09 without a technology change.
