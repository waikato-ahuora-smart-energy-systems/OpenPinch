# Unit 1 Code Generation Summary

## Outcome

Unit 1, TESPy HPR Performance-Map Contract and Golden Fixtures, is complete.
OpenPinch now owns a strict immutable alpha-1.0 plain-data contract for active
heat-pump and refrigeration performance points, a canonical request grid, a
generated JSON Schema, representative heat-pump and refrigeration fixtures,
and installed-package resource access.

The contract matches the corrected OpenUtility multi-period HPR boundary:
mode-specific useful duty and COP, explicit reference-capacity basis, adjacent
ordered part-load curves, separate energy and temperature tolerances, and
structured JSON-only provenance. It does not import OpenUtility or an optimizer.

## Production and Resource Files

| File | Responsibility |
|---|---|
| `OpenPinch/contracts/hpr_performance_map.py` | Frozen extra-forbid request, units, point, map, recursive JSON provenance, and physical/topology validation |
| `OpenPinch/resources.py` | Closed list/read/load functions using `importlib.resources` |
| `OpenPinch/data/contracts/__init__.py` | Importable contract-resource package marker |
| `OpenPinch/data/contracts/hpr_performance_map/__init__.py` | Importable HPR resource package marker |
| `OpenPinch/data/contracts/hpr_performance_map/schema-1.0.json` | Generated alpha-1.0 JSON Schema |
| `OpenPinch/data/contracts/hpr_performance_map/heat-pump-1.0.json` | Canonical representative heating map |
| `OpenPinch/data/contracts/hpr_performance_map/refrigeration-1.0.json` | Canonical representative cooling map |
| `scripts/generate_hpr_performance_map_contract.py` | Explicit check/write generator for all three canonical resources |

`jsonschema>=4.25.1` was added only to the development dependency group and the
lockfile. No runtime dependency or root-package export was added.

## Tests and Packaging Changes

- `tests/contracts/test_hpr_performance_map.py`
- `tests/contracts/test_hpr_performance_map_properties.py`
- `tests/strategies/hpr_performance_maps.py`
- narrow additions to `tests/packaging/test_resources.py`,
  `tests/packaging/test_release_artifacts.py`,
  `tests/packaging/test_packaging_metadata.py`, and
  `tests/architecture/test_dependency_rules.py`
- installed-wheel resource validation in `scripts/artifact_install_smoke.py`

Documentation was added to `docs/api/schemas-and-config.rst`,
`docs/guides/input-formats-and-validation.rst`, and
`docs/reference/api-heat-pump.rst`.

## Implemented Contract Rules

- Schema version is exactly `1.0`; unknown fields and versions fail closed.
- Units are exactly degrees Celsius for source/sink temperatures and kilowatts
  for source duty, sink duty, and electric power.
- Heat pumps use `q_sink` and heating COP; refrigeration uses `q_source` and
  cooling COP.
- Every point satisfies `q_sink = q_source + electric_power`, useful duty equals
  `load_fraction * reference_capacity`, and COP equals useful duty divided by
  electric power within `energy_balance_tolerance`.
- Only active points are allowed: load fraction is in `(0, 1]` and electrical
  power and COP are positive.
- Names and coordinates are unique; each curve has one temperature pair;
  part-load points and the complete map use canonical order.
- Provenance is non-empty structured JSON with finite numbers and string object
  keys. Serialization returns detached mutable plain data.
- Request grids reject empty, duplicate, non-finite, and invalid load values and
  canonicalize valid coordinates deterministically.
- Resource generation uses sorted keys, two-space indentation, UTF-8, and one
  terminal newline. Check mode proves committed-byte equality.

## RED-GREEN-REFACTOR Evidence

| Slice | RED evidence | GREEN/refactor evidence |
|---|---|---|
| Contract examples | Tests initially could not import the absent specialist contract | Strict request/map implementation made field, physical, curve, provenance, and serialization examples pass |
| Generated properties | Reusable strategies and contract behaviors were absent | Fixed-seed round-trip, detachment, canonicalization, corruption, and ordering properties pass with shrinking enabled |
| Schema and fixtures | Generator and resource files were absent | Canonical generation, drift, size, and independent `jsonschema` validation pass |
| Package access | HPR resource catalog/read/load functions were absent | Closed catalog, detached loads, invalid-name/value, and installed-resource paths pass |
| Architecture/distribution | No Unit 1 dependency or artifact guarantees existed | Static dependency, cold-import, wheel/sdist inventory, byte-equality, and installed-wheel smokes pass |

## Verification Results

| Gate | Result |
|---|---|
| New contract/property/resource suite | 64 passed |
| Focused complete Unit 1 and existing HPR/API/architecture regression | 101 passed; one full-build case intentionally deselected for the focused run |
| Packaging metadata and source/wheel archive gate | 33 passed |
| Deterministic 10,000-point validation | Approximately 0.02 seconds; requirement is at most 2.0 seconds |
| New contract statement/branch coverage | 100 percent; every new resource-helper path also exercised |
| Resource regeneration | Check mode passed with byte equality |
| Dependency/lock audit | `jsonschema` proven development-only; offline lock check resolved 149 packages |
| Documentation | Sphinx warning-as-error build passed for all 55 source pages |
| Ruff and patch hygiene | Lint, format check, and `git diff --check` passed |
| Distribution | `openpinch-0.6.4-py3-none-any.whl` and `openpinch-0.6.4.tar.gz` built successfully |
| Artifact contents | Schema and both fixtures present and byte-identical in source, wheel, and source distribution; size limits pass |
| Installed-wheel smoke | Clean temporary environment loaded/validated schema and both maps and passed the existing OpenPinch workflow smoke |

The cold-import test blocks optional TESPy and external optimization packages.
A separate static rule proves the Unit 1 owner has no direct CoolProp import.
Blocking CoolProp from the top-level package is not a Unit 1 requirement because
CoolProp remains OpenPinch's required default HPR backend.

## Property-Based Testing Compliance

| Rule | Result |
|---|---|
| PBT-01 | Example/property balance covers contract examples plus generated map and request domains |
| PBT-02 | Reusable HPR map, request-coordinate, provenance, and recursive JSON strategies were added |
| PBT-03 | Energy, capacity, COP, canonical-order, uniqueness, and detachment invariants pass |
| PBT-04 | Request canonicalization is permutation-independent; repeated canonical generation is byte-stable |
| PBT-05 | Independent JSON Schema validation and direct physical equations provide separate oracles |
| PBT-06 | N/A: Unit 1 values are immutable and stateless |
| PBT-07 | Reusable strategies live in `tests/strategies/hpr_performance_maps.py` |
| PBT-08 | Fixed seed `20260905` passes with normal Hypothesis shrinking enabled |
| PBT-09 | Existing pytest/Hypothesis/CI integration is reused |
| PBT-10 | Example and property modules remain separately identifiable and required |

There are no blocking PBT findings. Security and Resiliency extensions remain
disabled. Their full rules were not applied; the approved fail-closed and
data-only contract requirements still pass.

## Compatibility and Scope Control

- `OpenPinch.__all__` remains exactly `PinchProblem` and `PinchWorkspace`.
- Existing HPR target contracts and targeting methods remain unchanged.
- No runtime HPR target object, CoolProp/TESPy simulator, OpenUtility type,
  Pyomo model, HiGHS solver, database, network service, or mutable cache was
  added to Unit 1.
- Existing brownfield resource and packaging owners were extended in place;
  no duplicate facade or parallel resource loader was created.
- The generated fixtures are contract examples, not claims of validated
  equipment performance.

## Deferred Unit 2 and Unit 3 Work

Unit 2 will implement the selectable simulator registry, keep CoolProp as the
default, add TESPy only through an optional adapter, normalize engine results,
generate complete maps from temperature/load grids, and test partial failure
and fake/guarded real-engine paths.

Unit 3 will tie map generation to the current vapour-compression heat-pump and
refrigeration targeting methods, expose deliberate specialist access, publish
maps and metadata, and complete end-to-end documentation. Downstream
optimization remains outside OpenPinch and consumes only the plain versioned
mapping.
