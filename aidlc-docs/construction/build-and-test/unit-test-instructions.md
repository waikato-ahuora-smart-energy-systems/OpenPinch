# Unit Test Execution - RTD and Comprehensive HPR Notebook

## Run Focused Tutorial and Documentation Contracts

```bash
uv run pytest --hypothesis-seed=20260715 -q \
  tests/packaging/test_notebooks.py \
  tests/packaging/test_tutorial_coverage.py \
  tests/packaging/test_docs_consistency.py \
  tests/packaging/test_resources.py
```

This gate validates notebook 09 source, compilation, specialist-import policy,
generator idempotence, 197/197 manifest coverage, RTD consistency, and packaged
resources.

## Run HPR Unit and Property Tests

```bash
uv run pytest --hypothesis-seed=20260715 -q \
  tests/contracts/test_hpr_performance_map.py \
  tests/contracts/test_hpr_performance_map_properties.py \
  tests/contracts/test_hpr_target_simulation_record.py \
  tests/analysis/heat_pumps
```

This selection covers the versioned plain-data contract, fluid categories,
CoolProp and TESPy adapters, map generation, thermodynamic validation,
targeting lifecycle, exact cache, winning records, basis compatibility, and
failure policy.

## Coverage Gate

```bash
uv run coverage run --branch --source=OpenPinch -m pytest \
  --hypothesis-seed=20260715 -q tests/analysis/heat_pumps tests/contracts
uv run coverage report --show-missing
```

The existing HPR acceptance gate remains at least 95 percent combined statement
and branch coverage over the map and target integration surface. The notebook
and RTD-only follow-up does not lower that threshold.

## Review Failures

1. Reproduce the first failure with seed `20260715`.
2. For a generated failure, retain the minimized Hypothesis example and normal
   shrinking behavior.
3. Correct the owning contract, pure coordinator, adapter, or integration
   layer without adding a fallback.
4. Rerun the focused file, the complete HPR selection, then the repository
   regression profile.
