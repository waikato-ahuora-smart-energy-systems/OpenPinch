# Unit Test Execution

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

The Unit 3 acceptance gate is at least 95 percent combined statement and branch
coverage over new target modules and materially changed target integration
paths. The completed code-generation measurement is 97 percent.

## Review Failures

1. Reproduce the first failure with seed `20260715`.
2. For a generated failure, retain the minimized Hypothesis example and normal
   shrinking behavior.
3. Correct the owning contract, pure coordinator, adapter, or integration
   layer without adding a fallback.
4. Rerun the focused file, the complete HPR selection, then the repository
   regression profile.
