# HPR and MVR Benchmark E2E Integration Test Instructions

## Purpose

Verify the interaction among public target accessors, HPR preprocessing,
CoolProp cycle evaluation, bounded optimization, detached result contracts,
state transactions, process-MVR components, downstream targeting, packaged
resources, and application-level APIs.

## Focused HPR and MVR Integration Gate

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/analysis/heat_pumps \
  tests/contracts/test_hpr.py \
  tests/contracts/test_hpr_reliability_contracts.py \
  tests/contracts/test_hpr_target_simulation_record.py \
  tests/application/test_coolprop_hpr_audit.py \
  tests/application/test_hpr_change_audit.py \
  tests/application/test_hpr_residual_workflow.py \
  tests/e2e/test_hpr_benchmark_helpers.py \
  tests/e2e/test_hpr_mvr.py \
  -m "not solver"
```

Expected behavior:

- public service inputs reach the declared HPR profile unchanged;
- each transaction either commits one valid target, makes a documented no-op,
  or leaves state unchanged with a typed failure;
- the six sentinels select the best viable objective observed within their
  bounded search;
- direct process MVR produces replacement streams and supports downstream
  direct heat integration; and
- all detached results remain finite and serializable.

## Complete Repository Integration Gate

```bash
uv run --no-sync coverage erase
uv run --no-sync coverage run --branch --source=OpenPinch \
  -m pytest --hypothesis-seed=20260715 -m "not solver"
uv run --no-sync coverage report --fail-under=95
```

The verified run collected 3,450 tests, deselected four external-solver tests,
and selected 3,446 tests. Results were:

- 3,443 passed;
- three optional notebook profiles skipped because `slow-hpr`, `solver`, and
  `interactive` were not enabled through `OPENPINCH_TUTORIAL_PROFILES`;
- four solver-marked tests deselected; and
- 96% branch-aware project coverage, above the 95% requirement.

## Cleanup

No server, database, network service, or persistent test fixture is started.
Generated `.coverage`, `docs/_build`, and `dist` outputs are repository-ignored
build artifacts and may be replaced by the next local build.
